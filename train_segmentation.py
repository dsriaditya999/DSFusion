import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F

import tqdm
from data.cityscapes import CityScapes, classes
from data.kaist_segmentation import KAISTSegmentation
from models.segmentation_models import SegmentationModel
from data.mfn_segmentation import MF_dataset, mf_classes

from torch.utils.data import DataLoader

from ignite.metrics import ConfusionMatrix, IoU, mIoU

from utils.lr import PolynomialLR

def labels(num_classes):
    labels = {}
    for i, cls in enumerate(classes):
        if cls.train_id < 255 and cls.train_id >= 0:
            labels[cls.train_id] = cls.name
    assert len(labels) == num_classes, (len(labels), num_classes)
    return labels

def print_class_iou(ious, num_classes):
    format_string = "{:<20} {:.3f}"
    line = ''.join(['-']*30)
    for i, name in labels(num_classes).items():
        if i != 255:
            print_string = format_string.format(name, ious[i])
            print(line)
            print(print_string)
    print(line)
    print()
    print(format_string.format('mIoU', ious.mean()))

def wb_mask(bg_img, pred_mask, true_mask, num_classes):
    return wandb.Image(bg_img, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : labels(num_classes)},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : labels(num_classes)}})

def renormalize(img):
    img *= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img += np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    img = np.clip(255*img, 0, 255)
    img = np.uint8(img)
    return img


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

    parser.add_argument('root', metavar='DIR', help='path to dataset root')
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'kaist', 'mfn'])

    parser.add_argument('--model', '-m', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--branch', type=str, choices=['rgb', 'thermal', 'fusion'])

    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--img-size', default=768, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--pin-memory', action='store_true')

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    
    parser.add_argument('--output', default='', type=str)

    parser.add_argument('--cityscapes-split', nargs='*', choices=['fine', 'coarse'])
    parser.add_argument('--pretrained-weights', default=None, type=str)
    parser.add_argument('--freeze-encoder', action='store_true')

    parser.add_argument('--use-amp', action='store_true')

    parser.add_argument('--wandb', action='store_true',
                        help='use wandb for logging and visualization')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu))

    ### Create network, load weights, send to gpu
    net = SegmentationModel(args)
    
    if args.pretrained_weights is not None:
        print('loading weights from {}'.format(args.pretrained_weights))
        weights = torch.load(args.pretrained_weights, map_location=device)
        net.load_state_dict(weights, strict=False)

    net.to(device)     

    if args.freeze_encoder:
        for param in net.model.encoder.parameters():
            param.requires_grad = False

    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
    scheduler = PolynomialLR(optimizer, max_iter=args.epochs, decay_iter=1, gamma=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    ### Dataloading
    train_dataset, val_dataset = None, None
    if args.dataset == 'cityscapes':    
        train_dataset = CityScapes(args.root, split='train', mode=args.cityscapes_split, img_size=args.img_size)
        val_dataset = CityScapes(args.root, split='val', mode=['fine'], img_size=args.img_size)
    elif args.dataset == 'kaist':
        if args.branch == 'rgb':
            train_dataset = KAISTSegmentation(args.root, split='train', img_size=args.img_size)
            val_dataset = KAISTSegmentation(args.root, split='val', img_size=args.img_size)
        elif args.branch == 'thermal':
            train_dataset = KAISTSegmentation(args.root, split='train', img_size=args.img_size, is_thermal = True)
            val_dataset = KAISTSegmentation(args.root, split='val', img_size=args.img_size, is_thermal = True)     
    elif args.dataset == 'mfn':
        classes = mf_classes
        if args.branch == 'rgb':
            train_dataset = MF_dataset(args.root, split='train', have_label=True, is_thermal=False)
            val_dataset = MF_dataset(args.root, split='val', have_label=True, is_thermal=False)
        elif args.branch == 'thermal':
            train_dataset = MF_dataset(args.root, split='train', have_label=True, is_thermal=True)
            val_dataset = MF_dataset(args.root, split='val', have_label=True, is_thermal=True)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=min(4, args.batch_size), num_workers=min(4, args.num_workers), pin_memory=args.pin_memory, shuffle=False)

    ### Set up checkpoint saver
    if args.dataset == 'cityscapes':
        weights_path = 'weights'
        finetune = '-' + '-'.join(args.cityscapes_split)
        exp_name = '{}-{}-{}{}'.format(args.model, args.dataset, datetime.now().strftime("%Y%m%d-%H%M%S"), finetune)
        output_dir = os.path.join(weights_path, exp_name)
        os.makedirs(output_dir, exist_ok=True)
    elif args.dataset == 'kaist':
        weights_path = 'weights'
        exp_name = '{}-{}-{}-{}'.format(args.model, args.dataset, args.branch, datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_dir = os.path.join(weights_path, exp_name)
        os.makedirs(output_dir, exist_ok=True)
    elif args.dataset == 'mfn':
        weights_path = 'weights'
        exp_name = '{}-{}-{}-{}'.format(args.model, args.dataset, args.branch, datetime.now().strftime("%Y%m%d-%H%M%S"))
        output_dir = os.path.join(weights_path, exp_name)
        os.makedirs(output_dir, exist_ok=True)

    ### Setup logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
            project='deep-sensor-fusion-segmentation-lr-scheduler',
            config=config
        )
    cm = ConfusionMatrix(num_classes=args.num_classes, device=device)
    val_iou_metric = IoU(cm)

    best_val_loss = np.inf
    for epoch in range(1, args.epochs + 1):

        net.train()

        pbar = tqdm.tqdm(train_loader)
        for i, batch in enumerate(pbar):
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))
            
            imgs, masks = batch[0].to(device), batch[1].to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                pred_logits = net(imgs)

                train_loss = F.cross_entropy(pred_logits, masks, reduction='mean', ignore_index=-1)
                
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
            if args.wandb and i & 10 == 0:
                bg_img = renormalize(imgs[0].cpu().numpy())
                bg_img = bg_img.transpose(1, 2, 0)

                _, pred_masks = torch.max(pred_logits, dim=1)
                wandb.log({
                    'train batch loss' : train_loss.item(), 
                    "epoch" : epoch, 
                    "train predictions" : wb_mask(bg_img, pred_masks[0].cpu().numpy(), masks[0].cpu().numpy().astype(np.uint8), args.num_classes) 
                })

        if epoch % 1 == 0:
            torch.save(net.state_dict(), os.path.join(output_dir, 'weights-{}.pt'.format(str(epoch).zfill(4))))

        net.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(val_loader)
            val_loss_total = 0
            samples = 0
            for i, batch in enumerate(pbar):
                pbar.set_description('Validating...')
                imgs, masks = batch[0].to(device), batch[1].to(device)
                pred_logits = net(imgs)

                val_loss = F.cross_entropy(pred_logits, masks, reduction='sum', ignore_index=-1)
                val_loss_total += val_loss.item()
                samples += imgs.shape[0]

                val_iou_metric.update((pred_logits, masks))
                if args.wandb and i & 10 == 0:
                    bg_img = renormalize(imgs[0].cpu().numpy())
                    bg_img = bg_img.transpose(1, 2, 0)

                    _, pred_masks = torch.max(pred_logits, dim=1)
                    pred_mask = pred_masks[0].cpu().detach().numpy().astype(np.uint8)
                    wandb.log({
                        'val batch loss' : val_loss.item(), 
                        "epoch" : epoch, 
                        "val predictions" : wb_mask(bg_img, pred_mask, masks[0].cpu().numpy().astype(np.uint8), args.num_classes) 
                    })

            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]

            iou = val_iou_metric.compute()
            print_class_iou(iou, args.num_classes)
            miou = iou.mean()
            val_iou_metric.reset()

            total_val_loss = val_loss_total / samples
            if args.wandb:
                wandb.log({
                    'epoch val loss' : total_val_loss,
                    'epoch mIoU' : miou,
                    'learning rate' : last_lr,
                })
            if total_val_loss < best_val_loss:
                torch.save(net.state_dict(), os.path.join(output_dir, 'best.pt'.format(str(epoch).zfill(4))))
                best_val_loss = total_val_loss
