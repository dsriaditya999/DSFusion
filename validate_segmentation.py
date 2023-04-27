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
    parser.add_argument('--dataset', type=str, choices=['cityscapes', 'kaist'])

    parser.add_argument('--model', '-m', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=None)

    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--img-size', default=2048, type=int)

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--pretrained-weights', default=None, type=str)

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu))

    ### Create network, load weights, send to gpu
    net = SegmentationModel(args)
    
    if args.pretrained_weights is not None:
        print('loading weights from {}'.format(args.pretrained_weights))
        weights = torch.load(args.pretrained_weights)
        net.load_state_dict(weights, strict=True)

    net.to(device)     
    net.eval()


    ### Dataloading
    val_dataset = None
    if args.dataset == 'cityscapes':
            val_dataset = CityScapes(args.root, split='val', mode=['fine'], img_size=args.img_size)
    elif args.dataset == 'kaist':
        val_dataset = KAISTSegmentation(args.root, split='val', img_size=args.img_size)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    cm = ConfusionMatrix(num_classes=args.num_classes, device=device)
    val_iou_metric = IoU(cm)

    with torch.no_grad():
        pbar = tqdm.tqdm(val_loader)
        val_loss_total = 0
        samples = 0
        for i, batch in enumerate(pbar):
            imgs, masks = batch[0].to(device), batch[1].to(device)
            pred_logits = net(imgs)

            val_iou_metric.update((pred_logits, masks))

        iou = val_iou_metric.compute()
        print_class_iou(iou, args.num_classes)
        miou = iou.mean()
        val_iou_metric.reset()

