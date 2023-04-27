import argparse
import os
from datetime import datetime
from pprint import pprint
from turtle import forward

import torch
import torch.nn as nn
import tqdm
from effdet.data import resolve_input_config
from PIL import Image
from timm.models import load_checkpoint
from timm.utils import AverageMeter, CheckpointSaver, get_outdir
from torchsummary import summary

from data import create_dataset, create_loader
from models.detector import DetBenchTrainImagePair
from models.models import FusionNet_New
from utils.evaluator import CocoEvaluator
from utils.utils import visualize_detections, visualize_target
from torch.utils.data import random_split


def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer not in name:
            module.eval()

def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name:
            param.requires_grad = False


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('--branch', default='fusion', type=str, metavar='BRANCH',
                        help='the inference branch ("thermal", "rgb", "fusion", or "single")')
    parser.add_argument('root', metavar='DIR',
                        help='path to dataset root')
    parser.add_argument('--dataset', default='flir_aligned', type=str, metavar='DATASET',
                        help='Name of dataset (default: "coco"')
    parser.add_argument('--split', default='val',
                        help='validation split')
    parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                        help='model architecture (default: tf_efficientdet_d1)')
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--img-size', default=None, type=int,
                        metavar='N', help='Input image dimension, uses model default if empty')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                        help='Image augmentation fill (background) color ("mean" or int)')
    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    parser.add_argument('--freeze-layer', default='fusion_cbam', type=str, choices=['fusion_cbam'])

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--wandb', action='store_true',
                        help='use wandb for logging and visualization')

    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = FusionNet_New(args.num_classes)

    # load checkpoint
    if args.checkpoint:
        
        checkpoint = torch.load(args.checkpoint)
        checkpoint_dict = checkpoint["state_dict"]
        
        net_dict = net.state_dict()
        
        new_checkpoint_dict = {k: v for k, v in checkpoint_dict.items() if k in net_dict}
        net_dict.update(new_checkpoint_dict)
        net.load_state_dict(net_dict) 
        
        print('Loaded checkpoint from ', args.checkpoint)
    
    
    training_bench = DetBenchTrainImagePair(net, create_labeler=True)

    freeze(training_bench, args.freeze_layer)
    
    total_trainable_params = sum(p.numel() for p in training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))

    training_bench.cuda()

    optimizer = torch.optim.Adam(training_bench.parameters(), lr=1e-3, weight_decay=0.0001)

    model_config = training_bench.config
    input_config = resolve_input_config(args, model_config)

    train_dataset, val_dataset = create_dataset(args.dataset, args.root)
    # generator1 = torch.Generator().manual_seed(42)
    # train_set, val_set = random_split(train_dataset,[0.8,0.2],generator=generator1)

    # print(train_dataset.img_ids)
    # print(val_set.img_ids)

   

    # validation_split = 0.2
    # shuffle_dataset = True
    # random_seed= 42

    # # Creating data indices for training and validation splits:
    # dataset_size = len(train_dataset)
    # indices = list(range(dataset_size))
    # split = int(np.floor(validation_split * dataset_size))
    # if shuffle_dataset :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)
    
    # train_indices, val_indices = indices[split:], indices[:split]

    # # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # val_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = create_loader(
        train_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem,
        is_training=True
        )
        # sampler=train_sampler)

    val_dataloader = create_loader(
        val_dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)
        # sampler=val_sampler)

    evaluator = CocoEvaluator(val_dataset, distributed=False, pred_yxyx=False)


    # set up checkpoint saver
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.save
    ])
    output_dir = get_outdir(output_base, 'train_test_offline', exp_name)
    saver = CheckpointSaver(
        net, optimizer, args=args, checkpoint_dir=output_dir)

    # logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='deep-sensor-fusion-new-comp_att',
          config=config
        )

    for epoch in range(1, args.epochs + 1):
        
        train_losses_m = AverageMeter()
        val_losses_m = AverageMeter()

        training_bench.train()
        set_eval_mode(training_bench, args.freeze_layer) 

        pbar = tqdm.tqdm(train_dataloader)
        for batch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

            output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False,branch=args.branch)
            loss = output['loss']
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
               visualize_target(train_dataset, target, wandb, args, 'train')
        
        training_bench.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(val_dataloader)
            for batch in tqdm.tqdm(val_dataloader):
                pbar.set_description('Validating...')
                thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

                output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=True)
                loss = output['loss']
                val_losses_m.update(loss.item(), thermal_img_tensor.size(0))
                evaluator.add_predictions(output['detections'], target)
                if args.wandb and epoch == args.epochs:
                    visualize_detections(val_dataset, output['detections'], target, wandb, args, 'val')

        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=evaluator.evaluate())
