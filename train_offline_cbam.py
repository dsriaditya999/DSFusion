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
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)

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
    parser.add_argument('--rgb_checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--thermal_checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--head', default='', type=str, metavar='HEAD',
                        help='the inference head ("thermal", "rgb", "fusion", or "single")')
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

    # Load All Checkpoints
    if args.rgb_checkpoint != '' and args.thermal_checkpoint != '':

        rgb_checkpoint = torch.load(args.rgb_checkpoint)

        # Load RGB Backbone and FPN

        rgb_checkpoint_dict = deepcopy(rgb_checkpoint["state_dict"])

        for key in rgb_checkpoint["state_dict"].keys():
            if "backbone" in key or "fpn" in key:
                rgb_checkpoint_dict["rgb_"+key] = rgb_checkpoint["state_dict"][key]
            
            del rgb_checkpoint_dict[key]
            

        # Load Thermal Backbone and FPN

        thermal_checkpoint = torch.load(args.thermal_checkpoint)

        thermal_checkpoint_dict = deepcopy(thermal_checkpoint["state_dict"])

        for key in thermal_checkpoint["state_dict"].keys():
            if "backbone" in key or "fpn" in key:
                thermal_checkpoint_dict["rgb_"+key] = thermal_checkpoint["state_dict"][key]

            del thermal_checkpoint_dict[key]

        # Load Class and Head Weights

        if args.head == "rgb":

            fusion_checkpoint_dict =  deepcopy(rgb_checkpoint["state_dict"])

            for key in rgb_checkpoint["state_dict"].keys():

                if "class_net" in key or "box_net" in key:
                    fusion_checkpoint_dict["fusion"+key] = rgb_checkpoint["state_dict"][key]

                del fusion_checkpoint_dict[key]

        elif args.head == "thermal":

            fusion_checkpoint_dict =  deepcopy(thermal_checkpoint["state_dict"])

            for key in thermal_checkpoint["state_dict"].keys():

                if "class_net" in key or "box_net" in key:
                    fusion_checkpoint_dict["fusion"+key] = thermal_checkpoint["state_dict"][key]

                del fusion_checkpoint_dict[key]

        else:
            raise ValueError("Invalid Head")
        
        

        # Assert that checkpoint dicts are not empty

        assert len(rgb_checkpoint_dict.keys()) > 0
        assert len(thermal_checkpoint_dict.keys()) > 0
        assert len(fusion_checkpoint_dict.keys()) > 0


        net.rgb_backbone.load_state_dict({k.split(".", 1)[1]: v for k, v in rgb_checkpoint_dict.items() if k.split(".", 1)[1] in net.rgb_backbone.state_dict()})
        net.rgb_fpn.load_state_dict({k.split(".", 1)[1]: v for k, v in rgb_checkpoint_dict.items() if k.split(".", 1)[1] in net.rgb_fpn.state_dict()})

        net.thermal_backbone.load_state_dict({k.split(".", 1)[1]: v for k, v in thermal_checkpoint_dict.items() if k.split(".", 1)[1] in net.thermal_backbone.state_dict()})
        net.thermal_fpn.load_state_dict({k.split(".", 1)[1]: v for k, v in thermal_checkpoint_dict.items() if k.split(".", 1)[1] in net.thermal_fpn.state_dict()})

        net.fusion_class_net.load_state_dict({k.split(".", 1)[1]: v for k, v in fusion_checkpoint_dict.items() if k.split(".", 1)[1] in net.fusion_class_net.state_dict()})
        net.fusion_box_net.load_state_dict({k.split(".", 1)[1]: v for k, v in fusion_checkpoint_dict.items() if k.split(".", 1)[1] in net.fusion_box_net.state_dict()})


        # Print Loaded Checkpoint

        print("Loaded RGB Checkpoint from ", args.rgb_checkpoint)
        print("Loaded Thermal Checkpoint from ", args.thermal_checkpoint)
        print("Using ", args.head, " Head")


    else:

        print("No Checkpoint Loaded")
    
    training_bench = DetBenchTrainImagePair(net, create_labeler=True)

    freeze(training_bench, args.freeze_layer)
    
    full_backbone_params = count_parameters(training_bench.model.thermal_backbone) + count_parameters(training_bench.model.rgb_backbone)
    head_net_params = count_parameters(training_bench.model.fusion_class_net) + count_parameters(training_bench.model.fusion_box_net)
    bifpn_params = count_parameters(training_bench.model.rgb_fpn) + count_parameters(training_bench.model.thermal_fpn)
    full_params = count_parameters(training_bench.model)
    fusion_net_params = count_parameters(training_bench.model.fusion_cbam0)+count_parameters(training_bench.model.fusion_cbam1)+count_parameters(training_bench.model.fusion_cbam2)+count_parameters(training_bench.model.fusion_cbam3)+count_parameters(training_bench.model.fusion_cbam4)


    print("*"*50)
    print("Backbone Params : {}".format(full_backbone_params) )
    print("Head Network Params : {}".format(head_net_params) )
    print("BiFPN Params : {}".format(bifpn_params) )
    print("Fusion Nets Params : {}".format(fusion_net_params) )
    print("Total Model Parameters : {}".format(full_params) )
    total_trainable_params = sum(p.numel() for p in training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))
    print("*"*50)

    training_bench.cuda()

    optimizer = torch.optim.Adam(training_bench.parameters(), lr=1e-3, weight_decay=0.0001)

    model_config = training_bench.config
    input_config = resolve_input_config(args, model_config)

    train_dataset, val_dataset = create_dataset(args.dataset, args.root)

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

    evaluator = CocoEvaluator(val_dataset, distributed=False, pred_yxyx=False)


    # set up checkpoint saver
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.save
    ])
    output_dir = get_outdir(output_base, 'final_train_offline', exp_name)
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

    train_loss = []
    val_loss = []


    for epoch in range(1, args.epochs + 1):
        
        train_losses_m = AverageMeter()
        val_losses_m = AverageMeter()

        training_bench.train()
        set_eval_mode(training_bench, args.freeze_layer) 

        pbar = tqdm.tqdm(train_dataloader)
        batch_train_loss = []
        for batch in pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

            output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=False)
            loss = output['loss']
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            batch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.wandb:
               visualize_target(train_dataset, target, wandb, args, 'train')

        train_loss.append(sum(batch_train_loss)/len(batch_train_loss))
        
        training_bench.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(val_dataloader)
            batch_val_loss = []
            for batch in tqdm.tqdm(val_dataloader):
                pbar.set_description('Validating...')
                thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]

                output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=True)
                loss = output['loss']
                val_losses_m.update(loss.item(), thermal_img_tensor.size(0))
                batch_val_loss.append(loss.item())
                evaluator.add_predictions(output['detections'], target)
                if args.wandb and epoch == args.epochs:
                    visualize_detections(val_dataset, output['detections'], target, wandb, args, 'val')

            val_loss.append(sum(batch_val_loss)/len(batch_val_loss))

        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch=epoch, metric=evaluator.evaluate())

    # Plotting the training and validation loss curves and saving the plot

    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir,'loss_plot.png'))

