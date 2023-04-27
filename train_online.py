import argparse
import os
from datetime import datetime
from pprint import pprint
from turtle import forward
from PIL import Image
import numpy as np
import random
import cv2
import tqdm
from torchsummary import summary
from timm.models import load_checkpoint
from timm.utils import CheckpointSaver, get_outdir, AverageMeter

import torch
import torch.nn as nn
import torch.nn.functional as T

from models.models import FusionNet
from models.detector import DetBenchTrainImagePair

from effdet.data import resolve_input_config
from data import create_dataset, create_loader
from data.dataset import OnlineTrainingDataset
from utils.evaluator import CocoEvaluator
from utils.utils import visualize_detections, visualize_target, detections_to_self_label
from utils.data_aug import get_affine, get_affine_inv


def set_eval_mode(network, freeze_layer):
    for name, module in network.named_modules():
        if freeze_layer not in name:
            module.eval()

def freeze(network, freeze_layer):
    for name, param in network.named_parameters():
        if freeze_layer not in name:
            param.requires_grad = False

def multiscale_samples(tensor, sample_size=4):
    # generate random params
    params = np.zeros((sample_size, 5))
    tensors = torch.zeros(sample_size, 3, tensor.shape[2], tensor.shape[3]).to(tensor.device)
    for i in range(sample_size):
        params[i, 0] = random.randint(-tensor.shape[2] / 2, tensor.shape[2] / 2) # dx
        params[i, 1] = random.randint(-tensor.shape[3] / 2, tensor.shape[3] / 2) # dy
        params[i, 2] = 0 # alpha
        params[i, 3] = random.uniform(0.5, 1.0) # scale
        params[i, 4] = 1 if random.uniform(0, 1) < 0.5 else -1 # flip
        
        # params[i, 0] = -50 # dx
        # params[i, 1] = -100 # dy
        # params[i, 2] = 0 # alpha
        # params[i, 3] = 2 # scale
        # params[i, 4] = -1 # flip

        tensors[i] = tensor
    
    # warp images
    affine = get_affine(params)
    tensor_samples = warp_images(tensors, affine)
    affine_inv = get_affine_inv(affine, params)
    return tensor_samples, affine, affine_inv

def warp_images(tensors, affine):
    build_grid = T.affine_grid(affine, tensors.shape).to(tensors.device)
    warp_tensors = T.grid_sample(tensors, build_grid)
    return warp_tensors

def warp_detections(detections, affine, target_scales=None):
    B, num_bboxes, _ = detections.shape
    if target_scales is None:
        target_scales = torch.ones(B).to(detections.device)

    affine = affine.to(detections.device)

    for img_dets, warp_matrix, target_scale in zip(detections, affine, target_scales):
        # homogeneous coordinates

        # detected coordinates are in original image coordinate frame (i.e. img = cv2.imread(...)). 
        # Rescale to network input size.
        xs = img_dets[:, [0, 0, 2, 2]].view(num_bboxes * 4) / target_scale
        ys = img_dets[:, [1, 3, 3, 1]].view(num_bboxes * 4) / target_scale
        
        ### TODO: remove the hardcoding
        center_x = 768 / 2
        center_y = 768 / 2

        ### Bounding boxes are in u,v coordinate frame with origin at top left. 
        ### Shift the origin of the coordinate frame to image center. 
        xs = xs - center_x
        ys = center_y - ys 

        ones = torch.ones_like(xs).to(detections.device)
        points = torch.vstack([xs, ys, ones])
        
        ### the scale in transformation matrix assumes no cropping is done. manually account for it using zoom_factor
        ### TODO: unclear what is going on with the flip. without the flip, there may be incorrect x-axis offset. 
        zoom_factor = torch.abs(warp_matrix[0, 0])
        flip = torch.sign(warp_matrix[0, 0])
        ### Affine matrix is normalized w.r.t. the network input size. 
        ### Unnormalize it prior to transform. 
        ### See https://stackoverflow.com/questions/66987451/image-translation-in-pytorch-using-affine-grid-grid-sample-functions
        warp_matrix[0, 2] *= -flip*center_x * zoom_factor
        warp_matrix[1, 2] *= center_y * zoom_factor

        warp_points = warp_matrix @ points
        
        ### Bounding boxes are centered at origin. Shift back to u,v coordinate frame. 
        warp_points[0,:] = center_x + warp_points[0,:] / zoom_factor**2
        warp_points[1,:] = center_y - warp_points[1,:] / zoom_factor**2

        xs = warp_points[0].view(num_bboxes, 4)
        ys = warp_points[1].view(num_bboxes, 4)
        
        warp_bboxes = torch.t(torch.vstack(
                (torch.min(xs, 1).values, torch.min(ys, 1).values, torch.max(xs, 1).values, torch.max(ys, 1).values)))
        img_dets[:, 0:4] = warp_bboxes


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
    parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                        help='Override num_classes in model config if set. For fine-tuning from pretrained.')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    
    parser.add_argument('--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')

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

    parser.add_argument('--freeze-layer', default='fusion', type=str, choices=['fusion', 'fusion_cbam'])

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--init-fusion-head-weights', type=str, default=None, choices=['thermal', 'rgb', None])
    parser.add_argument('--thermal-checkpoint-path', type=str)
    parser.add_argument('--output', default='', type=str, metavar='PATH',
                        help='path to output folder (default: none, current dir)')
    parser.add_argument('--wandb', action='store_true',
                        help='use wandb for logging and visualization')

    # for online training
    parser.add_argument('--sample-size', default=4, type=int,
                        metavar='N', help='multiscale samples per image (default: 4)')
    parser.add_argument('--buffer-size', default=8, type=int,
                        metavar='N', help='online training data buffer size (default: 8)')
    parser.add_argument('--score-threshold', default=0.7, type=float)

    args = parser.parse_args()
    args.prefetcher = not args.no_prefetcher
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = FusionNet(args.num_classes, args.thermal_checkpoint_path, args)
    training_bench = DetBenchTrainImagePair(net, create_labeler=True)

    freeze(training_bench, args.freeze_layer)
    
    total_trainable_params = sum(p.numel() for p in training_bench.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in training_bench.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))

    training_bench.cuda()

    optimizer = torch.optim.Adam(training_bench.parameters(), lr=1e-3, weight_decay=0.0001)

    model_config = training_bench.config
    input_config = resolve_input_config(args, model_config)

    _, val_dataset = create_dataset(args.dataset, args.root)
    val_dataloader = create_loader(
        val_dataset,
        input_size=input_config['input_size'],
        batch_size=1,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    evaluator = CocoEvaluator(val_dataset, distributed=False, pred_yxyx=False)

    # load checkpoint
    if args.checkpoint:
        load_checkpoint(net, args.checkpoint)
        print('Loaded checkpoint from ', args.checkpoint)

    # set up checkpoint saver
    output_base = args.output if args.output else './output'
    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        args.model
    ])
    output_dir = get_outdir(output_base, 'train_online', exp_name)
    saver = CheckpointSaver(
        net, optimizer, args=args, checkpoint_dir=output_dir)

    # logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
          project='deep-sensor-fusion',
          config=config
        )

    
    train_losses_m = AverageMeter()
    val_losses_m = AverageMeter()

    online_training_data = OnlineTrainingDataset(args.buffer_size)
    pbar = tqdm.tqdm(val_dataloader)
    for iter_i, batch in enumerate(pbar):
        pbar.set_description('Iter {}/{}'.format(iter_i, len(pbar)))

        # self-supervised augmentation consistency
        thermal_img_tensor, rgb_img_tensor, target = batch[0], batch[1], batch[2]
        
        thermal_img_tensor_samples, thermal_affine, thermal_affine_inv = multiscale_samples(thermal_img_tensor)
        rgb_img_tensor_samples, rgb_affine, rgb_affine_inv = multiscale_samples(rgb_img_tensor)
        
        thermal_img_tensor_unwarped = warp_images(thermal_img_tensor_samples, thermal_affine_inv)

        # get current self label
        training_bench.eval()
        with torch.no_grad():
            target['img_scale'] = torch.cat((target['img_scale'], target['img_scale'], target['img_scale'], target['img_scale']), dim=0)
            target['img_size'] = torch.cat((target['img_size'], target['img_size'], target['img_size'], target['img_size']), dim=0)

            ### Note: in training_bench, detections have been automatically 
            ### rescaled into original dataset image coordinates (i.e for FLIR, it is 640x512)
            ### to allow for comparison to ground truth bboxes. 
            thermal_output = training_bench(thermal_img_tensor_samples, rgb_img_tensor_samples, target, eval_pass=True, branch='thermal')
            rgb_output = training_bench(thermal_img_tensor_samples, rgb_img_tensor_samples, target, eval_pass=True, branch='rgb')
            
            ######################################################
            ### TODO: remove debugging code ######################
            ######################################################

            # plot warped image detections on network input
            bboxes = thermal_output['detections'].cpu().numpy()
            i = 0
            for img, bbox, scale in zip(thermal_img_tensor_samples, bboxes, target['img_scale'].cpu().numpy()):
                img = img.cpu().numpy()
                img -= img.min()
                img /= img.max()
                img = np.clip(img, 0, 1)
                img_bgr = cv2.cvtColor(np.uint8(255*img.transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)

                ### display image on 768x768 so rescale
                bbox[:,:4] /= scale
                for box in bbox:
                    x1, y1, x2, y2 = box[:4].astype(int) 
                    if box[4] > 0.6:
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite('image_{}.png'.format(str(i).zfill(4)), img_bgr)
                i += 1


            # plot unwarped image detections on unwarped network input
            warp_detections(thermal_output['detections'], thermal_affine_inv, target_scales=target['img_scale'])
            bboxes = thermal_output['detections'].cpu().numpy()
            i = 0
            for img, bbox, scale in zip(thermal_img_tensor_unwarped, bboxes, target['img_scale'].cpu().numpy()):
                img = img.cpu().numpy()
                img -= img.min()
                img /= img.max()
                img = np.clip(img, 0, 1)
                img_bgr = cv2.cvtColor(np.uint8(255*img.transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                for box in bbox:
                    x1, y1, x2, y2 = box[:4].astype(int)
                    if box[4] > 0.6:
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite('image_{}_warped.png'.format(str(i).zfill(4)), img_bgr)
                i += 1

            exit(0)

            ######################################################
            # End debugging code #################################
            ######################################################


            #self_label = detections_to_self_label(
            #    thermal_output['detections'], rgb_output['detections'], 
            #    target, score_threshold=args.score_threshold
            #)
        
        if args.wandb:
            visualize_detections(val_dataset, thermal_output['detections'], target, wandb, args, 'val/warped', score_threshold=0.5, img_tensor=thermal_img_tensor_samples)
            warp_detections(thermal_output['detections'], thermal_affine_inv)
            visualize_detections(val_dataset, thermal_output['detections'], target, wandb, args, 'val/unwarped', score_threshold=0.5, img_tensor=thermal_img_tensor_unwarped)
            #visualize_detections(val_dataset, rgb_output['detections'], target, wandb, args, 'val', score_threshold=0.5, img_tensor=rgb_img_tensor_samples)
            #visualize_target(val_dataset, self_label, wandb, args, 'val/self_label')
        #online_training_data.add_current_data(thermal_img_tensor, rgb_img_tensor, self_label)
        
        # online training
        '''
        thermal_img_train, rgb_img_train, target_train = online_training_data.get_training_data()
        training_bench.eval()
        set_eval_mode(training_bench, args.freeze_layer)
        for epoch in range(1, args.epochs + 1):
            output = training_bench(thermal_img_train, rgb_img_train, target_train, eval_pass=False)
            loss = output['loss']
            train_losses_m.update(loss.item(), thermal_img_tensor.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # get current detections
        training_bench.eval()
        with torch.no_grad():
            output = training_bench(thermal_img_tensor, rgb_img_tensor, target, eval_pass=True)
            evaluator.add_predictions(output['detections'], target)
            if args.wandb:
                visualize_detections(val_dataset, output['detections'], target, wandb, args, 'val', score_threshold=0.1)
        '''

    if saver is not None:
        best_metric, best_epoch = saver.save_checkpoint(epoch=args.epochs, metric=evaluator.evaluate())
