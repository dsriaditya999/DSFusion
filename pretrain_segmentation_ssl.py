import argparse
import os
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import tqdm
from data.cityscapes import CityScapes, classes
from data.kaist_segmentation import KAISTSegmentation
from models.segmentation_models import SegmentationModel

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import ImageFilter

def renormalize(img):
    img *= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img += np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)

    img = np.clip(255*img, 0, 255)
    img = np.uint8(img)
    return img

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

    parser.add_argument('root', metavar='DIR', help='path to dataset root')

    parser.add_argument('--model', '-m', default='resnet50')
    parser.add_argument('--num-classes', type=int, default=None)

    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--img-size', default=360, type=int)

    parser.add_argument('--pin-memory', action='store_true')

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--wandb', action='store_true',
                        help='use wandb for logging and visualization')

    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu))

    net = SegmentationModel(args, pretrain_ssl=True)
    
    total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in net.parameters())
    print('Total Parameters: {:,} \nTotal Trainable: {:,}\n'.format(total_params, total_trainable_params))

    # optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0001)
    init_lr = 1e-3
    optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, weight_decay=0.0001, momentum=0.9)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    train_dataset = datasets.ImageFolder(
        args.root,
        TwoCropsTransform(transforms.Compose(augmentation))
    )


    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory, shuffle=True, drop_last=True)

    net.to(device)        

    # set up checkpoint saver
    weights_path = 'weights'
    exp_name = '{}-{}-{}{}'.format(args.model, 'flirV2_kaist3283', datetime.now().strftime("%Y%m%d-%H%M%S"), '-pretrain-ssl')
    output_dir = os.path.join(weights_path, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # logging
    if args.wandb:
        import wandb
        config = dict()
        config.update({arg: getattr(args, arg) for arg in vars(args)})
        wandb.init(
            project='deep-sensor-fusion-segmentation-pretrain-ssl',
            config=config
        )

    criterion = nn.CosineSimilarity(dim=1)
    for epoch in range(1, args.epochs + 1):
        
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        net.train()
        pbar = tqdm.tqdm(train_loader)
        for i, (batch, _) in enumerate(pbar):
            pbar.set_description('Epoch {}/{}'.format(epoch, args.epochs + 1))

            imgs_1, imgs_2 = batch[0].to(device), batch[1].to(device)
            imgs = torch.cat([imgs_1, imgs_2], dim=0)
            p1, p2, z1, z2 = net(imgs)

            train_loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if args.wandb and i & 10 == 0:
                bg_img1 = renormalize(imgs_1[0].cpu().numpy()).transpose(1, 2, 0)
                bg_img2 = renormalize(imgs_2[0].cpu().numpy()).transpose(1, 2, 0)
                wandb.log({
                    'image1' : wandb.Image(bg_img1),
                    'image2' : wandb.Image(bg_img2),
                    'train batch loss' : train_loss.item(), 
                    "epoch" : epoch, 
                })

        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(output_dir, 'weights-{}.pt'.format(str(epoch).zfill(4))))
