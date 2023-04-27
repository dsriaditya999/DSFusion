import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import FPN, DeepLabV3Plus

from models.fusion_modules import CBAMLayer


class SegmentationModel(nn.Module):

    def __init__(self, args, pretrain_ssl=False):
        super(SegmentationModel, self).__init__()

        aux_params = None
        self.pretrain_simsiam = pretrain_ssl
        if self.pretrain_simsiam:
            aux_params = dict(classes=2048)
            self.simsiam_projection = nn.Sequential(
                nn.Linear(2048, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True), # first layer
                nn.Linear(2048, 2048, bias=False),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=True), # second layer
                nn.Linear(2048, 2048, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(2048, affine=False)
            ) 

            self.simsiam_prediction = nn.Sequential(
                nn.Linear(2048, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True), # hidden layer
                nn.Linear(512, 2048)
            )

        self.model = DeepLabV3Plus(
            encoder_name=args.model, 
            in_channels=3, 
            classes=args.num_classes, 
            aux_params=aux_params, 
            encoder_weights='imagenet')

    def forward_classification_head_and_project(self, x):
        features = self.model.encoder(x)
        x = self.model.classification_head(features[-1])
        x = self.simsiam_projection(x)
        return x

    def forward_simsiam(self, x):
        x1, x2 = torch.chunk(x, 2)

        z1 = self.forward_classification_head_and_project(x1)
        z2 = self.forward_classification_head_and_project(x2)
        p1 = self.simsiam_prediction(z1)
        p2 = self.simsiam_prediction(z2)

        return p1, p2, z1.detach(), z2.detach()

    def forward(self, x):
        
        if self.pretrain_simsiam:
            output = self.forward_simsiam(x)
        else:
            output = self.model(x)
        
        return output