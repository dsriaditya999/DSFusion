import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import effdet
from effdet import EfficientDet
from effdet.efficientdet import get_feature_info
from effdet import DetBenchTrain

from models.fusion_modules import CBAMLayer
import math

class FusionNet(nn.Module):

    def __init__(self, thermal_checkpoint_path, args):
        super(FusionNet, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = num_classes

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)
        effdet.helpers.load_checkpoint(thermal_det, thermal_checkpoint_path)
        
        if args.rgb_checkpoint_path:
            effdet.helpers.load_checkpoint(rgb_det, args.rgb_checkpoint_path)
            print('Loading RGB from {}'.format(args.rgb_checkpoint_path))
        else:
            effdet.helpers.load_pretrained(rgb_det, self.config.url)
            print('Loading RGB from {}'.format(self.config))
        
        self.thermal_backbone = thermal_det.backbone
        self.thermal_fpn = thermal_det.fpn
        self.thermal_class_net = thermal_det.class_net
        self.thermal_box_net = thermal_det.box_net

        self.rgb_backbone = rgb_det.backbone
        self.rgb_fpn = rgb_det.fpn
        self.rgb_class_net = rgb_det.class_net
        self.rgb_box_net = rgb_det.box_net

        fusion_det = EfficientDet(self.config)
        self.fusion_fpn = fusion_det.fpn
        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        if args.init_fusion_head_weights == 'thermal':
            effdet.helpers.load_checkpoint(fusion_det, thermal_checkpoint_path) # This is optional
        elif args.init_fusion_head_weights == 'rgb':
            effdet.helpers.load_pretrained(fusion_det, self.config.url)
        else:
            print('Fusion head random init.')

        feature_info = get_feature_info(self.thermal_backbone)
        for level in range(self.config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                self.add_module(f'fusion_cbam{level}', CBAMLayer(2*in_chs))

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        fpn = getattr(self, f'{branch}_fpn')
        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')
        
        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)
            feats = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                cbam = getattr(self, f'fusion_cbam{i}')
                feats.append(cbam(x))
        else:
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
        
        x = fpn(feats)
        x_class = class_net(x)
        x_box = box_net(x)

        return x_class, x_box


##################################### New Def Fusion Net###############################################

class Att_FusionNet(nn.Module):

    def __init__(self, args):
        super(Att_FusionNet, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = args.num_classes
        self.config.image_size = (1280, 1280)

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)

        if args.thermal_checkpoint_path:
            effdet.helpers.load_checkpoint(thermal_det, args.thermal_checkpoint_path)
            print('Loading Thermal from {}'.format(args.thermal_checkpoint_path))
        else:
            # raise ValueError('Thermal checkpoint path not provided.')
            print('Thermal checkpoint path not provided.')
        
        if args.rgb_checkpoint_path:
            effdet.helpers.load_checkpoint(rgb_det, args.rgb_checkpoint_path)
            print('Loading RGB from {}'.format(args.rgb_checkpoint_path))
        else:
            # raise ValueError('RGB checkpoint path not provided.')
            print('RGB checkpoint path not provided.')

            
        
        self.thermal_backbone = thermal_det.backbone
        self.thermal_fpn = thermal_det.fpn
        self.thermal_class_net = thermal_det.class_net
        self.thermal_box_net = thermal_det.box_net

        self.rgb_backbone = rgb_det.backbone
        self.rgb_fpn = rgb_det.fpn
        self.rgb_class_net = rgb_det.class_net
        self.rgb_box_net = rgb_det.box_net

        fusion_det = EfficientDet(self.config)
        
        if args.init_fusion_head_weights == 'thermal':
            effdet.helpers.load_checkpoint(fusion_det, args.thermal_checkpoint_path) # This is optional
            print("Loading fusion head from thermal checkpoint.")
        elif args.init_fusion_head_weights == 'rgb':
            effdet.helpers.load_checkpoint(fusion_det, args.rgb_checkpoint_path)
            print("Loading fusion head from rgb checkpoint.")
        else:
            # raise ValueError('Fusion head random init.')
            print('Fusion head random init.')
        

        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        if args.branch == 'fusion':
            self.attention_type = args.att_type
            print("Using {} attention.".format(self.attention_type))
            in_chs = args.channels
            for level in range(self.config.num_levels):
                if self.attention_type=="cbam":
                    self.add_module("fusion_"+self.attention_type+str(level), CBAMLayer(2*in_chs))
                elif self.attention_type=="eca":
                    self.add_module("fusion_"+self.attention_type+str(level), attention_block(2*in_chs))
                elif self.attention_type=="shuffle":
                    self.add_module("fusion_"+self.attention_type+str(level), shuffle_attention_block(2*in_chs))
                else:
                    raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')
        
        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                attention = getattr(self, "fusion_"+self.attention_type+str(i))
                out.append(attention(x))
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)
        
        
        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box



##################################### Adaptive Fusion Net ###############################################
class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(208, n_classes)

    def forward(self, x):
        x = self.l1(x)
        return x

class Adaptive_Att_FusionNet(Att_FusionNet):

    def __init__(self, args):
        Att_FusionNet.__init__(self, args)

        self.num_scenes = args.num_scenes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(n_classes=self.num_scenes, dropout=0.5)

        if args.branch == 'fusion':
            in_chs = args.channels
            del self.fusion_cbam0
            del self.fusion_cbam1
            del self.fusion_cbam2
            del self.fusion_cbam3
            del self.fusion_cbam4
            for scene in range(self.num_scenes):
                for level in range(self.config.num_levels):
                    if self.attention_type=="cbam":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), CBAMLayer(2*in_chs))
                    elif self.attention_type=="eca":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), attention_block(2*in_chs))
                    elif self.attention_type=="shuffle":
                        self.add_module("fusion"+str(scene)+"_"+self.attention_type+str(level), shuffle_attention_block(2*in_chs))
                    else:
                        raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')

        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            feat = self.avgpool(rgb_x[len(rgb_x)-1])
            feat = feat.view(feat.size(0), -1)
            image_class_out = self.classifier(feat)
            image_class_out = torch.argmax(image_class_out, dim=1).cpu().numpy()[0]

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                x = torch.cat((tx, vx), dim=1)
                attention = getattr(self, "fusion"+str(image_class_out)+"_"+self.attention_type+str(i))
                out.append(attention(x))
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)


        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box
    
    

###################################New Channel Attention Block#####################################

class channel_attention_block(nn.Module):

    """ Implements a Channel Attention Block """

    def __init__(self,in_channels):

        super(channel_attention_block, self).__init__()
        
        adaptive_k = self.channel_att_kernel_calc(in_channels)
        

        self.pool_types = ["max","avg"]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Conv1d(1,1,kernel_size=adaptive_k,padding=(adaptive_k-1)//2,bias=False)
        
        self.combine = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        channel_att_sum = None

        for pool_type in self.pool_types:

            if pool_type == "avg":

                avg_pool = self.avg_pool(x)
                channel_att_raw = self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            elif pool_type == "max":

                max_pool = self.max_pool(x)
                channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            if channel_att_sum is None:

                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        gate = self.sigmoid(channel_att_sum).expand_as(x)

        return self.combine(x*gate)
    
    
    def channel_att_kernel_calc(self,num_channels,gamma=2,b=1):
        b=1
        gamma = 2
        t = int(abs((math.log(num_channels,2)+b)/gamma))
        k = t if t%2 else t+1
        
        return k


###################################New Spatial Attention Block#####################################


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )



class spatial_attention_block(nn.Module):

    """ Implements a Spatial Attention Block """

    def __init__(self):

        super(spatial_attention_block,self).__init__()

        kernel_size = 7

        self.compress = ChannelPool()

        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        x_compress = self.compress(x)

        x_out = self.spatial(x_compress)

        gate = self.sigmoid(x_out)

        return x*gate

###################################Complete Attention Block#####################################

class attention_block(nn.Module):

    def __init__(self,in_channels):

        super(attention_block,self).__init__()

        self.channel_attention_block = channel_attention_block(in_channels=in_channels)

        self.spatial_attention_block = spatial_attention_block()

    def forward(self,x):

        x_out = self.channel_attention_block(x)
        x_out_1 = self.spatial_attention_block(x_out)

        return x_out_1
    
    

    



################################################################################################################################
################################################# Shuffle Attention ############################################################

class shuffle_attention_block(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channels, groups=16):
        super(shuffle_attention_block, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.cbias =  nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))
        self.sweight =  nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.sbias =  nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))
        
        self.combine = nn.Conv2d(in_channels, int(in_channels/2), kernel_size=1)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        
        
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)

        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)

        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        
        # Reduce the Channels
        out = self.combine(out)
        
        return out



