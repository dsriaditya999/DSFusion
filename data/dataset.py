""" Fusion dataset for detection
"""
import torch.utils.data as data
import numpy as np
import copy
import torch
import os

from .cv2_transforms import transforms_coco_eval, transforms_coco_train
from PIL import Image
from effdet.data.parsers import create_parser
from .parsers import create_parser as create_parser_stf
import cv2

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class OnlineTrainingDataset:
    def __init__(self, buffer_size):
        super(OnlineTrainingDataset, self).__init__()
        self.training_sample_buffer = [] # Implement as bounded queue
        self.buffer_size = buffer_size

    def get_training_data(self):
        for i, batch in enumerate(self.training_sample_buffer):
            thermal_img, rgb_img, target = batch[0], batch[1], batch[2]
            if i == 0:
                thermal_img_train = thermal_img
                rgb_img_train = rgb_img
                target_train = copy.deepcopy(target)
            else:
                thermal_img_train = torch.cat((thermal_img_train, thermal_img), dim=0)
                rgb_img_train = torch.cat((rgb_img_train, rgb_img), dim=0)
                for k, v in target_train.items():
                    target_train[k] = torch.cat((target_train[k], target[k]), dim=0)

        return thermal_img_train, rgb_img_train, target_train

    def len(self):
        return len(self.training_sample_buffer)

    def add_current_data(self, thermal_img, rgb_img, target):
        if self.len() == self.buffer_size:
            self.training_sample_buffer.pop(0)
        self.training_sample_buffer.append([thermal_img, rgb_img, target])


class FusionDatasetFLIR(data.Dataset):
    """ Fusion Dataset for Object Detection. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """
    def __init__(self, thermal_data_dir, rgb_data_dir, parser=None, parser_kwargs=None, transform=None):
        super(FusionDatasetFLIR, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.thermal_data_dir = thermal_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (thermal_image, rgb_image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        thermal_img_path = self.thermal_data_dir / img_info['file_name']
        thermal_img = Image.open(thermal_img_path).convert('RGB')
        rgb_img_path = self.rgb_data_dir / img_info['file_name'].replace('PreviewData', 'RGB')
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        if self.transform is not None:
            thermal_img, rgb_img, target = self.transform(thermal_img, rgb_img, target)

        return thermal_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class XBitFusionDatsetSTF(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, 
                 gated_data_dir, rgb_data_dir,
                 parser=None, parser_kwargs=None, transform=None, 
                 mode='train', rgb_bits=16, gated_bits=16,
                 rgb_mean=IMAGENET_DEFAULT_MEAN, 
                 rgb_std=IMAGENET_DEFAULT_STD,
                 gated_mean=IMAGENET_DEFAULT_MEAN,
                 gated_std=IMAGENET_DEFAULT_STD):
        
        super(XBitFusionDatsetSTF, self).__init__()

        parser_kwargs = parser_kwargs or {}
        self.gated_data_dir = gated_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser_stf(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        
        self.mode = mode
        self.gated_bits = gated_bits
        self.rgb_bits = rgb_bits

        if mode == 'train':
            self.fixed_transform = transforms_coco_train(
                1280,
                use_prefetcher=True,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                gated_mean=gated_mean,
                gated_std=gated_std,
            )
        else:
            self.fixed_transform = transforms_coco_eval(
                1280,
                use_prefetcher=True,
                rgb_mean=rgb_mean,
                rgb_std=rgb_std,
                gated_mean=gated_mean,
                gated_std=gated_std,
            )

        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        gated_img_path = os.path.join(self.gated_data_dir, img_info['file_name'])
        gated_img = cv2.imread(gated_img_path, -1) / (2**self.gated_bits - 1) * 255
        if len(gated_img.shape) < 3:
            gated_img = np.stack([gated_img]*3, axis=2)
        else:
            gated_img = gated_img[:,:,::-1]

        rgb_img_path = os.path.join(self.rgb_data_dir, img_info['file_name'])
        rgb_img = cv2.imread(rgb_img_path, -1) / (2**self.rgb_bits - 1) * 255
        if len(rgb_img.shape) < 3:
            rgb_img = np.stack([rgb_img]*3, axis=2)
        else:
            rgb_img = rgb_img[:,:,::-1]


        
        gated_img, rgb_img, target = self.fixed_transform(gated_img, rgb_img, target)
        # print(target)
        # draw_img = img.transpose(1, 2, 0)
        # for box in target['bbox']:
        #     y1, x1, y2, x2 = box.astype(int)
            # cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # cv2.imwrite('testimg.png', draw_img)
        # exit(0)
        return gated_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class FusionDatasetM3FD(data.Dataset):
    """ Fusion Dataset for Object Detection. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
    """
    def __init__(self, thermal_data_dir, rgb_data_dir, parser=None, parser_kwargs=None, transform=None):
        super(FusionDatasetM3FD, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.thermal_data_dir = thermal_data_dir
        self.rgb_data_dir = rgb_data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (thermal_image, rgb_image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        thermal_img_path = self.thermal_data_dir / img_info['file_name']
        thermal_img = Image.open(thermal_img_path).convert('RGB')
        rgb_img_path = self.rgb_data_dir / img_info['file_name']
        rgb_img = Image.open(rgb_img_path).convert('RGB')
        if self.transform is not None:
            thermal_img, rgb_img, target = self.transform(thermal_img, rgb_img, target)

        return thermal_img, rgb_img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t