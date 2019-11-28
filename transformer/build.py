# encoding: utf-8
"""
@author:  zhoumi
@contact: zhoumi281571814@126.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing
from .crop import center_crop, crop_lb, crop_lt, crop_rb, crop_rt

def build_transforms(opt, is_train=True, flip=False, crop = ''):
    normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        transform = T.Compose([
            T.Resize(opt.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=opt.PROB),
            T.Pad(opt.PADDING),
            T.RandomCrop(opt.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=opt.RE_PROB, mean=opt.PIXEL_MEAN)
        ])
    else:
        if flip:
            transform = T.Compose([
                T.Resize(opt.SIZE_TEST),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
                normalize_transform
            ])
        else:
            if crop == 'center':
                transform = T.Compose([
                    T.Resize([x+10 for x in opt.SIZE_TEST]),
                    center_crop(384, 128),
                    T.ToTensor(),
                    normalize_transform
                ])
            elif crop == 'lt':
                transform = T.Compose([
                    T.Resize([x+10 for x in opt.SIZE_TEST]),
                    crop_lt(384, 128),
                    T.ToTensor(),
                    normalize_transform
                ])
            elif crop == 'rt':
                transform = T.Compose([
                    T.Resize([x+10 for x in opt.SIZE_TEST]),
                    crop_rt(384, 128),
                    T.ToTensor(),
                    normalize_transform
                ])
            elif crop == 'lb':
                transform = T.Compose([
                    T.Resize([x+10 for x in opt.SIZE_TEST]),
                    crop_lb(384, 128),
                    T.ToTensor(),
                    normalize_transform
                ])
            elif crop == 'rb':
                transform = T.Compose([
                    T.Resize([x+10 for x in opt.SIZE_TEST]),
                    crop_rb(384, 128),
                    T.ToTensor(),
                    normalize_transform
                ])
            else:
                transform = T.Compose([
                    T.Resize(opt.SIZE_TEST),
                    T.ToTensor(),
                    normalize_transform
                ])

    return transform
