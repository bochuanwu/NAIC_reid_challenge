# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing


def build_transforms(opt, is_train=True, flip=False):
    normalize_transform = T.Normalize(mean=opt.PIXEL_MEAN, std=opt.PIXEL_STD)
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
            transform = T.Compose([
                T.Resize(opt.SIZE_TEST),
                T.ToTensor(),
                normalize_transform
            ])

    return transform
