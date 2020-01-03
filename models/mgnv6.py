#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/12/05 9:33 下午
# @author  : wuh-xmu
# @FileName: mgn.py
# @Software: PyCharm


import copy

import torch
from torch import nn
import torch.nn.functional as F

from models import resnext101_ibn_a_32x4d
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .kaiming_init import weights_init_classifier, weights_init_kaiming
from .seresnet_ibn_a import seresnet101_ibn_a
from .attention import CAM_Module, PAM_Module

_all__ = ['mgnv6_resnet101_256', 'mgnv6_seresnet101_256', 'mgnv6_seresnet101_512', 'mgnv6_resnext101_32x4d_256',
          'mgnv6_resnext101_32x4d_512']


def center_crop(x, crop_h, crop_w):
    """make center crop"""

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]


class AnnularPool(nn.Module):
    def __init__(self, scale, pool='avg'):
        super(AnnularPool, self).__init__()
        assert scale in [2, 3]
        self.scale = scale
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        h, w = x.shape[2:]

        margin_h = h // self.scale
        margin_w = w // self.scale
        res = [self.pool(center_crop(x, h - margin_h * i, w - margin_w * i)) for i in range(1, self.scale)]
        return torch.cat(res, 2)


class MGN(nn.Module):
    def __init__(self, resnet, num_classes, num_dim=256):
        super(MGN, self).__init__()
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )

        # self.part0 = copy.deepcopy(resnet.layer4)
        self.part1 = copy.deepcopy(resnet.layer4)
        self.part2 = copy.deepcopy(resnet.layer4)
        self.part3 = copy.deepcopy(resnet.layer4)
        self.part4 = copy.deepcopy(resnet.layer4)
        self.part5 = copy.deepcopy(resnet.layer4)
        self.part6 = copy.deepcopy(resnet.layer4)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.part1_pool = nn.AdaptiveMaxPool2d((2, 1))
        self.part2_pool = nn.AdaptiveMaxPool2d((3, 1))
        self.part3_pool = nn.AdaptiveMaxPool2d((1, 2))
        self.part4_pool = nn.AdaptiveMaxPool2d((1, 3))
        self.part5_pool = AnnularPool(scale=2, pool='max')
        self.part6_pool = AnnularPool(scale=3, pool='max')

        reduction = nn.Sequential(
            nn.Conv2d(2048, num_dim, 1, bias=False),
            nn.BatchNorm2d(num_dim),
            nn.ReLU()
        )
        reduction.apply(weights_init_kaiming)

        self.part1_f0_reduction = copy.deepcopy(reduction)
        self.part1_f1_reduction = copy.deepcopy(reduction)
        self.part1_f2_reduction = copy.deepcopy(reduction)

        self.part2_f0_reduction = copy.deepcopy(reduction)
        self.part2_f1_reduction = copy.deepcopy(reduction)
        self.part2_f2_reduction = copy.deepcopy(reduction)
        self.part2_f3_reduction = copy.deepcopy(reduction)

        self.part3_f0_reduction = copy.deepcopy(reduction)
        self.part3_f1_reduction = copy.deepcopy(reduction)
        self.part3_f2_reduction = copy.deepcopy(reduction)

        self.part4_f0_reduction = copy.deepcopy(reduction)
        self.part4_f1_reduction = copy.deepcopy(reduction)
        self.part4_f2_reduction = copy.deepcopy(reduction)
        self.part4_f3_reduction = copy.deepcopy(reduction)

        self.part5_f0_reduction = copy.deepcopy(reduction)
        self.part5_f1_reduction = copy.deepcopy(reduction)

        self.part6_f0_reduction = copy.deepcopy(reduction)
        self.part6_f1_reduction = copy.deepcopy(reduction)
        self.part6_f2_reduction = copy.deepcopy(reduction)

        bnneck = nn.BatchNorm1d(num_dim)
        bnneck.bias.requires_grad_(False)
        bnneck.apply(weights_init_kaiming)

        self.part1_f0_bnneck = copy.deepcopy(bnneck)
        self.part1_f1_bnneck = copy.deepcopy(bnneck)
        self.part1_f2_bnneck = copy.deepcopy(bnneck)

        self.part2_f0_bnneck = copy.deepcopy(bnneck)
        self.part2_f1_bnneck = copy.deepcopy(bnneck)
        self.part2_f2_bnneck = copy.deepcopy(bnneck)
        self.part2_f3_bnneck = copy.deepcopy(bnneck)

        self.part3_f0_bnneck = copy.deepcopy(bnneck)
        self.part3_f1_bnneck = copy.deepcopy(bnneck)
        self.part3_f2_bnneck = copy.deepcopy(bnneck)

        self.part4_f0_bnneck = copy.deepcopy(bnneck)
        self.part4_f1_bnneck = copy.deepcopy(bnneck)
        self.part4_f2_bnneck = copy.deepcopy(bnneck)
        self.part4_f3_bnneck = copy.deepcopy(bnneck)

        self.part5_f0_bnneck = copy.deepcopy(bnneck)
        self.part5_f1_bnneck = copy.deepcopy(bnneck)

        self.part6_f0_bnneck = copy.deepcopy(bnneck)
        self.part6_f1_bnneck = copy.deepcopy(bnneck)
        self.part6_f2_bnneck = copy.deepcopy(bnneck)

        classifier = nn.Linear(num_dim, num_classes, bias=False)
        classifier.apply(weights_init_classifier)

        self.part1_classifier0 = copy.deepcopy(classifier)
        self.part1_classifier1 = copy.deepcopy(classifier)
        self.part1_classifier2 = copy.deepcopy(classifier)

        self.part2_classifier0 = copy.deepcopy(classifier)
        self.part2_classifier1 = copy.deepcopy(classifier)
        self.part2_classifier2 = copy.deepcopy(classifier)
        self.part2_classifier3 = copy.deepcopy(classifier)

        self.part3_classifier0 = copy.deepcopy(classifier)
        self.part3_classifier1 = copy.deepcopy(classifier)
        self.part3_classifier2 = copy.deepcopy(classifier)

        self.part4_classifier0 = copy.deepcopy(classifier)
        self.part4_classifier1 = copy.deepcopy(classifier)
        self.part4_classifier2 = copy.deepcopy(classifier)
        self.part4_classifier3 = copy.deepcopy(classifier)

        self.part5_classifier0 = copy.deepcopy(classifier)
        self.part5_classifier1 = copy.deepcopy(classifier)

        self.part6_classifier0 = copy.deepcopy(classifier)
        self.part6_classifier1 = copy.deepcopy(classifier)
        self.part6_classifier2 = copy.deepcopy(classifier)

    def forward(self, x):
        x = self.backbone(x)

        part1 = self.part1(x)
        part2 = self.part2(x)
        part3 = self.part3(x)
        part4 = self.part4(x)
        part5 = self.part5(x)
        part6 = self.part6(x)

        part1_f0_triplet_feature = self.part1_f0_reduction(self.global_pool(part1)).flatten(1)
        part1_f1_triplet_feature = self.part1_f1_reduction(self.part1_pool(part1)[:, :, 0:1, :]).flatten(1)
        part1_f2_triplet_feature = self.part1_f2_reduction(self.part1_pool(part1)[:, :, 1:2, :]).flatten(1)

        part2_f0_triplet_feature = self.part2_f0_reduction(self.global_pool(part2)).flatten(1)
        part2_f1_triplet_feature = self.part2_f1_reduction(self.part2_pool(part2)[:, :, 0:1, :]).flatten(1)
        part2_f2_triplet_feature = self.part2_f2_reduction(self.part2_pool(part2)[:, :, 1:2, :]).flatten(1)
        part2_f3_triplet_feature = self.part2_f3_reduction(self.part2_pool(part2)[:, :, 2:3, :]).flatten(1)

        part3_f0_triplet_feature = self.part3_f0_reduction(self.global_pool(part3)).flatten(1)
        part3_f1_triplet_feature = self.part3_f1_reduction(self.part3_pool(part3)[:, :, :, 0:1]).flatten(1)
        part3_f2_triplet_feature = self.part3_f2_reduction(self.part3_pool(part3)[:, :, :, 1:2]).flatten(1)

        part4_f0_triplet_feature = self.part4_f0_reduction(self.global_pool(part4)).flatten(1)
        part4_f1_triplet_feature = self.part4_f1_reduction(self.part4_pool(part4)[:, :, :, 0:1]).flatten(1)
        part4_f2_triplet_feature = self.part4_f2_reduction(self.part4_pool(part4)[:, :, :, 1:2]).flatten(1)
        part4_f3_triplet_feature = self.part4_f3_reduction(self.part4_pool(part4)[:, :, :, 2:3]).flatten(1)

        part5_f0_triplet_feature = self.part5_f0_reduction(self.global_pool(part5)).flatten(1)
        part5_f1_triplet_feature = self.part5_f1_reduction(self.part5_pool(part5)[:, :, 0:1, :]).flatten(1)

        part6_f0_triplet_feature = self.part6_f0_reduction(self.global_pool(part6)).flatten(1)
        part6_f1_triplet_feature = self.part6_f1_reduction(self.part6_pool(part6)[:, :, 0:1, :]).flatten(1)
        part6_f2_triplet_feature = self.part6_f2_reduction(self.part6_pool(part6)[:, :, 1:2, :]).flatten(1)

        part1_f0_softmax_feature = self.part1_f0_bnneck(part1_f0_triplet_feature)
        part1_f1_softmax_feature = self.part1_f1_bnneck(part1_f1_triplet_feature)
        part1_f2_softmax_feature = self.part1_f2_bnneck(part1_f2_triplet_feature)

        part2_f0_softmax_feature = self.part2_f0_bnneck(part2_f0_triplet_feature)
        part2_f1_softmax_feature = self.part2_f1_bnneck(part2_f1_triplet_feature)
        part2_f2_softmax_feature = self.part2_f2_bnneck(part2_f2_triplet_feature)
        part2_f3_softmax_feature = self.part2_f3_bnneck(part2_f3_triplet_feature)

        part3_f0_softmax_feature = self.part3_f0_bnneck(part3_f0_triplet_feature)
        part3_f1_softmax_feature = self.part3_f1_bnneck(part3_f1_triplet_feature)
        part3_f2_softmax_feature = self.part3_f2_bnneck(part3_f2_triplet_feature)

        part4_f0_softmax_feature = self.part4_f0_bnneck(part4_f0_triplet_feature)
        part4_f1_softmax_feature = self.part4_f1_bnneck(part4_f1_triplet_feature)
        part4_f2_softmax_feature = self.part4_f2_bnneck(part4_f2_triplet_feature)
        part4_f3_softmax_feature = self.part4_f3_bnneck(part4_f3_triplet_feature)

        part5_f0_softmax_feature = self.part5_f0_bnneck(part5_f0_triplet_feature)
        part5_f1_softmax_feature = self.part5_f1_bnneck(part5_f1_triplet_feature)

        part6_f0_softmax_feature = self.part6_f0_bnneck(part6_f0_triplet_feature)
        part6_f1_softmax_feature = self.part6_f1_bnneck(part6_f1_triplet_feature)
        part6_f2_softmax_feature = self.part6_f2_bnneck(part6_f2_triplet_feature)

        part1_local_triplet_features = [part1_f1_triplet_feature, part1_f2_triplet_feature]
        part2_local_triplet_features = [part2_f1_triplet_feature, part2_f2_triplet_feature, part2_f3_triplet_feature]

        part3_local_triplet_features = [part3_f1_triplet_feature, part3_f2_triplet_feature]
        part4_local_triplet_features = [part4_f1_triplet_feature, part4_f2_triplet_feature, part4_f3_triplet_feature]

        part5_local_triplet_features = [part5_f1_triplet_feature]
        part6_local_triplet_features = [part6_f1_triplet_feature, part6_f2_triplet_feature]

        part1_local_softmax_features = [part1_f1_softmax_feature, part1_f2_softmax_feature]
        part2_local_softmax_features = [part2_f1_softmax_feature, part2_f2_softmax_feature, part2_f3_softmax_feature]

        part3_local_softmax_features = [part3_f1_softmax_feature, part3_f2_softmax_feature]
        part4_local_softmax_features = [part4_f1_softmax_feature, part4_f2_softmax_feature, part4_f3_softmax_feature]

        part5_local_softmax_features = [part5_f1_softmax_feature]
        part6_local_softmax_features = [part6_f1_softmax_feature, part6_f2_softmax_feature]

        global_triplet_features = [part1_f0_triplet_feature, part2_f0_triplet_feature, part3_f0_triplet_feature,
                                   part4_f0_triplet_feature, part5_f0_triplet_feature, part6_f0_triplet_feature]

        global_softmax_features = [part1_f0_softmax_feature, part2_f0_softmax_feature, part3_f0_softmax_feature,
                                   part4_f0_softmax_feature, part5_f0_softmax_feature, part6_f0_softmax_feature]
        f = global_softmax_features + part1_local_softmax_features + \
            part2_local_softmax_features + part3_local_softmax_features \
            + part4_local_softmax_features + part5_local_softmax_features \
            + part6_local_softmax_features
        if not self.training:
            return torch.cat(f, 1)

        part1_y0 = self.part1_classifier0(part1_f0_softmax_feature)
        part1_y1 = self.part1_classifier1(part1_f1_softmax_feature)
        part1_y2 = self.part1_classifier2(part1_f2_softmax_feature)

        part2_y0 = self.part2_classifier0(part2_f0_softmax_feature)
        part2_y1 = self.part2_classifier1(part2_f1_softmax_feature)
        part2_y2 = self.part2_classifier2(part2_f2_softmax_feature)
        part2_y3 = self.part2_classifier3(part2_f3_softmax_feature)

        part3_y0 = self.part3_classifier0(part3_f0_softmax_feature)
        part3_y1 = self.part3_classifier1(part3_f1_softmax_feature)
        part3_y2 = self.part3_classifier2(part3_f2_softmax_feature)

        part4_y0 = self.part4_classifier0(part4_f0_softmax_feature)
        part4_y1 = self.part4_classifier1(part4_f1_softmax_feature)
        part4_y2 = self.part4_classifier2(part4_f2_softmax_feature)
        part4_y3 = self.part4_classifier3(part4_f3_softmax_feature)

        part5_y0 = self.part5_classifier0(part5_f0_softmax_feature)
        part5_y1 = self.part5_classifier1(part5_f1_softmax_feature)

        part6_y0 = self.part6_classifier0(part6_f0_softmax_feature)
        part6_y1 = self.part6_classifier1(part6_f1_softmax_feature)
        part6_y2 = self.part6_classifier2(part6_f2_softmax_feature)

        y = [part1_y0, part1_y1, part1_y2,
             part2_y0, part2_y1, part2_y2, part2_y3,
             part3_y0, part3_y1, part3_y2,
             part4_y0, part4_y1, part4_y2, part4_y3,
             part5_y0, part5_y1,
             part6_y0, part6_y1, part6_y2]

        part1_local_triplet_features = torch.cat(part1_local_triplet_features, 1)
        part2_local_triplet_features = torch.cat(part2_local_triplet_features, 1)
        part3_local_triplet_features = torch.cat(part3_local_triplet_features, 1)
        part4_local_triplet_features = torch.cat(part4_local_triplet_features, 1)
        part5_local_triplet_features = torch.cat(part5_local_triplet_features, 1)
        part6_local_triplet_features = torch.cat(part6_local_triplet_features, 1)

        return y, global_triplet_features + [part1_local_triplet_features, part2_local_triplet_features,
                                             part3_local_triplet_features, part4_local_triplet_features,
                                             part5_local_triplet_features, part6_local_triplet_features]


def mgnv6_resnet101_256(num_classes, loss='softmax', **kwargs):
    resnet = resnet101_ibn_a(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=256)
    return model


def mgnv6_seresnet101_256(num_classes, loss='softmax', **kwargs):
    resnet = seresnet101_ibn_a(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=256)
    return model


def mgnv6_seresnet101_512(num_classes, loss='softmax', **kwargs):
    resnet = seresnet101_ibn_a(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=512)
    return model


def mgnv6_resnext101_32x4d_256(num_classes, loss='softmax', **kwargs):
    resnet = resnext101_ibn_a_32x4d(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=256)
    return model


def mgnv6_resnext101_32x4d_512(num_classes, loss='softmax', **kwargs):
    resnet = resnext101_ibn_a_32x4d(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=512)
    return model
