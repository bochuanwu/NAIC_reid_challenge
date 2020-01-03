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

_all__ = ['mgnv4_resnet101_512', 'mgnv4_seresnet101_512', 'mgnv4_resnext101_512']


class BasicOpeartor(nn.Module):
    def __init__(self, out_shape):
        super(BasicOpeartor, self).__init__()
        self.out_shape = out_shape
        self.gap = nn.AdaptiveAvgPool2d(out_shape)
        self.gmp = nn.AdaptiveMaxPool2d(out_shape)

    def forward(self, x):
        return self.gap(x) + self.gmp(x)


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

        self.part1 = nn.Sequential(copy.deepcopy(resnet.layer4))
        self.part2 = nn.Sequential(copy.deepcopy(resnet.layer4))
        self.part3 = nn.Sequential(copy.deepcopy(resnet.layer4))

        self.gmp = nn.AdaptiveMaxPool2d((1, 1))
        self.part1_pool = nn.AdaptiveMaxPool2d((2, 1))
        self.part2_pool = nn.AdaptiveMaxPool2d((3, 1))
        self.part3_pool = nn.AdaptiveMaxPool2d((4, 1))

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
        self.part3_f3_reduction = copy.deepcopy(reduction)
        self.part3_f4_reduction = copy.deepcopy(reduction)

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
        self.part3_f3_bnneck = copy.deepcopy(bnneck)
        self.part3_f4_bnneck = copy.deepcopy(bnneck)

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
        self.part3_classifier3 = copy.deepcopy(classifier)
        self.part3_classifier4 = copy.deepcopy(classifier)

    def forward(self, x):
        x = self.backbone(x)

        part1 = self.part1(x)
        part2 = self.part2(x)
        part3 = self.part3(x)

        part1_f0_triplet_feature = self.part1_f0_reduction(self.gmp(part1)).flatten(1)
        part1_f1_triplet_feature = self.part1_f1_reduction(self.part1_pool(part1)[:, :, 0:1, :]).flatten(1)
        part1_f2_triplet_feature = self.part1_f2_reduction(self.part1_pool(part1)[:, :, 1:2, :]).flatten(1)

        part2_f0_triplet_feature = self.part2_f0_reduction(self.gmp(part2)).flatten(1)
        part2_f1_triplet_feature = self.part2_f1_reduction(self.part2_pool(part2)[:, :, 0:1, :]).flatten(1)
        part2_f2_triplet_feature = self.part2_f2_reduction(self.part2_pool(part2)[:, :, 1:2, :]).flatten(1)
        part2_f3_triplet_feature = self.part2_f3_reduction(self.part2_pool(part2)[:, :, 2:3, :]).flatten(1)

        part3_f0_triplet_feature = self.part3_f0_reduction(self.gmp(part3)).flatten(1)
        part3_f1_triplet_feature = self.part3_f1_reduction(self.part3_pool(part3)[:, :, 0:1, :]).flatten(1)
        part3_f2_triplet_feature = self.part3_f2_reduction(self.part3_pool(part3)[:, :, 1:2, :]).flatten(1)
        part3_f3_triplet_feature = self.part3_f3_reduction(self.part3_pool(part3)[:, :, 2:3, :]).flatten(1)
        part3_f4_triplet_feature = self.part3_f4_reduction(self.part3_pool(part3)[:, :, 3:4, :]).flatten(1)

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
        part3_f3_softmax_feature = self.part3_f3_bnneck(part3_f3_triplet_feature)
        part3_f4_softmax_feature = self.part3_f4_bnneck(part3_f4_triplet_feature)

        part1_local_triplet_features = [part1_f1_triplet_feature, part1_f2_triplet_feature]
        part2_local_triplet_features = [part2_f1_triplet_feature, part2_f2_triplet_feature, part2_f3_triplet_feature]
        part3_local_triplet_features = [part3_f1_triplet_feature, part3_f2_triplet_feature, part3_f3_triplet_feature,
                                        part3_f4_triplet_feature]

        part1_local_softmax_features = [part1_f1_softmax_feature, part1_f2_softmax_feature]
        part2_local_softmax_features = [part2_f1_softmax_feature, part2_f2_softmax_feature, part2_f3_softmax_feature]
        part3_local_softmax_features = [part3_f1_softmax_feature, part3_f2_softmax_feature, part3_f3_softmax_feature,
                                        part3_f4_softmax_feature]
        global_triplet_features = [part1_f0_triplet_feature, part2_f0_triplet_feature, part3_f0_triplet_feature]
        global_softmax_features = [part1_f0_softmax_feature, part2_f0_softmax_feature, part3_f0_softmax_feature]

        f = global_softmax_features + part1_local_softmax_features + part2_local_softmax_features + part3_local_softmax_features
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
        part3_y3 = self.part3_classifier3(part3_f3_softmax_feature)
        part3_y4 = self.part3_classifier4(part3_f4_softmax_feature)

        y = [part1_y0, part1_y1, part1_y2,
             part2_y0, part2_y1, part2_y2, part2_y3,
             part3_y0, part3_y1, part3_y2, part3_y3, part3_y4]

        part1_local_triplet_features = torch.cat(part1_local_triplet_features, 1)
        part2_local_triplet_features = torch.cat(part2_local_triplet_features, 1)
        part3_local_triplet_features = torch.cat(part3_local_triplet_features, 1)
        return y, global_triplet_features + [part1_local_triplet_features, part2_local_triplet_features,
                                             part3_local_triplet_features]


def mgnv4_resnet101_512(num_classes, loss='softmax', **kwargs):
    resnet = resnet101_ibn_a(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=512)
    return model

def mgnv4_seresnet101_512(num_classes, loss='softmax', **kwargs):
    resnet = seresnet101_ibn_a(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=512)
    return model

def mgnv4_resnext101_512(num_classes, loss='softmax', **kwargs):
    resnet = resnext101_ibn_a_32x4d(num_classes, loss=loss, **kwargs)
    model = MGN(resnet, num_classes, num_dim=512)
    return model