#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/3 5:43 下午
# @Author  : wuh-xmu
# @FileName: dualatt_seresnet.py
# @Software: PyCharm
import copy

import torch
import torch.nn as nn

from .attention import CAM_Module, PAM_Module
from .kaiming_init import weights_init_kaiming, weights_init_classifier
from .seresnet_ibn_a import seresnet101_ibn_a
_all__ = ['duallatt_seresnet101v2']


class DualAttentionModelv2(nn.Module):

    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(DualAttentionModelv2, self).__init__()
        # self.input_norm = nn.BatchNorm2d(3, affine=False)
        self.backbone = seresnet101_ibn_a(num_classes, loss, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.loss = loss

        self.layer3_cam = CAM_Module(1024)
        self.layer3_pam = PAM_Module(1024)

        self.layer34_cam = CAM_Module(2048)
        self.layer34_pam = PAM_Module(2048)

        self.layer3am4_cam = CAM_Module(2048)
        self.layer3am4_pam = PAM_Module(2048)

        layer3_bn = nn.BatchNorm1d(1024)
        layer3_bn.bias.requires_grad_(False)
        layer3_bn.apply(weights_init_kaiming)

        layer4_bn = nn.BatchNorm1d(2048)
        layer4_bn.bias.requires_grad_(False)
        layer4_bn.apply(weights_init_kaiming)

        self.layer3_bn = copy.deepcopy(layer3_bn)
        self.layer3am_bn = copy.deepcopy(layer3_bn)

        self.layer34_bn = copy.deepcopy(layer4_bn)
        self.layer34am_bn = copy.deepcopy(layer4_bn)

        self.layer3am4_bn = copy.deepcopy(layer4_bn)
        self.layer3am4am_bn = copy.deepcopy(layer4_bn)

        layer3_classifier = nn.Linear(1024, num_classes, bias=False)
        layer3_classifier.apply(weights_init_classifier)

        layer4_classifier = nn.Linear(2048, num_classes, bias=False)
        layer4_classifier.apply(weights_init_classifier)

        self.layer3_classifier = copy.deepcopy(layer3_classifier)
        self.layer3am_classifier = copy.deepcopy(layer3_classifier)

        self.layer3am4_classifier = copy.deepcopy(layer4_classifier)
        self.layer3am4am_classifier = copy.deepcopy(layer4_classifier)

        self.layer34_classifier = copy.deepcopy(layer4_classifier)
        self.layer34am_classifier = copy.deepcopy(layer4_classifier)

    def forward(self, x):

        # x = self.input_norm(x)

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)

        layer3 = self.backbone.layer3(x)
        layer3_am = self.layer3_cam(layer3) + self.layer3_pam(layer3)

        layer34 = self.backbone.layer4(layer3)
        layer3am4 = self.backbone.layer4(layer3_am)

        layer3am4_am = self.layer3am4_cam(layer3am4) + self.layer3am4_pam(layer3am4)
        layer34_am = self.layer34_cam(layer34) + self.layer34_pam(layer34)

        layer3 = self.layer3_bn(self.avgpool(layer3).flatten(1))
        layer3_am = self.layer3am_bn(self.avgpool(layer3_am).flatten(1))

        layer34 = self.layer34_bn(self.avgpool(layer34).flatten(1))
        layer34_am = self.layer34am_bn(self.avgpool(layer34_am).flatten(1))

        layer3am4 = self.layer3am4_bn(self.avgpool(layer3am4).flatten(1))
        layer3am4_am = self.layer3am4am_bn(self.avgpool(layer3am4_am).flatten(1))

        f = [layer3, layer3_am, layer34, layer3am4, layer3am4_am, layer34_am]
        if not self.training:
            # return torch.cat(f, 1)
            return torch.cat(f, 1)

        layer3_y = self.layer3_classifier(layer3)
        layer3_am_y = self.layer3am_classifier(layer3_am)
        layer34_y = self.layer34_classifier(layer34)
        layer3am4_y = self.layer3am4_classifier(layer3am4)
        layer3am4_am_y = self.layer3am4am_classifier(layer3am4_am)
        layer34_am_y = self.layer34am_classifier(layer34_am)
        y = [layer3_y, layer3_am_y, layer34_y, layer3am4_y, layer3am4_am_y, layer34_am_y]

        if self.loss == 'softmax':
            return y
        elif self.loss in ['triplet', 'center']:
            return y, f


def duallatt_seresnet101v2(num_classes, loss='softmax', **kwargs):
    model = DualAttentionModelv2(num_classes, loss, **kwargs)
    return model
