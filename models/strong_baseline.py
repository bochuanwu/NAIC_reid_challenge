#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/12/12 10:36 下午
# @author  : wuh-xmu
# @FileName: strong_baseline.py
# @Software: PyCharm

import torch.nn as nn

from models import weights_init_kaiming, weights_init_classifier, resnet50_ibn_a


class StrongBaseline(nn.Module):

    def __init__(self, resnet, num_classes):
        super(StrongBaseline, self).__init__()
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        self.backbone =  self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(weights_init_kaiming)

        self.fc = nn.Linear(2048, num_classes, bias=False)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        x = self.backbone(x)
        triplet_feature = self.global_pool(x).flatten(1)
        softmax_feature = self.bnneck(triplet_feature)

        if not self.training:
            return softmax_feature

        y = self.fc(softmax_feature)
        return y, triplet_feature



def sb_ibn50(num_classes, loss='softmax', **kwargs):
    resnet = resnet50_ibn_a(num_classes, loss=loss, **kwargs)
    model = StrongBaseline(resnet, num_classes)
    return model

