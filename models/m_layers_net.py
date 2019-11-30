#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/4 9:07 下午
# @author  : wuh-xmu
# @FileName: m_layers_net.py
# @Software: PyCharm
import torch
import torch.nn as nn

from .kaiming_init import weights_init_kaiming, weights_init_classifier
from .seresnet_ibn_a import seresnet101_ibn_a

__all__ = ['mlayer_seresnet101']


class MLayersModel(nn.Module):
    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(MLayersModel, self).__init__()
        self.loss = loss
        self.input_norm = nn.BatchNorm2d(3)
        resnet_based = seresnet101_ibn_a(num_classes, loss, **kwargs)
        self.pre_layer = nn.Sequential(
            resnet_based.conv1,
            resnet_based.bn1,
            resnet_based.relu,
            resnet_based.maxpool,
            resnet_based.layer1,
            resnet_based.layer2,
            resnet_based.layer3,
        )

        self.layer4 = resnet_based.layer4
        self.up_channel0 = nn.Sequential(
            nn.Conv2d(2048, 8192, kernel_size=1, bias=False),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True)
        )
        self.up_channel1 = nn.Sequential(
            nn.Conv2d(2048, 8192, kernel_size=1, bias=False),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True)
        )
        self.up_channel2 = nn.Sequential(
            nn.Conv2d(2048, 8192, kernel_size=1, bias=False),
            nn.BatchNorm2d(8192),
            nn.ReLU(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.bn = nn.BatchNorm1d(8192 * 3)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(8192 * 3, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def hbp(self, proj_1, proj_2):
        N = proj_1.size()[0]
        assert (proj_1.size() == (N, 8192, 16, 8))
        X = proj_1 * proj_2
        assert (X.size() == (N, 8192, 16, 8))
        X = torch.sum(X.view(X.size()[0], X.size()[1], -1), dim=2)
        X = X.view(N, 8192)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, x):
        x = self.input_norm(x)
        x = self.pre_layer(x)
        x0 = self.layer4[0](x)
        x1 = self.layer4[1](x0)
        x2 = self.layer4[2](x1)

        x0 = self.up_channel0(x0)
        x1 = self.up_channel1(x1)
        x2 = self.up_channel2(x2)

        f1 = self.hbp(x0, x1)
        f2 = self.hbp(x0, x2)
        f3 = self.hbp(x1, x2)
        f = torch.cat((f1, f2, f3), 1)
        f = self.bn(f)

        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == 'softmax':
            return y
        elif self.loss in ['triplet', 'center']:
            return y, f


def mlayer_seresnet101(num_classes, loss='softmax', **kwargs):
    model = MLayersModel(num_classes, loss, **kwargs)
    return model
