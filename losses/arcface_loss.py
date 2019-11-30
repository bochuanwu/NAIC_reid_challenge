#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 7:52 下午
# @Author  : wuh-xmu
# @FileName: arcface_loss.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .cross_entropy_loss  import CrossEntropyLoss


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, num_classes, s=32.0, m=0.30, use_gpu=True, label_smooth=False):
        super(ArcMarginProduct, self).__init__()
        self.use_gpu = use_gpu
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = CrossEntropyLoss(num_classes, label_smooth=label_smooth)

    def forward(self, inputs, targets):
        sine = torch.sqrt((1.0 - torch.pow(inputs, 2)).clamp(1e-8, 1))
        phi = inputs * self.cos_m - sine * self.sin_m
        phi = torch.where(inputs > self.th, phi, inputs - self.mm)

        one_hot = torch.zeros(size=inputs.size(),
                              device='cuda' if self.use_gpu else 'cpu')\
                        .scatter_(1, targets.view(-1, 1).long(), 1)\
                        .float()
        output = (one_hot * phi) + ((1.0 - one_hot) * inputs)
        output *= self.s
        loss = self.criterion(output, targets)
        return loss
