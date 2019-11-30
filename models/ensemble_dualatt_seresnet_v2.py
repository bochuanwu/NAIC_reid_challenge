#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/14 11:12 上午
# @author  : wuh-xmu
# @FileName: ensemble_dualatt_seresnet_v2.py
# @Software: PyCharm
import torch.nn as nn
import torch
from .dualatt_seresnet_v2 import duallatt_seresnet101v2
from utils import check_isfile, load_pretrained_weights

__all__ = ['ensemble_duallatt_seresnet']


class EnsembleDualAttSeResnet(nn.Module):
    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(EnsembleDualAttSeResnet, self).__init__()
        self.model1 = duallatt_seresnet101v2(num_classes, loss=loss, **kwargs)
        self.model2 = duallatt_seresnet101v2(num_classes, loss=loss, **kwargs)
        self.model3 = duallatt_seresnet101v2(num_classes, loss=loss, **kwargs)

        model1_path = 'log/dualattv2_ser101_ibn_softmax_triplet_256x128_adam_cosine_v3.yaml/model-best-score.pth.tar'
        model2_path = 'log/dualattv2_ser101_ibn_softmax_ranked_256x128_adam_cosine_v1.yaml/model-best-score.pth.tar'
        model3_path = 'log/dualattv2_ser101_ibn_softmax_cranked_256x128_adam_cosine_v1.yaml/model-best-score.pth.tar'
        if check_isfile(model1_path):
            load_pretrained_weights(self.model1, model1_path)
        if check_isfile(model2_path):
            load_pretrained_weights(self.model2, model2_path)
        if check_isfile(model3_path):
            load_pretrained_weights(self.model3, model3_path)

    def forward(self, x):
        self.eval()
        f1 = self.model1(x)
        f2 = self.model2(x)
        f3 = self.model3(x)
        f = torch.cat((f1, f2, f3), 1)
        return f


def ensemble_duallatt_seresnet(num_classes, loss='softmax', **kwargs):
    model = EnsembleDualAttSeResnet(num_classes, loss, **kwargs)
    return model
