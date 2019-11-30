#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/7 7:29 下午
# @author  : wuh-xmu
# @FileName: amsoftmax_linear.py
# @Software: PyCharm

import torch.nn as nn
import torch


class AMSoftmax(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AMSoftmax, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.xavier_normal_(self.fc.weight, gain=1)

    def forward(self, x):

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.fc.weight, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.fc.weight, w_norm)
        output = torch.mm(x_norm, w_norm)
        return output
