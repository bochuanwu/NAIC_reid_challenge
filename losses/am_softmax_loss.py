#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/7 7:43 下午
# @author  : wuh-xmu
# @FileName: am_softmax_loss.py
# @Software: PyCharm

import torch.nn as nn
import torch


class AMSoftmaxLoss(nn.Module):
    def __init__(self, m=0.3, s=15):
        super(AMSoftmaxLoss, self).__init__()
        self.m = m
        self.s = s
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        delt_costh = torch.zeros(inputs.size()).scatter_(1, targets.unsqueeze(1).cpu(), self.m)
        if inputs.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = inputs - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.criterion(costh_m_s, targets)
        return loss
