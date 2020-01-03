#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/15 5:46 下午
# @author  : wuh-xmu
# @FileName: focal_loss.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, inputs, target, ohem=None):
        max_val = (-inputs).clamp(min=0)
        loss = inputs - inputs * target + max_val + ((-max_val).exp() + (-inputs - max_val).exp()).log()
        invprobs = F.logsigmoid(-inputs * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        if ohem is not None:
            loss, _ = loss.topk(k=int(self.num_classes * ohem),
                                dim=1,
                                largest=True,
                                sorted=True)
        return loss.mean()

class BCELossWithLogit(nn.Module):
    def __init__(self, nun_classes):
        super(BCELossWithLogit, self).__init__()
        self.num_classes = nun_classes

    def forward(self, inputs, target, ohem=None):
        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none')
        if ohem is not None:
            loss, _ = loss.topk(int(self.num_classes * ohem),
                                dim=1,
                                largest=True,
                                sorted=True)
        return loss.mean()

class FocalLossWithOHEM(nn.Module):
    def __init__(self, num_classes):
        super(FocalLossWithOHEM, self).__init__()
        self.bce_loss = BCELossWithLogit(num_classes)
        self.focal_loss = BCELossWithLogit(num_classes)

    def forward(self, inputs, targets, ohem=1e-2):
        batch_size, num_classes = inputs.shape
        onehot = torch.zeros((batch_size, num_classes)).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if inputs.is_cuda:
            onehot = onehot.cuda()
        loss0 = self.bce_loss(inputs, onehot, ohem)
        loss1 = self.focal_loss(inputs, onehot, ohem)
        return loss0 + loss1


