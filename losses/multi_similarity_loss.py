#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/13 12:50 下午
# @author  : wuh-xmu
# @FileName: multi_similarity_loss.py
# @Software: PyCharm
import torch
import torch.nn as nn
from metrics import compute_distance_matrix
from torch.nn import functional as F


class MultiSimilarityLoss(nn.Module):
    def __init__(self, thresh=0.5, margin=0.5, scale_pos=2., scale_neg=40.):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = thresh
        self.margin = margin

        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        N = feats.size(0)
        feats = F.normalize(feats, p=2, dim=1)
        sim_mat = torch.matmul(feats, torch.t(feats))
        # sim_mat = compute_distance_matrix(feats, feats, metric='cosine')
        epsilon = 1e-5

        loss = list()
        for i in range(N):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / N
        return loss
