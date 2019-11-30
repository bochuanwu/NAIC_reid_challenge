#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/8 5:30 下午
# @author  : wuh-xmu
# @FileName: of_penalty_loss.py
# @Software: PyCharm
import torch.nn as nn
import torch


class OFPenalty(nn.Module):

    def __init__(self):
        super(OFPenalty, self).__init__()

    def dominant_eigenvalue(self, A):
        B, N, _ = A.size()
        x = torch.randn(B, N, 1, device='cuda')

        for _ in range(1):
            x = torch.bmm(A, x)
        # x: 'B x N x 1'
        numerator = torch.bmm(
            torch.bmm(A, x).view(B, 1, N),
            x
        ).squeeze()
        denominator = (torch.norm(x.view(B, N), p=2, dim=1) ** 2).squeeze()

        return numerator / denominator

    def get_singular_values(self, A):
        AAT = torch.bmm(A, A.permute(0, 2, 1))
        B, N, _ = AAT.size()
        largest = self.dominant_eigenvalue(AAT)
        I = torch.eye(N, device='cuda').expand(B, N, N)  # noqa
        I = I * largest.view(B, 1, 1).repeat(1, N, N)  # noqa
        tmp = self.dominant_eigenvalue(AAT - I)
        return tmp + largest, largest

    def forward(self, x):

        batches, channels, height, width = x.size()
        W = x.view(batches, channels, -1)
        smallest, largest = self.get_singular_values(W)
        singular_penalty = largest - smallest

        return singular_penalty.sum() / (x.size(0) / 32.)  # Quirk: normalize to 32-batch case

