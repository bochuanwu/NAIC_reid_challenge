#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/19 10:31 下午
# @author  : wuh-xmu
# @FileName: stack_pcb.py
# @Software: PyCharm
import torch
import torch.nn as nn
import copy
import gc

from models import weights_init_kaiming, resnet50_ibn_a, weights_init_classifier

__all__ = ['spcb_resnet50_512']


class Flatten(nn.Module):
    def __init__(self, dim):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.flatten(self.dim)


class StackPCB(nn.ModuleList):
    def __init__(self, resnet, num_classes, num_dim=256):
        super(StackPCB, self).__init__()

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        self.part_1 = nn.Sequential(copy.deepcopy(resnet.layer3[1:]), copy.deepcopy(resnet.layer4))
        self.part_2 = nn.Sequential(copy.deepcopy(resnet.layer3[1:]), copy.deepcopy(resnet.layer4))
        self.part_3 = nn.Sequential(copy.deepcopy(resnet.layer3[1:]), copy.deepcopy(resnet.layer4))
        self.part_4 = nn.Sequential(copy.deepcopy(resnet.layer3[1:]), copy.deepcopy(resnet.layer4))
        self.part_5 = nn.Sequential(copy.deepcopy(resnet.layer3[1:]), copy.deepcopy(resnet.layer4))

        self.global_pool = nn.AvgPool2d((24, 8))
        self.local_pool_1 = nn.MaxPool2d((4, 8), stride=(4, 8))
        self.local_pool_2 = nn.MaxPool2d((8, 8), stride=(4, 8))
        self.local_pool_3 = nn.MaxPool2d((12, 8), stride=(4, 8))
        self.local_pool_4 = nn.MaxPool2d((16, 8), stride=(4, 8))
        self.local_pool_5 = nn.MaxPool2d((20, 8), stride=(4, 8))

        local_bnneck = nn.BatchNorm1d(num_dim)
        local_bnneck.bias.requires_grad_(False)
        local_bnneck.apply(weights_init_kaiming)

        reduction = nn.Sequential(
            nn.Conv2d(2048, num_dim, 1, bias=False),
            nn.BatchNorm2d(num_dim),
            nn.ReLU(),
            local_bnneck
        )

        reduction.apply(weights_init_kaiming)

        self.p1_f1_embedding = copy.deepcopy(reduction)
        self.p1_f2_embedding = copy.deepcopy(reduction)
        self.p1_f3_embedding = copy.deepcopy(reduction)
        self.p1_f4_embedding = copy.deepcopy(reduction)
        self.p1_f5_embedding = copy.deepcopy(reduction)
        self.p1_f6_embedding = copy.deepcopy(reduction)

        self.p2_f1_embedding = copy.deepcopy(reduction)
        self.p2_f2_embedding = copy.deepcopy(reduction)
        self.p2_f3_embedding = copy.deepcopy(reduction)
        self.p2_f4_embedding = copy.deepcopy(reduction)
        self.p2_f5_embedding = copy.deepcopy(reduction)

        self.p3_f1_embedding = copy.deepcopy(reduction)
        self.p3_f2_embedding = copy.deepcopy(reduction)
        self.p3_f3_embedding = copy.deepcopy(reduction)
        self.p3_f4_embedding = copy.deepcopy(reduction)

        self.p4_f1_embedding = copy.deepcopy(reduction)
        self.p4_f2_embedding = copy.deepcopy(reduction)
        self.p4_f3_embedding = copy.deepcopy(reduction)

        self.p5_f1_embedding = copy.deepcopy(reduction)
        self.p5_f2_embedding = copy.deepcopy(reduction)

        self.p1_global_bn = copy.deepcopy(bnneck)
        self.p2_global_bn = copy.deepcopy(bnneck)
        self.p3_global_bn = copy.deepcopy(bnneck)
        self.p4_global_bn = copy.deepcopy(bnneck)
        self.p5_global_bn = copy.deepcopy(bnneck)
        self.p1_global_bn = copy.deepcopy(bnneck)

        classifier = nn.Linear(num_dim, num_classes, bias=False)
        classifier.apply(weights_init_classifier)

        self.p1_f1_classifier = copy.deepcopy(classifier)
        self.p1_f2_classifier = copy.deepcopy(classifier)
        self.p1_f3_classifier = copy.deepcopy(classifier)
        self.p1_f4_classifier = copy.deepcopy(classifier)
        self.p1_f5_classifier = copy.deepcopy(classifier)
        self.p1_f6_classifier = copy.deepcopy(classifier)

        self.p2_f1_classifier = copy.deepcopy(classifier)
        self.p2_f2_classifier = copy.deepcopy(classifier)
        self.p2_f3_classifier = copy.deepcopy(classifier)
        self.p2_f4_classifier = copy.deepcopy(classifier)
        self.p2_f5_classifier = copy.deepcopy(classifier)

        self.p3_f1_classifier = copy.deepcopy(classifier)
        self.p3_f2_classifier = copy.deepcopy(classifier)
        self.p3_f3_classifier = copy.deepcopy(classifier)
        self.p3_f4_classifier = copy.deepcopy(classifier)

        self.p4_f1_classifier = copy.deepcopy(classifier)
        self.p4_f2_classifier = copy.deepcopy(classifier)
        self.p4_f3_classifier = copy.deepcopy(classifier)

        self.p5_f1_classifier = copy.deepcopy(classifier)
        self.p5_f2_classifier = copy.deepcopy(classifier)

        self.p1_global_classifier = copy.deepcopy(classifier)
        self.p2_global_classifier = copy.deepcopy(classifier)
        self.p3_global_classifier = copy.deepcopy(classifier)
        self.p4_global_classifier = copy.deepcopy(classifier)
        self.p5_global_classifier = copy.deepcopy(classifier)

        del resnet
        del classifier
        del bnneck, reduction
        gc.collect()

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.part_1(x)
        p2 = self.part_2(x)
        p3 = self.part_3(x)
        p4 = self.part_4(x)
        p5 = self.part_5(x)

        p1_global_feature = self.global_pool(p1)
        p2_global_feature = self.global_pool(p2)
        p3_global_feature = self.global_pool(p3)
        p4_global_feature = self.global_pool(p4)
        p5_global_feature = self.global_pool(p5)

        p1_global_triplet_feature = self.p1_embedding(p1_global_feature).flatten(1)
        p2_global_triplet_feature = self.p2_embedding(p2_global_feature).flatten(1)
        p3_global_triplet_feature = self.p3_embedding(p3_global_feature).flatten(1)
        p4_global_triplet_feature = self.p4_embedding(p4_global_feature).flatten(1)
        p5_global_triplet_feature = self.p5_embedding(p5_global_feature).flatten(1)

        p1_global_softmax_feature = self.p1_global_bn(p1_global_triplet_feature)
        p2_global_softmax_feature = self.p2_global_bn(p2_global_triplet_feature)
        p3_global_softmax_feature = self.p3_global_bn(p3_global_triplet_feature)
        p4_global_softmax_feature = self.p4_global_bn(p4_global_triplet_feature)
        p5_global_softmax_feature = self.p5_global_bn(p5_global_triplet_feature)

        p1 = self.local_pool_1(p1)
        p2 = self.local_pool_2(p2)
        p3 = self.local_pool_3(p3)
        p4 = self.local_pool_4(p4)
        p5 = self.local_pool_5(p5)

        p1_f1_triplet_feature = self.p1_f1_embedding(p1[:, :, 0:1, :]).flatten(1)
        p1_f2_triplet_feature = self.p1_f2_embedding(p1[:, :, 1:2, :]).flatten(1)
        p1_f3_triplet_feature = self.p1_f3_embedding(p1[:, :, 2:3, :]).flatten(1)
        p1_f4_triplet_feature = self.p1_f4_embedding(p1[:, :, 3:4, :]).flatten(1)
        p1_f5_triplet_feature = self.p1_f5_embedding(p1[:, :, 4:5, :]).flatten(1)
        p1_f6_triplet_feature = self.p1_f6_embedding(p1[:, :, 5:6, :]).flatten(1)

        p2_f1_triplet_feature = self.p2_f1_embedding(p2[:, :, 0:1, :]).flatten(1)
        p2_f2_triplet_feature = self.p2_f2_embedding(p2[:, :, 1:2, :]).flatten(1)
        p2_f3_triplet_feature = self.p2_f3_embedding(p2[:, :, 2:3, :]).flatten(1)
        p2_f4_triplet_feature = self.p2_f4_embedding(p2[:, :, 3:4, :]).flatten(1)
        p2_f5_triplet_feature = self.p2_f5_embedding(p2[:, :, 4:5, :]).flatten(1)

        p3_f1_triplet_feature = self.p3_f1_embedding(p3[:, :, 0:1, :]).flatten(1)
        p3_f2_triplet_feature = self.p3_f2_embedding(p3[:, :, 1:2, :]).flatten(1)
        p3_f3_triplet_feature = self.p3_f3_embedding(p3[:, :, 2:3, :]).flatten(1)
        p3_f4_triplet_feature = self.p3_f4_embedding(p3[:, :, 3:4, :]).flatten(1)

        p4_f1_triplet_feature = self.p4_f1_embedding(p4[:, :, 0:1, :]).flatten(1)
        p4_f2_triplet_feature = self.p4_f2_embedding(p4[:, :, 1:2, :]).flatten(1)
        p4_f3_triplet_feature = self.p4_f3_embedding(p4[:, :, 2:3, :]).flatten(1)

        p5_f1_triplet_feature = self.p5_f1_embedding(p5[:, :, 0:1, :]).flatten(1)
        p5_f2_triplet_feature = self.p5_f2_embedding(p5[:, :, 1:2, :]).flatten(1)

        p1_f1_softmax_feature = self.p1_f1_bn(p1_f1_triplet_feature)
        p1_f2_softmax_feature = self.p1_f2_bn(p1_f2_triplet_feature)
        p1_f3_softmax_feature = self.p1_f3_bn(p1_f3_triplet_feature)
        p1_f4_softmax_feature = self.p1_f4_bn(p1_f4_triplet_feature)
        p1_f5_softmax_feature = self.p1_f5_bn(p1_f5_triplet_feature)
        p1_f6_softmax_feature = self.p1_f6_bn(p1_f6_triplet_feature)

        p2_f1_softmax_feature = self.p2_f1_bn(p2_f1_triplet_feature)
        p2_f2_softmax_feature = self.p2_f2_bn(p2_f2_triplet_feature)
        p2_f3_softmax_feature = self.p2_f3_bn(p2_f3_triplet_feature)
        p2_f4_softmax_feature = self.p2_f4_bn(p2_f4_triplet_feature)
        p2_f5_softmax_feature = self.p2_f5_bn(p2_f5_triplet_feature)

        p3_f1_softmax_feature = self.p3_f1_bn(p3_f1_triplet_feature)
        p3_f2_softmax_feature = self.p3_f2_bn(p3_f2_triplet_feature)
        p3_f3_softmax_feature = self.p3_f3_bn(p3_f3_triplet_feature)
        p3_f4_softmax_feature = self.p3_f4_bn(p3_f4_triplet_feature)

        p4_f1_softmax_feature = self.p4_f1_bn(p4_f1_triplet_feature)
        p4_f2_softmax_feature = self.p4_f2_bn(p4_f2_triplet_feature)
        p4_f3_softmax_feature = self.p4_f3_bn(p4_f3_triplet_feature)

        p5_f1_softmax_feature = self.p5_f1_bn(p5_f1_triplet_feature)
        p5_f2_softmax_feature = self.p5_f2_bn(p5_f2_triplet_feature)

        global_softmax_features = [p1_global_softmax_feature, p2_global_softmax_feature,
                                   p3_global_softmax_feature, p4_global_softmax_feature, p5_global_softmax_feature]

        p1_softmax_features = [p1_f1_softmax_feature, p1_f2_softmax_feature, p1_f3_softmax_feature,
                               p1_f4_softmax_feature, p1_f5_softmax_feature, p1_f6_softmax_feature]
        p2_softmax_features = [p2_f1_softmax_feature, p2_f2_softmax_feature, p2_f3_softmax_feature,
                               p2_f4_softmax_feature, p2_f5_softmax_feature]
        p3_softmax_features = [p3_f1_softmax_feature, p3_f2_softmax_feature, p3_f3_softmax_feature,
                               p3_f4_softmax_feature]
        p4_softmax_features = [p4_f1_softmax_feature, p4_f2_softmax_feature, p4_f3_softmax_feature]
        p5_softmax_features = [p5_f1_softmax_feature, p5_f2_softmax_feature]

        global_triplet_features = [p1_global_triplet_feature, p2_global_triplet_feature,
                                   p3_global_triplet_feature, p4_global_triplet_feature, p5_global_triplet_feature]

        # p1_triplet_features = [p1_f1_triplet_feature, p1_f2_triplet_feature, p1_f3_triplet_feature,
        #                        p1_f4_triplet_feature, p1_f5_triplet_feature, p1_f6_triplet_feature]
        # p2_triplet_features = [p2_f1_triplet_feature, p2_f2_triplet_feature, p2_f3_triplet_feature,
        #                        p2_f4_triplet_feature, p2_f5_triplet_feature]
        # p3_triplet_features = [p3_f1_triplet_feature, p3_f2_triplet_feature, p3_f3_triplet_feature,
        #                        p3_f4_triplet_feature]
        # p4_triplet_features = [p4_f1_triplet_feature, p4_f2_triplet_feature, p4_f3_triplet_feature]
        # p5_triplet_features = [p5_f1_triplet_feature, p5_f2_triplet_feature]

        local_softmax_features = []
        local_softmax_features += p1_softmax_features
        local_softmax_features += p2_softmax_features
        local_softmax_features += p3_softmax_features
        local_softmax_features += p4_softmax_features
        local_softmax_features += p5_softmax_features

        f = global_softmax_features + local_softmax_features
        if not self.training:
            return torch.cat(f, 1)

        p1_f1_y = self.p1_f1_classifier(p1_f1_softmax_feature)
        p1_f2_y = self.p1_f2_classifier(p1_f2_softmax_feature)
        p1_f3_y = self.p1_f3_classifier(p1_f3_softmax_feature)
        p1_f4_y = self.p1_f4_classifier(p1_f4_softmax_feature)
        p1_f5_y = self.p1_f5_classifier(p1_f5_softmax_feature)
        p1_f6_y = self.p1_f6_classifier(p1_f6_softmax_feature)

        p2_f1_y = self.p2_f1_classifier(p2_f1_softmax_feature)
        p2_f2_y = self.p2_f2_classifier(p2_f2_softmax_feature)
        p2_f3_y = self.p2_f3_classifier(p2_f3_softmax_feature)
        p2_f4_y = self.p2_f4_classifier(p2_f4_softmax_feature)
        p2_f5_y = self.p2_f5_classifier(p2_f5_softmax_feature)

        p3_f1_y = self.p3_f1_classifier(p3_f1_softmax_feature)
        p3_f2_y = self.p3_f2_classifier(p3_f2_softmax_feature)
        p3_f3_y = self.p3_f3_classifier(p3_f3_softmax_feature)
        p3_f4_y = self.p3_f4_classifier(p3_f4_softmax_feature)

        p4_f1_y = self.p4_f1_classifier(p4_f1_softmax_feature)
        p4_f2_y = self.p4_f2_classifier(p4_f2_softmax_feature)
        p4_f3_y = self.p4_f3_classifier(p4_f3_softmax_feature)

        p5_f1_y = self.p5_f1_classifier(p5_f1_softmax_feature)
        p5_f2_y = self.p5_f2_classifier(p5_f2_softmax_feature)

        p1_global_y = self.p1_global_classifier(p1_global_softmax_feature)
        p2_global_y = self.p2_global_classifier(p2_global_softmax_feature)
        p3_global_y = self.p3_global_classifier(p3_global_softmax_feature)
        p4_global_y = self.p4_global_classifier(p4_global_softmax_feature)
        p5_global_y = self.p5_global_classifier(p5_global_softmax_feature)

        y = [p1_global_y, p2_global_y, p3_global_y, p4_global_y, p5_global_y,
             p1_f1_y, p1_f2_y, p1_f3_y, p1_f4_y, p1_f5_y, p1_f6_y,
             p2_f1_y, p2_f2_y, p2_f3_y, p2_f4_y, p2_f5_y,
             p3_f1_y, p3_f2_y, p3_f3_y, p3_f4_y,
             p4_f1_y, p4_f2_y, p4_f3_y,
             p5_f1_y, p5_f2_y]
        #
        # p1_triplet_features = torch.cat(p1_triplet_features, 1)
        # p2_triplet_features = torch.cat(p2_triplet_features, 1)
        # p3_triplet_features = torch.cat(p3_triplet_features, 1)
        # p4_triplet_features = torch.cat(p4_triplet_features, 1)
        # p5_triplet_features = torch.cat(p5_triplet_features, 1)

        return y, global_triplet_features


def spcb_resnet50_512(num_classes, loss='softmax', **kwargs):
    resnet = resnet50_ibn_a(num_classes, loss=loss, **kwargs)
    model = StackPCB(resnet, num_classes, num_dim=512)
    return model
