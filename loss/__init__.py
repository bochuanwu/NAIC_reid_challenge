# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .rank_loss import RankedLoss


def make_loss(opt):
    num_classes = opt.NUM_CLASS
    sampler = opt.sampler
    if opt.loss_type == 'triplet':
        triplet = TripletLoss(opt.margin)
    elif opt.loss_type == 'rank':
        triplet = RankedLoss()
    else:
        print('expected loss_type should be triplet'
              'but got {}'.format(opt.loss_type))

    if opt.label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif opt.sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif opt.sampler == 'softmax_triplet':
        def loss_func(score, feat, target, weight_s = 1.0, weight_t = 1.0):
            if opt.loss_type == 'triplet' or opt.loss_type == 'rank':
                if opt.label_smooth == 'on':
                    return weight_s * xent(score, target) + weight_t * opt.triplet_weight * triplet(feat, target)[0]
                else:
                    return weight_s * F.cross_entropy(score, target) + weight_t *opt.triplet_weight * triplet(feat, target)[0]
            else:
                print('expected loss_type should be triplet'
                      'but got {}'.format(opt.loss_type))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(opt.sampler))
    return loss_func


def make_loss_with_center(opt):    # modified by gu
    num_classes = opt.NUM_CLASS
    if opt.model_name == 'resnet18' or opt.model_name == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if opt.loss_type == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif opt.loss_type == 'triplet_center':
        triplet = TripletLoss(opt.margin)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected loss_type with center should be center, triplet_center'
              'but got {}'.format(opt.loss_type))

    if opt.label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if opt.loss_type == 'center':
            if opt.label_smooth == 'on':
                return xent(score, target) + \
                        opt.center_weight * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        opt.center_weight * center_criterion(feat, target)

        elif opt.loss_type == 'triplet_center':
            if opt.label_smooth == 'on':
                return xent(score, target) + \
                        triplet(feat, target)[0] + \
                        opt.center_weight * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        opt.center_weight * center_criterion(feat, target)

        else:
            print('expected loss_type with center should be center, triplet_center'
                  'but got {}'.format(opt.loss_type))
    return loss_func