# encoding: utf-8
"""
@author:  zhoumi
@contact: zhoumi281571814@126.com
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
# from .backbones.resnet import ResNet, Bottleneck
from .backbones.resnet_ibn_a import ResNet_IBN as ResNet
from .backbones.resnet_ibn_a import Bottleneck_IBN as Bottleneck
from .backbones.attention import CAM_Module, PAM_Module

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class MGN(nn.Module):
    def __init__(self, num_classes, pretrain_choice, model_path, neck, neck_feat,
                 attention = True, sep_bn = True, last_stride = 2, pool = 'avg', feats = 256):
        super(MGN, self).__init__()
        num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.attention = attention
        self.sep_bn = sep_bn

        resnet = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])

        if pretrain_choice == 'imagenet':
            resnet.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.backone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        if pool == 'max':
            pool2d = nn.MaxPool2d
        elif pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()

        self.maxpool_zg_p1 = pool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = pool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = pool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = pool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = pool2d(kernel_size=(8, 8))

        reduction = nn.Sequential(nn.Conv2d(2048, feats, 1, bias=False), nn.BatchNorm2d(feats), nn.ReLU())

        self._init_reduction(reduction)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)
        self.reduction_7 = copy.deepcopy(reduction)

        # self.fc_id_2048_0 = nn.Linear(2048, num_classes)

        if self.neck == 'no':
            self.fc_id_2048_0 = nn.Linear(feats, num_classes)
            self.fc_id_2048_1 = nn.Linear(feats, num_classes)
            self.fc_id_2048_2 = nn.Linear(feats, num_classes)

            self.fc_id_256_1_0 = nn.Linear(feats, num_classes)
            self.fc_id_256_1_1 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_0 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_1 = nn.Linear(feats, num_classes)
            self.fc_id_256_2_2 = nn.Linear(feats, num_classes)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(feats)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.fc_id_2048_0 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_2048_1 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_2048_2 = nn.Linear(feats, num_classes)#, bias=False)

            self.fc_id_256_1_0 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_256_1_1 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_256_2_0 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_256_2_1 = nn.Linear(feats, num_classes)#, bias=False)
            self.fc_id_256_2_2 = nn.Linear(feats, num_classes)#, bias=False)

            self.bottleneck.apply(weights_init_kaiming)

            if self.sep_bn:
                self.bottleneck_fg1 = copy.deepcopy(self.bottleneck)
                self.bottleneck_fg2 = copy.deepcopy(self.bottleneck)
                self.bottleneck_fg3 = copy.deepcopy(self.bottleneck)
                self.bottleneck_l0g2 = copy.deepcopy(self.bottleneck)
                self.bottleneck_l1g2 = copy.deepcopy(self.bottleneck)
                self.bottleneck_l0g3 = copy.deepcopy(self.bottleneck)
                self.bottleneck_l1g3 = copy.deepcopy(self.bottleneck)
                self.bottleneck_l2g3 = copy.deepcopy(self.bottleneck)

        if self.attention:
            self.cam_global = CAM_Module(2048)
            self.pam_global = PAM_Module(2048)
            self.cam_local0 = CAM_Module(2048)
            self.pam_local0 = PAM_Module(2048)
            self.cam_local1 = CAM_Module(2048)
            self.pam_local1 = PAM_Module(2048)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):

        x = self.backone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        if self.attention:
            p1_cam = self.cam_global(p1) + self.pam_global(p1)
            p2_cam = self.cam_local0(p2) + self.pam_local0(p2)
            p3_cam = self.cam_local1(p3) + self.pam_local1(p3)
        else:
            p1_cam = p1
            p2_cam = p2
            p3_cam = p3

        zg_p1 = self.maxpool_zg_p1(p1_cam)
        zg_p2 = self.maxpool_zg_p2(p2_cam)
        zg_p3 = self.maxpool_zg_p3(p3_cam)

        zp2 = self.maxpool_zp2(p2)
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]

        fg_p1 = self.reduction_0(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_1(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_2(zg_p3).squeeze(dim=3).squeeze(dim=2)
        f0_p2 = self.reduction_3(z0_p2).squeeze(dim=3).squeeze(dim=2)
        f1_p2 = self.reduction_4(z1_p2).squeeze(dim=3).squeeze(dim=2)
        f0_p3 = self.reduction_5(z0_p3).squeeze(dim=3).squeeze(dim=2)
        f1_p3 = self.reduction_6(z1_p3).squeeze(dim=3).squeeze(dim=2)
        f2_p3 = self.reduction_7(z2_p3).squeeze(dim=3).squeeze(dim=2)

        '''
        l_p1 = self.fc_id_2048_0(zg_p1.squeeze(dim=3).squeeze(dim=2))
        l_p2 = self.fc_id_2048_1(zg_p2.squeeze(dim=3).squeeze(dim=2))
        l_p3 = self.fc_id_2048_2(zg_p3.squeeze(dim=3).squeeze(dim=2))
        '''

        if self.neck == 'no':
            fg_p1_feat = fg_p1
            fg_p2_feat = fg_p2
            fg_p3_feat = fg_p3
            f0_p2_feat = f0_p2
            f1_p2_feat = f1_p2
            f0_p3_feat = f0_p3
            f1_p3_feat = f1_p3
            f2_p3_feat = f2_p3
        elif self.neck == 'bnneck':
            if self.sep_bn:
                fg_p1_feat = self.bottleneck_fg1(fg_p1)
                fg_p2_feat = self.bottleneck_fg2(fg_p2)
                fg_p3_feat = self.bottleneck_fg3(fg_p3)
                f0_p2_feat = self.bottleneck_l0g2(f0_p2)
                f1_p2_feat = self.bottleneck_l1g2(f1_p2)
                f0_p3_feat = self.bottleneck_l0g3(f0_p3)
                f1_p3_feat = self.bottleneck_l1g3(f1_p3)
                f2_p3_feat = self.bottleneck_l2g3(f2_p3)
            else:
                fg_p1_feat = self.bottleneck(fg_p1)
                fg_p2_feat = self.bottleneck(fg_p2)
                fg_p3_feat = self.bottleneck(fg_p3)
                f0_p2_feat = self.bottleneck(f0_p2)
                f1_p2_feat = self.bottleneck(f1_p2)
                f0_p3_feat = self.bottleneck(f0_p3)
                f1_p3_feat = self.bottleneck(f1_p3)
                f2_p3_feat = self.bottleneck(f2_p3)

        if self.training:
            l_p1 = self.fc_id_2048_0(fg_p1_feat)
            l_p2 = self.fc_id_2048_1(fg_p2_feat)
            l_p3 = self.fc_id_2048_2(fg_p3_feat)

            l0_p2 = self.fc_id_256_1_0(f0_p2_feat)
            l1_p2 = self.fc_id_256_1_1(f1_p2_feat)
            l0_p3 = self.fc_id_256_2_0(f0_p3_feat)
            l1_p3 = self.fc_id_256_2_1(f1_p3_feat)
            l2_p3 = self.fc_id_256_2_2(f2_p3_feat)
            feats = [fg_p1, fg_p2, fg_p3]
            scores = [l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3]
            return scores, feats
        else:
            if self.neck_feat == 'after':
                predict = torch.cat([fg_p1_feat, fg_p2_feat, fg_p3_feat, f0_p2_feat, f1_p2_feat, f0_p3_feat, f1_p3_feat, f2_p3_feat], dim=1)

            else:
                predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)

            return predict

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optim_policy(self):
        return self.parameters()




