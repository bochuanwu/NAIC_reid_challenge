from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from .backbones.resnet_ibn_a import resnet50_ibn_a

from .backbones.attention import CAM_Module, PAM_Module, ShallowCAM
from copy import deepcopy

def init_params(x):

    if x is None:
        return

    for m in x.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class ResNetCommonBranch(nn.Module):

    def __init__(self, owner, backbone):

        super().__init__()

        self.backbone1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1
        )
        self.shallow_cam = ShallowCAM(True, 256)
        init_params(self.shallow_cam)
        self.backbone2 = nn.Sequential(
            backbone.layer2,
            backbone.layer3,
        )

    def backbone_modules(self):

        return [self.backbone1, self.backbone2]

    def forward(self, x):
        x = self.backbone1(x)

        intermediate = x = self.shallow_cam(x)
        x = self.backbone2(x)

        return x, intermediate

class ResNetDeepBranch(nn.Module):

    def __init__(self, owner, backbone):

        super().__init__()

        self.backbone = deepcopy(backbone.layer4)

        self.out_dim = 2048

    def backbone_modules(self):

        return [self.backbone]

    def forward(self, x):
        return self.backbone(x)

class DropBlock2D(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            # mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            mask=self._compute_mask(x,gamma)
            # print('mask:', mask)

            # print(mask.shape)
            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # print('block_mask:', block_mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        h=x.shape[-2]
        w=x.shape[-1]
        # print('h', h)
        # print('w', w)
        gamma=(1. - self.drop_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))
        # print('gamma:',gamma)
        # return self.drop_prob / (self.block_size ** 2)
        return gamma
    def _compute_mask(self,x,gamma):
        block_size= self.block_size
        mask=(torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        base_mask = torch.ones_like(mask)
        base_mask[:, int(block_size / 2):base_mask.shape[1] - int(block_size / 2),
        int(block_size / 2):base_mask.shape[2] - int(block_size / 2)] = 0
        base_mask =1 - base_mask
        # print(mask)
        # print(base_mask)
        mask=mask*base_mask
        return mask

class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)[::-1]

    def forward(self, x):
        if self.training==False:
            self.dropblock.drop_prob=1.
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1

class resnet50_ibn_dropblock_pa_ca(nn.Module):
    def __init__(self,num_classes, model_path, pretrain_choice, bn='bn',pooling='avg', last_stride = 2):
        super(resnet50_ibn_dropblock_pa_ca,self).__init__()
        backbone=resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.common_branch=ResNetCommonBranch(self,backbone)
        self.deep_branch_1=ResNetDeepBranch(self,backbone)
        self.deep_branch_2=ResNetDeepBranch(self,backbone)
        self.com_pa=PAM_Module(1024)
        self.com_ca=CAM_Module(1024)
        self.deep1_pa=PAM_Module(2048)
        self.deep1_ca=CAM_Module(2048)

        init_params(self.com_pa)
        init_params(self.com_ca)
        init_params(self.deep1_pa)
        init_params(self.deep1_ca)


        sum_conv_com = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        init_params(sum_conv_com)

        sum_conv_deep = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(2048, 2048, kernel_size=1)
        )

        init_params(sum_conv_deep)

        self.sum_conv_com = sum_conv_com
        self.sum_conv_deep=sum_conv_deep
        if pooling=='avg':
            self.gap=nn.AdaptiveAvgPool2d(1)

        self.att_clf=nn.Linear(2048,num_classes)
        self.ori_clf=nn.Linear(2048,num_classes)
        init_params(self.att_clf)
        init_params(self.ori_clf)


        if bn=='bn':
            self.com_bn=nn.BatchNorm1d(1024)
            self.com_att_bn=nn.BatchNorm1d(1024)
            self.deep_bn=nn.BatchNorm1d(2048)
            self.deep_att_bn=nn.BatchNorm1d(2048)

        init_params(self.com_bn)
        init_params(self.com_att_bn)
        init_params(self.deep_bn)
        init_params(self.deep_att_bn)


            # self
    def forward(self, x):
        com_x=self.common_branch(x)[0] #output_1
        com_x_pa=self.com_pa(com_x)
        com_x_ca=self.com_ca(com_x)
        com_sum=self.sum_conv_com(com_x+com_x_pa+com_x_ca) #output_2

        deep_x=self.deep_branch_2(com_x) #output_3

        com_att_deep_x=self.deep_branch_1(com_sum)
        com_att_deep_x_pa=self.deep1_pa(com_att_deep_x)
        com_att_deep_x_ca=self.deep1_ca(com_att_deep_x)
        com_att_deep_sum=self.sum_conv_deep(com_att_deep_x+com_att_deep_x_ca+com_att_deep_x_pa) # output_4


        com_x=self.com_bn(self.gap(com_x).flatten(1))
        com_sum=self.com_att_bn(self.gap(com_sum).flatten(1))

        deep_x=self.deep_bn(self.gap(deep_x).flatten(1))
        com_att_deep_sum=self.deep_att_bn(self.gap(com_att_deep_sum).flatten(1))

        clf_output_ori=self.ori_clf(deep_x)
        clf_output_att=self.att_clf(com_att_deep_sum)
        # print('com_x',com_x.shape)
        # print('com_att_deep_sum',com_att_deep_sum.shape)
        ft = [com_x, com_sum, deep_x, com_att_deep_sum]
        clf_output=[clf_output_ori, clf_output_att]

        if self.training:
            return clf_output, ft
        else:
            return ft

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        print('load:', trained_path)
        for i in param_dict:
            if i not in self.state_dict() or 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def get_optim_policy(self):
        return self.parameters()