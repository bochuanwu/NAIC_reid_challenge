# encoding: utf-8
"""
@author:  zhoumi
@contact: zhoumi281571814@126.com
"""

from .baseline import Baseline
from .pcb import pcb_p6, pcb_p4
from .MGN import MGN
from .stack_pcbv2 import StackPCBv2
from .stack_pcb import StackPCB
from .drop_block import resnet50_ibn_dropblock_pa_ca

def build_model(opt):

    if opt.model_name == "pcb":
        model = pcb_p6(num_classes=opt.NUM_CLASS, neck = opt.bnneck, neck_feat=opt.neck_feat)
    elif opt.model_name == "MGN":
        model =MGN(opt.NUM_CLASS, opt.pretrained_choice, opt.pretrained_model,
                   opt.bnneck, opt.neck_feat, attention=opt.attention, sep_bn=opt.sep_bn, last_stride = 2, pool = 'avg', feats = opt.feat)
    elif opt.model_name == 'StackPCBv2':
        model = StackPCBv2(opt.NUM_CLASS, opt.pretrained_choice, opt.pretrained_model, opt.bnneck, opt.neck_feat, last_stride = 2)
    elif opt.model_name == 'StackPCB':
        model = StackPCB(opt.NUM_CLASS, opt.pretrained_choice, opt.pretrained_model, last_stride = 1)
    elif opt.model_name == 'drop_block':
        model = resnet50_ibn_dropblock_pa_ca(opt.NUM_CLASS, opt.pretrained_model, opt.pretrained_choice, last_stride = 2)
    else:
        model = Baseline(opt.NUM_CLASS, opt.last_stride, opt.pretrained_model,
                         opt.bnneck, opt.neck_feat, opt.model_name, opt.pretrained_choice)
    return model
