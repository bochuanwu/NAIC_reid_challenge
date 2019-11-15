# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline
from .pcb import pcb_p6, pcb_p4
from .MGN import MGN


def build_model(opt):

    if opt.model_name == "pcb":
        model = pcb_p6(num_classes=opt.NUM_CLASS, neck = opt.bnneck, neck_feat=opt.neck_feat)
    elif opt.model_name == "MGN":
        model =MGN(opt.NUM_CLASS, opt.pretrained_choice, opt.pretrained_model, opt.bnneck, opt.neck_feat, last_stride = 2, pool = 'avg', feats = 256)
    else:
        model = Baseline(opt.NUM_CLASS, opt.last_stride, opt.pretrained_model, opt.bnneck, opt.neck_feat, opt.model_name, opt.pretrained_choice)
    return model
