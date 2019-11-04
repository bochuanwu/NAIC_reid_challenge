# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    mode = 'retrieval'
    # optimization options
    label_smooth = 'on'
    use_center = False
    sampler = 'softmax_triplet' #softmax, triplet， softmax_triplet
    sampler_new = True
    loss_type = 'triplet' #'triplet_center'， 'triplet'， 'center', 'softmax', 'softmax_triplet'
    center_weight = 0.0005
    optim = 'adam'
    max_epoch = 150
    train_batch = 128
    test_batch = 32
    adjust_lr = True
    lr = 0.00035
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    margin = None
    num_instances = 4
    num_gpu = 1
    evaluate = False
    savefig = None
    eval_flip = False

    #data augment
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PROB = 0.5
    RE_PROB = 0.5
    PIXEL_STD = [0.229, 0.224, 0.225]
    PADDING = 10
    SIZE_TRAIN = [256, 128]
    SIZE_TEST = [256, 128]



    # model option
    model_name = 'resnet50'  # triplet, softmax_triplet, bfe, ide
    last_stride = 1
    pretrained_model = '/home/zhoumi/.torch/models/resnet50-19c8e357.pth'
    bnneck = 'bnneck'  # bnneck, no
    MHN_parts = 6
    pretrained_choice = 'imagenet' #'imagenet' or 'self'

    # test option
    neck_feat = 'after' #before after

    # miscs
    print_freq = 10
    eval_step = 10
    save_dir = './pytorch-ckpt/market'
    workers = 10
    start_epoch = 0
    best_rank = -np.inf
    norm = False

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}


opt = DefaultConfig()
