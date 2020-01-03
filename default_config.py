import argparse
from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()

    # model
    cfg.model = CN()
    cfg.model.name = 'resnet50'
    cfg.model.pretrained = True  # automatically load pretrained model weights if available
    cfg.model.load_weights = ''  # path to model weights
    cfg.model.resume = ''  # path to checkpoint for resume training

    # data
    cfg.data = CN()
    cfg.data.type = 'image'
    cfg.data.root = 'input'
    cfg.data.sources = ['market1501']
    cfg.data.targets = ['market1501']
    cfg.data.workers = 8  # number of data loading workers
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.transforms = ['random_flip']  # data augmentation
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.erase_mean = [0.175, 0.214, 0.247] # default is imagenet mean
    cfg.data.save_dir = 'log'  # path to save log
    cfg.data.is_train = True
    cfg.data.appended_train = False
    cfg.data.appended_gallery = False
    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = 'RandomSampler'
    cfg.sampler.num_instances = 4  # number of instances per identity for RandomIdentitySampler

    # train
    cfg.train = CN()
    cfg.train.optim = 'adam'
    cfg.train.lr = 0.0003
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 120
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 64
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers
    cfg.train.open_layers = []  # layers for training while keeping others frozen
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.base_layers = []  # based layer with default lr * base_lr_mult
    cfg.train.base_lr_mult = 0.1  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = 'warmup_linear'
    cfg.train.stepsize = [40, 70]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier or warmup
    cfg.train.print_freq = 40  # print frequency
    cfg.train.seed = 1  # random seed #
    cfg.train.warmup_factor = 0.01  # warmup rate
    cfg.train.warmup_iters = 10  # warmup steps
    cfg.train.apex = False

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.loss_reduce = 'mean'
    cfg.loss.name = 'softmax'
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.  # distance margin for triplet loss
    cfg.loss.triplet.weight_t = 1.  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 1.  # weight to balance cross entropy loss
    cfg.loss.triplet.metric = 'euclidean'  # metric for dist matrix in triplet loss
    cfg.loss.triplet.ranked_loss = False
    cfg.loss.triplet.ms_loss = False
    cfg.loss.dynamic = CN()
    cfg.loss.dynamic.T = 2

    # test
    cfg.test = CN()
    cfg.test.batch_size = 100
    cfg.test.dist_metric = 'euclidean'  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = False  # normalize feature vectors before computing distance
    cfg.test.ranks = [1, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.eval_freq = -1  # evaluation frequency (-1 means to only test after training)
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = True  # use person re-ranking
    cfg.test.visrank = False  # visualize ranked results (only available when cfg.test.evaluate=True)
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visactmap = False  # visualize CNN activation maps

    # rerank
    cfg.rerank = CN()
    cfg.rerank.k1 = 6
    cfg.rerank.k2 = 2
    cfg.rerank.lambda_value = 0.3

    return cfg


def imagedata_kwargs(cfg):
    return {
        'root': cfg.data.root,
        'sources': cfg.data.sources,
        'targets': cfg.data.targets,
        'height': cfg.data.height,
        'width': cfg.data.width,
        'transforms': cfg.data.transforms,
        'norm_mean': cfg.data.norm_mean,
        'norm_std': cfg.data.norm_std,
        'erase_mean': cfg.data.erase_mean,
        'use_gpu': cfg.use_gpu,
        'combineall': cfg.data.combineall,
        'batch_size_train': cfg.train.batch_size,
        'batch_size_test': cfg.test.batch_size,
        'workers': cfg.data.workers,
        'num_instances': cfg.sampler.num_instances,
        'train_sampler': cfg.sampler.train_sampler,
        'is_train': cfg.data.is_train,
        'appended_train': cfg.data.appended_train,
        'appended_gallery': cfg.data.appended_gallery,
    }


def optimizer_kwargs(cfg):
    return {
        'optim': cfg.train.optim,
        'lr': cfg.train.lr,
        'weight_decay': cfg.train.weight_decay,
        'momentum': cfg.sgd.momentum,
        'sgd_dampening': cfg.sgd.dampening,
        'sgd_nesterov': cfg.sgd.nesterov,
        'rmsprop_alpha': cfg.rmsprop.alpha,
        'adam_beta1': cfg.adam.beta1,
        'adam_beta2': cfg.adam.beta2,
        'staged_lr': cfg.train.staged_lr,
        'base_layers': cfg.train.base_layers,
        'base_lr_mult': cfg.train.base_lr_mult
    }


def lr_scheduler_kwargs(cfg):
    return {
        'lr_scheduler': cfg.train.lr_scheduler,
        'stepsize': cfg.train.stepsize,
        'gamma': cfg.train.gamma,
        'max_epoch': cfg.train.max_epoch,
        'warmup_factor': 0.01,
        'warmup_iters': 10
    }


def engine_run_kwargs(cfg):
    return {
        'save_dir': cfg.data.save_dir,
        'max_epoch': cfg.train.max_epoch,
        'start_epoch': cfg.train.start_epoch,
        'fixbase_epoch': cfg.train.fixbase_epoch,
        'open_layers': cfg.train.open_layers,
        'start_eval': cfg.test.start_eval,
        'eval_freq': cfg.test.eval_freq,
        'test_only': cfg.test.evaluate,
        'print_freq': cfg.train.print_freq,
        'dist_metric': cfg.test.dist_metric,
        'normalize_feature': cfg.test.normalize_feature,
        'visrank': cfg.test.visrank,
        'visrank_topk': cfg.test.visrank_topk,
        'ranks': cfg.test.ranks,
        'rerank': cfg.test.rerank,
        'visactmap': cfg.test.visactmap,
        'k1': cfg.rerank.k1,
        'k2': cfg.rerank.k2,
        'lambda_value': cfg.rerank.lambda_value,
        'loss_reduce': cfg.loss.loss_reduce
    }
