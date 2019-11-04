# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from datasets.init_dataset import Tx_dataset
from datasets.dataset_loader import ImageDataset
from models import build_model
from logger import Logger
from transformer import build_transforms
from config import opt
from evaluator import Evaluator
import json

def test(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing tx_chanllege dataset')

    pin_memory = True if use_gpu else False
    query_dataset = Tx_dataset(set='query_a', file_list='query_a_list.txt').dataset
    gallery_dataset = Tx_dataset(set='gallery_a', file_list='gallery_a_list.txt').dataset


    queryloader = DataLoader(
        ImageDataset(query_dataset, transform=build_transforms(opt, is_train=False)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory)

    galleryloader = DataLoader(
        ImageDataset(gallery_dataset, transform=build_transforms(opt, is_train=False)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory)

    queryFliploader = DataLoader(
        ImageDataset(query_dataset, transform=build_transforms(opt, is_train=False, flip=True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageDataset(gallery_dataset, transform=build_transforms(opt, is_train=False, flip=True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')

    model = build_model(opt, num_classes=4768)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = Evaluator(model, norm=opt.norm)

    results = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader, eval_flip=True)

    # reid_evaluator.validation(queryloader, galleryloader)

    with open('./result/submission_example_A.json', "w", encoding='utf-8') as fd:
        json.dump(results, fd)

if __name__ == '__main__':
    import fire
    fire.Fire()
