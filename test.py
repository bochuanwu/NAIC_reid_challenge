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
    # torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    # sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

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

    model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = Evaluator(model, norm=opt.norm, eval_flip=opt.eval_flip, re_ranking=opt.re_ranking)

    results = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader, k1=6, k2=2, lambda_value=0.3)

    # reid_evaluator.validation(queryloader, galleryloader)

    with open('./result/submission_example_A.json', "w", encoding='utf-8') as fd:
        json.dump(results, fd)

def multi_test(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    # torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    # sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

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

    model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    # opt.model_name = "pcb"

    pcb_model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load('/data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/r50_ibn_a_bigsize_era/model_best.pth.tar')['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        pcb_model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('pcb model size: {:.5f}M'.format(sum(p.numel() for p in pcb_model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
        pcb_model = nn.DataParallel(pcb_model).cuda()

    reid_evaluator = Evaluator(model, pcb_model= pcb_model, norm=opt.norm, eval_flip=opt.eval_flip, re_ranking=opt.re_ranking, concate=True)

    results = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader, k1=6, k2=2, lambda_value=0.3)

    # reid_evaluator.validation(queryloader, galleryloader)

    with open('./result/submission_example_A.json', "w", encoding='utf-8') as fd:
        json.dump(results, fd)

def validate(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    # os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    # sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

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
    query_dataset = Tx_dataset(set='train_set', file_list='val_query_list.txt').dataset
    gallery_dataset = Tx_dataset(set='train_set', file_list='val_gallery_list.txt').dataset


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

    model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = Evaluator(model, norm=opt.norm, eval_flip=opt.eval_flip)

    print("without reranking testing......")
    reid_evaluator.validation(queryloader, galleryloader, queryFliploader, galleryFliploader)

    max_score = 0
    k = 0
    for k1 in range(1, 21):
    # for la in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            print("**********k1:{}***********".format(k1))
            score = reid_evaluator.validation(queryloader, galleryloader, queryFliploader, galleryFliploader, re_ranking=True,
                                              k1=k1, k2=2, lambda_value=0.3)
            if score > max_score:
                max_score = score
                k = k1

    print("max_score: {} at k: {}".format(max_score, k))


    # with open('./result/submission_example_A.json', "w", encoding='utf-8') as fd:
    #     json.dump(results, fd)

def multi_validate(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    # os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    # sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

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
    query_dataset = Tx_dataset(set='train_set', file_list='val_query_list.txt').dataset
    gallery_dataset = Tx_dataset(set='train_set', file_list='val_gallery_list.txt').dataset


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

    model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    opt.model_name = "pcb"

    pcb_model = build_model(opt)

    if opt.pretrained_choice == 'self':
        state_dict = torch.load('/data/zhoumi/train_project/REID/tx_challenge/pytorch-ckpt/pcb_ibn_bnneck/model_best.pth.tar')['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        pcb_model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('pcb model size: {:.5f}M'.format(sum(p.numel() for p in pcb_model.parameters()) / 1e6))


    if use_gpu:
        model = nn.DataParallel(model).cuda()
        pcb_model = nn.DataParallel(pcb_model).cuda()
    reid_evaluator = Evaluator(model, pcb_model=pcb_model, norm=opt.norm, eval_flip=opt.eval_flip,concate=True)

    print("without reranking testing......")
    reid_evaluator.validation(queryloader, galleryloader, queryFliploader, galleryFliploader)

    max_score = 0
    k = 0
    for k1 in range(1, 21):
    # for la in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            print("**********k1:{}***********".format(k1))
            score = reid_evaluator.validation(queryloader, galleryloader, queryFliploader, galleryFliploader, re_ranking=True,
                                              k1=k1, k2=2, lambda_value=0.3)
            if score > max_score:
                max_score = score
                k = k1

    print("max_score: {} at k: {}".format(max_score, k))


    # with open('./result/submission_example_A.json', "w", encoding='utf-8') as fd:
    #     json.dump(results, fd)

if __name__ == '__main__':
    import fire
    fire.Fire()
