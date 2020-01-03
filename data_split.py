#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/12/4 12:17 下午
# @author  : wuh-xmu
# @FileName: data_split.py
# @Software: PyCharm

import os.path as osp
import copy
import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

seed = 1
ration = 0.7
one_sampler_add_gallery = True
random.seed(seed)
np.random.seed(seed)
second_stage_data = 'input/train/second_stage_train_list_refine.txt'
save_dir = './input/train'

img_names = []
pids = []
with open(second_stage_data, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        img_name, pid = line.split(' ')
        img_name = osp.basename(img_name)
        img_names.append(img_name)
        pids.append(pid)
pids_counter = Counter(pids)

gallery_pids = []
query_pids = []

avai_pids = copy.deepcopy(pids_counter)
if one_sampler_add_gallery:
    for pid, num_samplers in pids_counter.items():
        if num_samplers == 1:
            gallery_pids.append(pid)
            avai_pids.pop(pid)
avai_pids = list(avai_pids.keys())

# random sample pid for train
train_pids = np.random.choice(avai_pids, size=round(len(avai_pids) * ration), replace=False)
query_pids = [pid for pid in avai_pids if pid not in train_pids]
gallery_pids.extend([pid for pid in avai_pids if pid not in train_pids])

train_x = []
train_y = []

test_set = defaultdict(list)

query_x = []
query_y = []
gallery_x = []
gallery_y = []

# split

for img_name, pid in zip(img_names, pids):
    if pid in train_pids:
        train_x.append(img_name)
        train_y.append(pid)
    else:
        test_set[pid].append(img_name)

for pid, imgs in test_set.items():
    if len(imgs) == 1:
        gallery_x.append(imgs[0])
        gallery_y.append(pid)
    else:
        query_img = random.sample(imgs, 1)[0]
        query_x.append(query_img)
        query_y.append(pid)
        for img in imgs:
            if img != query_img:
                gallery_x.append(img)
                gallery_y.append(pid)
# save
pd.DataFrame({
    'img_name': train_x,
    'label': train_y}
).to_csv(
    osp.join(save_dir, 'train.txt'),
    index=None,
    header=None,
    sep=' '
)
pd.DataFrame({
    'img_name': query_x,
    'label': query_y}
).to_csv(
    osp.join(save_dir, 'query.txt'),
    index=None,
    header=None,
    sep=' '
)
pd.DataFrame({
    'img_name': gallery_x,
    'label': gallery_y}
).to_csv(
    osp.join(save_dir, 'gallery.txt'),
    index=None,
    header=None,
    sep=' '
)


