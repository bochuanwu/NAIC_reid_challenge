#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 12:30 下午
# @Author  : wuh-xmu
# @FileName: kesicreid.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import print_function

from . import ImageDataset
from collections import defaultdict
import os.path as osp
import pandas as pd
import numpy as np
import shutil
import os
import re
import random
import glob


class KesciReID(ImageDataset):
    dataset_dir = 'kesci-reid'
    raw = 'raw'
    data = 'data'

    def __init__(self, root, is_train=True, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.raw = osp.join(self.dataset_dir, self.raw)
        self.data = osp.join(self.dataset_dir, self.data)

        self.train_dir = osp.join(self.data, 'train')
        self.query_dir = osp.join(self.data, 'val', 'query')
        self.gallery_dir = osp.join(self.data, 'val', 'gallery')

        # required_files = [
        #     self.dataset_dir,
        #     self.train_dir,
        #     self.query_dir,
        #     self.gallery_dir
        # ]
        #
        # if not self.check_before_run(required_files):
        #     self.train_split_val()

        train = self.process_dir(self.train_dir, is_train=is_train, relabel=True)
        query = self.process_dir(self.query_dir, is_train=is_train, relabel=False)
        gallery = self.process_dir(self.gallery_dir, is_train=is_train, relabel=False)

        # to inference
        if not is_train:
            self.query_dir = osp.join(self.data, 'query')
            self.gallery_dir = osp.join(self.data, 'gallery')
            if not osp.exists(self.query_dir):
                shutil.copytree(osp.join(self.raw, 'query_a'), self.query_dir)
            if not osp.exists(osp.join(self.gallery_dir)):
                shutil.copytree(osp.join(self.raw, 'gallery_a'), self.gallery_dir)

            query = self.process_dir(self.query_dir, is_train=is_train, relabel=False)
            gallery = self.process_dir(self.gallery_dir, is_train=is_train, relabel=False)

        super(KesciReID, self).__init__(train, query, gallery, **kwargs)

    def train_split_val(self):

        label_file = 'train_list.txt'
        label_file = osp.join(self.raw, label_file)
        label_file = pd.read_csv(label_file, sep=' ', header=None)
        pids, num_samples = [], []

        for pid, val in label_file[1].value_counts().items():
            pids.append(pid)
            num_samples.append(val)

        pids = np.asarray(pids)
        num_samples = np.asarray(num_samples)

        # 这些id的样本数量太多或太少， 当验证集做检索用
        val_pids = pids[num_samples > 6].tolist()
        val_pids += pids[num_samples < 4].tolist()

        train_pids = set(pids) - set(val_pids)
        train_pids = sorted(list(train_pids))

        train_root = osp.join(self.data, 'train')
        val_root = osp.join(self.data, 'val')
        val_query = osp.join(val_root, 'query')
        val_gallery = osp.join(val_root, 'gallery')
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(val_query, exist_ok=True)
        os.makedirs(val_gallery, exist_ok=True)
        val_imgs = defaultdict(list)

        for idx in range(len(label_file)):
            img_name, pid = label_file.iloc[idx]
            if pid in train_pids:
                basename = osp.basename(img_name)
                shutil.copy(osp.join(self.raw, img_name),
                            osp.join(self.data,
                                     train_root,
                                     "{pid}_{basename}"
                                     .format(pid=pid,
                                             basename=basename)))
            elif pid in val_pids:
                val_imgs[pid].append(img_name)
            else:
                pass

        for pid, samplers in val_imgs.items():
            num_query = round(len(samplers) * 0.35)
            np.random.shuffle(samplers)
            query_imgs = random.sample(samplers, num_query)
            for img_name in query_imgs:
                basename = osp.basename(img_name)
                shutil.copy(osp.join(self.raw, img_name),
                            osp.join(self.data,
                                     val_query,
                                     "{pid}_{basename}"
                                     .format(pid=pid,
                                             basename=basename)))

            gallery_imgs = list(set(samplers) - set(query_imgs))
            for img_name in gallery_imgs:
                basename = osp.basename(img_name)
                shutil.copy(osp.join(self.raw, img_name),
                            osp.join(self.data,
                                     val_gallery,
                                     "{pid}_{basename}"
                                     .format(pid=pid,
                                             basename=basename)))

    def process_dir(self, dir_path, is_train=True, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        data = []
        if is_train:
            pid_container = set()
            for img_path in img_paths:
                pid = osp.basename(img_path).split('_')[0]
                pid_container.add(int(pid))
            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            for img_path in img_paths:
                pid = osp.basename(img_path).split('_')[0]
                pid = int(pid)
                if relabel: pid = pid2label[pid]
                data.append((img_path, pid))
        else:
            for img_path in img_paths:
                data.append((img_path, -1))
        return data
