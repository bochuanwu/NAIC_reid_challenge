#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/26 12:30 下午
# @Author  : wuh-xmu
# @FileName: kesicreid.py
# @Software: PyCharm


from __future__ import absolute_import
from __future__ import print_function

import shutil

from . import ImageDataset
import os.path as osp
import os
import glob
import json
import pandas as pd

data_paths = {
    'train': 'train/second_stage_train',
    'val_gallery': 'train/second_stage_train',
    'val_query': 'train/second_stage_train',
    'appended_train': 'train/appended_train',
    'test_query': 'test/query_a',
    'test_gallery': 'test/gallery_a',
    'appended_gallery': 'test/gallery_a'
}

label_paths = {
    'train': 'train/train.txt',
    'val_query': 'train/query.txt',
    'val_gallery': 'train/gallery.txt',
    'appended_gallery': 'train/appended_gallery.txt',
    'appended_train': 'train/appended_train.txt'
}


class KesciReID(ImageDataset):

    def __init__(self, root, is_train=True, appended_gallery=False, appended_train=False, **kwargs):

        self.root = osp.abspath(osp.expanduser(root))
        self.train_dir = osp.join(self.root, data_paths['train'])
        self.train_label = osp.join(self.root, label_paths['train'])
        self.pid2label = {}

        if is_train:
            self.query_dir = osp.join(self.root, data_paths['val_query'])
            self.gallery_dir = osp.join(self.root, data_paths['val_gallery'])
            self.query_label = osp.join(self.root, label_paths['val_query'])
            self.gallery_label = osp.join(self.root, label_paths['val_gallery'])

            train = self.process_dir(self.train_dir, self.train_label, relabel=True)
            query = self.process_dir(self.query_dir, self.query_label, relabel=False)
            gallery = self.process_dir(self.gallery_dir, self.gallery_label, relabel=False)

            if appended_gallery:
                self.appended_gallery_dir = osp.join(self.root, data_paths['appended_gallery'])
                self.appended_gallery_label = osp.join(self.root, label_paths['appended_gallery'])
                gallery += self.process_dir(self.appended_gallery_dir, self.appended_gallery_label, relabel=False)

            if appended_train:

                self.appended_train_dir = osp.join(self.root, data_paths['appended_train'])
                self.appended_train_label = osp.join(self.root, label_paths['appended_train'])

                if not osp.exists(self.appended_train_dir) or len(os.listdir(self.appended_train_dir)) == 0:
                    os.makedirs(self.appended_train_dir, exist_ok=True)

                    test_query = os.listdir(osp.join(self.root, data_paths['test_query']))
                    test_gallery = os.listdir(osp.join(self.root, data_paths['test_gallery']))

                    with open(self.appended_train_label, 'r') as f:
                        for line in f.readlines():
                            line = line.strip().replace('\n', '').replace('\r', '')
                            img_name, pid = line.split(' ')

                            if img_name in test_gallery:
                                src = osp.join(self.root, data_paths['test_gallery'], img_name)
                            else:
                                src = osp.join(self.root, data_paths['test_query'], img_name)
                            shutil.copy(src, osp.join(self.appended_train_dir, img_name))

                train += self.process_dir(self.appended_train_dir, self.appended_train_label, relabel=True)

        else:
            self.query_dir = osp.join(self.root, data_paths['test_query'])
            self.gallery_dir = osp.join(self.root, data_paths['test_gallery'])
            self.query_label = None
            self.gallery_label = None

            train = self.process_dir(self.train_dir, self.train_label, relabel=True)
            query = self.process_dir(self.query_dir, self.query_label, relabel=False)
            gallery = self.process_dir(self.gallery_dir, self.gallery_label, relabel=False)

        super(KesciReID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label_path=None, relabel=False):
        img_paths = []
        data = []
        pids = []
        if label_path is not None:
            pid_container = set()
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().replace('\n', '').replace('\r', '')
                    img_name, pid = line.split(' ')
                    # appended first stage train data to val set
                    img_name = osp.join(dir_path, osp.basename(img_name))
                    pid = int(pid)
                    pids.append(pid)
                    img_paths.append(img_name)
                    pid_container.add(pid)

            if relabel:
                append = {pid: len(self.pid2label) + label for label, pid in enumerate(pid_container)}
                self.pid2label.update(append)
                pids = [self.pid2label[pid] for pid in pids]

        else:
            img_paths = glob.glob(dir_path + '/*.png')
            pids = [-1 for i in range(len(img_paths))]
        for img_path, pid in zip(img_paths, pids):
            data.append((img_path, pid))
        return data
