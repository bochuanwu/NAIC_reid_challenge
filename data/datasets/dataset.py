from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import numpy as np
import tarfile
import zipfile
import copy

import torch

from utils import read_image, mkdir_if_missing, download_url


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid in other.train:
            pid += self.num_train_pids
            train.append((img_path, pid))

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train, self.query, self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            raise ValueError("")

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        for _, pid in data:
            pids.add(pid)
        return len(pids)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for _, pid in self.gallery:
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid in data:
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                return False
        return True

    def __repr__(self):
        num_train_pids = self.parse_data(self.train)
        num_query_pids = self.parse_data(self.query)
        num_gallery_pids = self.parse_data(self.gallery)

        msg = '  ---------------------------\n' \
              '  subset   | # ids | # items \n' \
              '  ---------------------------\n' \
              '  train    | {:5d} | {:7d}   \n' \
              '  query    | {:5d} | {:7d}   \n' \
              '  gallery  | {:5d} | {:7d}   \n' \
              '  ---------------------------\n' \
              '  items: images for image dataset\n'.format(
            num_train_pids, len(self.train),
            num_query_pids, len(self.query),
            num_gallery_pids, len(self.gallery)
        )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, img_path

    def show_summary(self):
        num_train_pids = self.parse_data(self.train)
        num_query_pids = self.parse_data(self.query)
        num_gallery_pids = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ---------------------------')
        print('  subset  | # ids | # images')
        print('  ---------------------------')
        print('  train   | {:5d} | {:8d}'.format(num_train_pids, len(self.train)))
        print('  query   | {:5d} | {:8d}'.format(num_query_pids, len(self.query)))
        print('  gallery | {:5d} | {:8d}'.format(num_gallery_pids, len(self.gallery)))
        print('  ---------------------------')
