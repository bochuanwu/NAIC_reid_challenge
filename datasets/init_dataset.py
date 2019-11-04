# encoding: utf-8
"""
@author:  zhoumi
@contact: zhoumi281571814@126.com
"""

import glob
import os.path as osp
import numpy as np

class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]
        pids = set(pids)
        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  ----------------------------------------")


class Tx_dataset(BaseImageDataset):
    dataset_dir = 'tx_challenge'
    def __init__(self, root_path = '/data/zhoumi/datasets/reid',
                 set = 'train_set', file_list = 'train_list.txt', verbose=True):
        self.dataset_dir = osp.join(root_path, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, set)
        self.train_list = osp.join(self.dataset_dir, file_list)

        self._check_before_run()

        self.dataset = self._process_dir(self.train_dir, self.train_list)

        if verbose:
            print("=> tx_chanlleng loaded")
            self.print_dataset_statistics(self.dataset)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))


    def _process_dir(self, dir_path, file_list):
        dataset = []
        with open(file_list) as fd:
            lines = fd.readlines()

            for line in lines:
                image_name = line.split(' ')[0].split('/')[-1]
                pid = eval(line.split(' ')[-1].replace('\n', ''))

                img_path = osp.join(dir_path, image_name)

                dataset.append((img_path, pid))

        return dataset





