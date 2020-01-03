from __future__ import absolute_import
from __future__ import division

import math
from collections import defaultdict, Counter
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler, RandomSampler


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs.extend(np.random.choice(
                    idxs, size=self.num_instances - len(idxs), replace=True
                ))
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomBalancedIdentitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_instances=6):
        self.data_source = data_source
        self.num_instances = num_instances
        self.batch_size = batch_size
        self.index_dic = defaultdict(list)
        for index, (_, pid,) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            idxs = self.index_dic[pid]
            if len(idxs) < self.num_instances:
                extend_idxs = np.random.choice(
                    idxs, size=self.num_instances % len(idxs), replace=True
                )
                idxs = np.repeat(idxs, self.num_instances // len(idxs)).tolist()
                idxs.extend(extend_idxs)
            elif len(idxs) > self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=False
                )
            ret.extend(idxs)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances


class RandomIdentityOverSampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        num_samplers = len(self.data_source)

        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)

        self.pids = list(self.index_dic.keys())
        self.mean = round(1. * num_samplers / len(self.pids))

        # estimate number of examples in an epoch
        # TODO: improve precision
        over_sampler_index_dic = copy.deepcopy(self.index_dic)

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.mean:
                extend_idxs = np.random.choice(
                    idxs, size=self.mean % len(idxs), replace=True
                )
                idxs = np.repeat(idxs, self.mean // len(idxs)).tolist()
                idxs.extend(extend_idxs)
                num = self.mean
                over_sampler_index_dic[pid] = idxs
            self.length += num - num % self.num_instances
        self.index_dic = over_sampler_index_dic

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs.extend(np.random.choice(
                    idxs, size=self.num_instances - len(idxs), replace=True
                ))
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(data_source, train_sampler, batch_size=32, num_instances=4, **kwargs):
    """Builds a training sampler.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
    """
    if train_sampler == 'RandomIdentitySampler':
        # sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomBalancedIdentitySampler':
        sampler = RandomBalancedIdentitySampler(data_source, batch_size, num_instances)
    elif train_sampler == 'RandomIdentityOverSampler':
        sampler = RandomIdentityOverSampler(data_source, batch_size, num_instances)
    else:
        sampler = RandomSampler(data_source)

    return sampler
