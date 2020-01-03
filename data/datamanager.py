from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

from data.sampler import build_train_sampler
from data.transforms import build_transforms
from data.datasets import init_image_dataset


class DataManager(object):
    r"""Base data manager.

    Args:
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        erase_mean=(list or None, optional): data std. Default is None (use 【0，0, 0]).
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, sources=None, targets=None, height=256, width=128, transforms='random_flip',
                 norm_mean=None, norm_std=None, erase_mean=None, is_train=True, use_gpu=False,
                 appended_gallery=False, appended_train=False):
        self.sources = sources
        self.targets = targets
        self.height = height
        self.width = width

        if self.sources is None:
            raise ValueError('sources must not be None')

        if isinstance(self.sources, str):
            self.sources = [self.sources]

        if self.targets is None:
            self.targets = self.sources

        if isinstance(self.targets, str):
            self.targets = [self.targets]

        self.transform_tr, self.transform_te = build_transforms(
            self.height, self.width, transforms=transforms,
            norm_mean=norm_mean, norm_std=norm_std, erase_mean=erase_mean
        )
        self.is_train = is_train
        self.use_gpu = (torch.cuda.is_available() and use_gpu)

    @property
    def num_train_pids(self):
        """Returns the number of training person identities."""
        return self._num_train_pids

    def return_dataloaders(self):
        """Returns trainloader and testloader."""
        return self.trainloader, self.testloader

    def return_testdataset_by_name(self, name):
        """Returns query and gallery of a test dataset, each containing
        tuples of (img_path(s), pid, camid).

        Args:
            name (str): dataset name.
        """
        return self.testdataset[name]['query'], self.testdataset[name]['gallery']


class ImageDataManager(DataManager):
    r"""Image data manager.

    Args:
        root (str): root path to datasets.
        sources (str or list): source dataset(s).
        targets (str or list, optional): target dataset(s). If not given,
            it equals to ``sources``.
        height (int, optional): target image height. Default is 256.
        width (int, optional): target image width. Default is 128.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): data mean. Default is None (use imagenet mean).
        norm_std (list or None, optional): data std. Default is None (use imagenet std).
        use_gpu (bool, optional): use gpu. Default is True.
        split_id (int, optional): split id (*0-based*). Default is 0.
        combineall (bool, optional): combine train, query and gallery in a dataset for
            training. Default is False.
        batch_size_train (int, optional): number of images in a training batch. Default is 32.
        batch_size_test (int, optional): number of images in a test batch. Default is 32.
        workers (int, optional): number of workers. Default is 4.
        num_instances (int, optional): number of instances per identity in a batch.
            Default is 4.
        is_train (bool, optional): train stage or inference stage
        train_sampler (str, optional): sampler. Default is empty (``RandomSampler``).
        cuhk03_labeled (bool, optional): use cuhk03 labeled images.
            Default is False (defaul is to use detected images).
        cuhk03_classic_split (bool, optional): use the classic split in cuhk03.
            Default is False.
        market1501_500k (bool, optional): add 500K distractors to the gallery
            set in market1501. Default is False.

    Examples::

        data = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            batch_size_train=32,
            batch_size_test=100
        )
    """
    data_type = 'image'

    def __init__(self, root='', sources=None, targets=None, height=256, width=128, transforms='random_flip',
                 norm_mean=None, norm_std=None, erase_mean=None, use_gpu=True, combineall=False, is_train=True,
                 appended_gallery=False, appended_train=False, batch_size_train=32, batch_size_test=32,
                 workers=4, num_instances=4, train_sampler=''):
        super(ImageDataManager, self).__init__(sources=sources, targets=targets, height=height, width=width,
                                               transforms=transforms, norm_mean=norm_mean, norm_std=norm_std,
                                               erase_mean=erase_mean, is_train=is_train,
                                               appended_gallery=appended_gallery,
                                               appended_train=appended_train, use_gpu=use_gpu)

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                mode='train',
                combineall=combineall,
                root=root,
                is_train=is_train,
                appended_train=appended_train,
                appended_gallery=appended_gallery,
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids

        train_sampler = build_train_sampler(
            trainset.train, train_sampler,
            batch_size=batch_size_train,
            num_instances=num_instances
        )

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            sampler=train_sampler,
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        print('=> Loading test (target) dataset')
        self.testloader = {name: {'query': None, 'gallery': None} for name in self.targets}
        self.testdataset = {name: {'query': None, 'gallery': None} for name in self.targets}

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                is_train=is_train,
                appended_train=appended_train,
                appended_gallery=appended_gallery,
            )
            self.testloader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                is_train=is_train,
                appended_train=appended_train,
                appended_gallery=appended_gallery,
            )
            self.testloader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.testdataset[name]['query'] = queryset.query
            self.testdataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  train            : {}'.format(self.sources))
        print('  # train datasets : {}'.format(len(self.sources)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(trainset)))
        print('  test             : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
