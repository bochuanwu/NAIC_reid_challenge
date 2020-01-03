from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import time
import datetime
import numpy as np
import cv2

import metrics
import torch
import torch.nn as nn
from torch.nn import functional as F, DataParallel
from torch.utils.tensorboard import SummaryWriter
from utils import AverageMeter, visualize_ranked_results, save_checkpoint, re_ranking, mkdir_if_missing, rot90
from losses import DeepSupervision
from utils import crop_lb, crop_lt, crop_rt, crop_rb, center_crop, resize

try:
    from apex import amp
except:
    pass
GRID_SPACING = 10


class TrainEngine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, model, optimizer=None, scheduler=None, use_gpu=True, apex=False):
        self.datamanager = datamanager
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = None
        self.apex = apex
        if self.apex:
            if isinstance(self.model, nn.DataParallel):
                self.model = self.model.module
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
                self.model = DataParallel(self.model)
            else:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0, open_layers=None,
            start_eval=0, eval_freq=-1, test_only=False, print_freq=10,
            dist_metric='euclidean', normalize_feature=False, visrank=False, visrank_topk=10,
            ranks=[1, 5, 10, 20], rerank=False, visactmap=False, **kwargs):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
            max_epoch (int): maximum epoch.
            start_epoch (int, optional): starting epoch. Default is 0.
            fixbase_epoch (int, optional): number of epochs to train ``open_layers`` (new layers)
                while keeping base layers frozen. Default is 0. ``fixbase_epoch`` is counted
                in ``max_epoch``.
            open_layers (str or list, optional): layers (attribute names) open for training.
            start_eval (int, optional): from which epoch to start evaluation. Default is 0.
            eval_freq (int, optional): evaluation frequency. Default is -1 (meaning evaluation
                is only performed at the end of training).
            test_only (bool, optional): if True, only runs evaluation on test datasets.
                Default is False.
            print_freq (int, optional): print_frequency. Default is 10.
            dist_metric (str, optional): distance metric used to compute distance matrix
                between query and gallery. Default is "euclidean".
            normalize_feature (bool, optional): performs L2 normalization on feature vectors before
                computing feature distance. Default is False.
            visrank (bool, optional): visualizes ranked results. Default is False. It is recommended to
                enable ``visrank`` when ``test_only`` is True. The ranked images will be saved to
                "save_dir/visrank_dataset", e.g. "save_dir/visrank_market1501".
            visrank_topk (int, optional): top-k ranked images to be visualized. Default is 10.
            ranks (list, optional): cmc ranks to be computed. Default is [1, 5, 10, 20].
            rerank (bool, optional): uses person re-ranking (by Zhong et al. CVPR'17).
                Default is False. This is only enabled when test_only=True.
            visactmap (bool, optional): visualizes activation maps. Default is False.
        """
        trainloader, testloader = self.datamanager.return_dataloaders()

        if visrank and not test_only:
            raise ValueError('visrank=True is valid only if test_only=True')

        if test_only:
            self.test(
                0,
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                ranks=ranks,
                rerank=rerank,
                **kwargs
            )
            return

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        if visactmap:
            self.visactmap(testloader, save_dir, self.datamanager.width, self.datamanager.height, print_freq)
            return

        time_start = time.time()
        print('=> Start training')
        best_rank1 = 0.
        best_mAP = 0.
        best_score = 0.
        for epoch in range(start_epoch, max_epoch):
            torch.cuda.empty_cache()
            self.train(epoch, max_epoch, trainloader, fixbase_epoch, open_layers, print_freq, **kwargs)

            if (epoch + 1) >= start_eval and eval_freq > 0 and (epoch + 1) % eval_freq == 0 or (
                    epoch + 1) == max_epoch:
                rank1, mAP = self.test(
                    epoch,
                    testloader,
                    dist_metric=dist_metric,
                    normalize_feature=normalize_feature,
                    visrank=visrank,
                    visrank_topk=visrank_topk,
                    save_dir=save_dir,
                    ranks=ranks,
                    rerank=rerank,
                    **kwargs
                )
                score = 0.5 * rank1 + 0.5 * mAP
                if best_mAP < mAP:
                    best_mAP = mAP
                    self._save_checkpoint(epoch, rank1, mAP, score, save_dir, mode='mAP')
                if best_rank1 < rank1:
                    best_rank1 = rank1
                    self._save_checkpoint(epoch, rank1, mAP, score, save_dir, mode='rank1')
                if best_score < score:
                    best_score = score
                    self._save_checkpoint(epoch, rank1, mAP, score, save_dir, mode='score')
            # if (epoch + 1) >= 70 and (epoch + 1) % 5 == 0:
            #     state = {
            #         'state_dict': self.model.state_dict(),
            #         'epoch': epoch + 1,
            #         'optimizer': self.optimizer.state_dict(),
            #     }
            #
            #     fpath = osp.join(save_dir, 'model-{}.pth.tar'.format(epoch + 1))
            #     torch.save(state, fpath)
            #     print('Checkpoint saved to "{}"'.format(fpath))

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is None:
            self.writer.close()

    def train(self):
        r"""Performs training on source datasets for one epoch.

        This will be called every epoch in ``run()``, e.g.

        .. code-block:: python

            for epoch in range(start_epoch, max_epoch):
                self.train(some_arguments)

        .. note::

            This must be implemented in subclasses.
        """
        raise NotImplementedError

    def test(self, epoch, testloader, dist_metric='euclidean', normalize_feature=False,
             visrank=False, visrank_topk=10, save_dir='', ranks=[1, 5, 10, 20], rerank=False, **kwargs):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``_extract_features()`` and ``_parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        targets = list(testloader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            queryloader = testloader[name]['query']
            galleryloader = testloader[name]['gallery']

            rank1, mAP = self._evaluate(
                epoch,
                dataset_name=name,
                queryloader=queryloader,
                galleryloader=galleryloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                ranks=ranks,
                rerank=rerank,
                **kwargs
            )

        return rank1, mAP

    @torch.no_grad()
    def _evaluate(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=10, save_dir='', ranks=[1, 5, 10, 20], rerank=False, **kwargs):
        torch.cuda.empty_cache()
        batch_time = AverageMeter()

        print('Extracting features from query set ...')
        qf, q_pids = [], []  # query features, query person IDs
        for batch_idx, data in enumerate(queryloader):
            imgs, pids = self._parse_data_for_eval(data)
            imgs_hflip = self._hflip_image(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                imgs_hflip = imgs_hflip.cuda()
            end = time.time()

            # tta
            features0 = self._extract_features(imgs)
            features1 = self._extract_features(imgs_hflip)
            features = (features0 + features1) / 2.

            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids = [], []  # gallery features, gallery person IDs and gallery camera IDs
        end = time.time()
        for batch_idx, data in enumerate(galleryloader):
            imgs, pids = self._parse_data_for_eval(data)
            imgs_hflip = self._hflip_image(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                imgs_hflip = imgs_hflip.cuda()
            end = time.time()
            # tta
            features0 = self._extract_features(imgs)
            features1 = self._extract_features(imgs_hflip)
            features = (features0 + features1) / 2.


            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking k1: {} k2: {} lambda_value: {}...'
                  .format(kwargs['k1'], kwargs['k2'], kwargs['lambda_value']))
            distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric).numpy()
            distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric).numpy()

            distmat = re_ranking(distmat, distmat_qq, distmat_gg,
                                 k1=kwargs['k1'], k2=kwargs['k2'], lambda_value=kwargs['lambda_value'])

        print('Computing CMC and mAP ...')
        cmc, mAP = metrics.evaluate_rank(distmat, q_pids, g_pids)

        print('** Results **')
        print('mAP: {:.4%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.4%}'.format(r, cmc[r - 1]))
        print('offline score: {:.4%}'.format(0.5 * mAP + 0.5 * cmc[0]))
        if visrank:
            visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                self.datamanager.data_type,
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

        return cmc[0], mAP

    @torch.no_grad()
    def visactmap(self, testloader, save_dir, width, height, print_freq):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        self.model.eval()

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'actmap_' + target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[2]
                if self.use_gpu:
                    imgs = imgs.cuda()

                # forward to get convolutional feature maps
                try:
                    outputs = self.model(imgs, return_featuremaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_featuremaps". ' \
                                    'Please add return_featuremaps as an input argument to forward(). When ' \
                                    'return_featuremaps=True, return feature maps only.')

                if outputs.dim() != 4:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs.dim()))

                # compute activation maps
                outputs = (outputs ** 2).sum(1)
                b, h, w = outputs.size()
                outputs = outputs.view(b, h * w)
                outputs = F.normalize(outputs, p=2, dim=1)
                outputs = outputs.view(b, h, w)

                if self.use_gpu:
                    imgs, outputs = imgs.cpu(), outputs.cpu()

                for j in range(outputs.size(0)):
                    # get image name
                    path = paths[j]
                    imname = osp.basename(osp.splitext(path)[0])

                    # RGB image
                    img = imgs[j, ...]
                    for t, m, s in zip(img, imagenet_mean, imagenet_std):
                        t.mul_(s).add_(m).clamp_(0, 1)
                    img_np = np.uint8(np.floor(img.numpy() * 255))
                    img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                    # activation map
                    am = outputs[j, ...].numpy()
                    am = cv2.resize(am, (width, height))
                    am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                    # overlapped
                    overlapped = img_np * 0.3 + am * 0.7
                    overlapped[overlapped > 255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image
                    grid_img = 255 * np.ones((height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:, width + GRID_SPACING: 2 * width + GRID_SPACING, :] = am
                    grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
                    cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

                if (batch_idx + 1) % print_freq == 0:
                    print('- done batch {}/{}'.format(batch_idx + 1, len(queryloader)))

    def _compute_loss(self, criterion, outputs, targets, reduce='mean'):
        if isinstance(outputs, (tuple, list)):
            loss = DeepSupervision(criterion, outputs, targets, reduce)
        else:
            loss = criterion(outputs, targets)
        return loss

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _hflip_image(self, input):
        '''flip horizontal'''
        inv_idx = torch.arange(input.size(3) - 1, -1, -1, dtype=torch.int64)  # N x C x H x W
        if input.is_cuda:
            inv_idx = inv_idx.cuda()
        image_hflip = input.index_select(3, inv_idx)
        return image_hflip

    def _parse_data_for_train(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        return imgs, pids

    def _save_checkpoint(self, epoch, rank1, mAP, score, save_dir, mode='rank1'):
        save_checkpoint({
            'state_dict': self.model.state_dict(),
            'epoch': epoch + 1,
            'rank1': rank1,
            'mAP': mAP,
            'score': score,
            'optimizer': self.optimizer.state_dict(),
        }, save_dir, mode=mode)
