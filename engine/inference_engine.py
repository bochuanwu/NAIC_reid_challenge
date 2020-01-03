#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 1:40 下午
# @Author  : wuh-xmu
# @FileName: inference_engine.py
# @Software: PyCharm
import time
import os.path as osp
import datetime
import numpy as np
import cv2
import metrics
import torch
import shutil
import json
from torch import nn
from collections import defaultdict
from utils import AverageMeter, visualize_ranked_results, save_checkpoint, re_ranking, mkdir_if_missing
from torch.nn import functional as F
from utils import crop_lb, crop_lt, crop_rt, crop_rb, center_crop, resize

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


class InfenerceEngine(object):
    r"""A generic base Engine class for both image- and video-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        use_gpu (bool, optional): use gpu. Default is True.
    """

    def __init__(self, datamanager, model, use_gpu=True):
        self.datamanager = datamanager
        self.model = model
        self.use_gpu = (torch.cuda.is_available() and use_gpu)

        # check attributes
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', dist_metric='euclidean',
            normalize_feature=False, visrank=False, visrank_topk=10,
            ranks=[1, 5, 10, 20], rerank=False, visactmap=False, **kwargs):
        r"""A unified pipeline for training and evaluating a model.

        Args:
            save_dir (str): directory to save model.
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
        _, testloader = self.datamanager.return_dataloaders()

        if visactmap:
            self.visactmap(testloader, save_dir, self.datamanager.width, self.datamanager.height)
            return

        time_start = time.time()
        print('=> Start inferencing')
        targets = list(testloader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            self.inference(
                testloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                ranks=ranks,
                rerank=rerank,
                dataset_name=name,
                **kwargs
            )

            elapsed = round(time.time() - time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print('Elapsed {}'.format(elapsed))

    @torch.no_grad()
    def inference(self, testloader, dataset_name='', dist_metric='euclidean', normalize_feature=False,
                  visrank=False, visrank_topk=20, save_dir='', rerank=False, **kwargs):

        targets = list(testloader.keys())

        for name in targets:
            domain = 'source' if name in self.datamanager.sources else 'target'
            print('##### Inferencing {} ({}) #####'.format(name, domain))
            queryloader = testloader[name]['query']
            galleryloader = testloader[name]['gallery']

        batch_time = AverageMeter()
        res = {}
        print('Extracting features from query set ...')
        torch.cuda.is_available()
        qf, qpaths = [], []  # query features, query person IDs
        for batch_idx, data in enumerate(queryloader):
            imgs, paths = data[0], data[2]
            paths = list(map(lambda x: osp.basename(x), paths))
            imgs_hflip = self._hflip_image(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                imgs_hflip = imgs_hflip.cuda()
            end = time.time()
            # tta
            features0 = self._extract_features(imgs)
            features1 = self._extract_features(imgs_hflip)
            features = (features0 + features1) / 2

            batch_time.update(time.time() - end)
            features = features.data.cpu()
            qf.append(features)
            qpaths.extend(paths)
        qf = torch.cat(qf, 0)
        res['query_path'] = qpaths  # list
        qpaths = np.asarray(qpaths, dtype=np.str)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        torch.cuda.is_available()
        print('Extracting features from gallery set ...')
        gf, gpaths = [], []  # gallery features, gallery person IDs and gallery camera IDs
        end = time.time()
        for batch_idx, data in enumerate(galleryloader):
            imgs, paths = data[0], data[2]
            paths = list(map(lambda x: osp.basename(x), paths))
            imgs_hflip = self._hflip_image(imgs)
            if self.use_gpu:
                imgs = imgs.cuda()
                imgs_hflip = imgs_hflip.cuda()
            end = time.time()
            # tta
            features0 = self._extract_features(imgs)
            features1 = self._extract_features(imgs_hflip)
            features = (features0 + features1) / 2

            batch_time.update(time.time() - end)
            features = features.data.cpu()
            gf.append(features)
            gpaths.extend(paths)
        gf = torch.cat(gf, 0)
        res['gallery_path'] = gpaths  # list
        gpaths = np.asarray(gpaths, dtype=np.str)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        print('Speed: {:.4f} sec/batch'.format(batch_time.avg))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        res['query_feat'] = qf
        res['gallery_feat'] = gf

        torch.save(res, osp.join(save_dir, 'features.pth'))
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

        res['dist_mat'] = distmat
        # to save
        torch.save(res, osp.join(save_dir, 'result.pth'))
        _, indices = torch.topk(torch.from_numpy(distmat), k=200, dim=1, largest=False)
        res = defaultdict(list)
        for i, qpath in enumerate(qpaths):
            img_name = osp.basename(qpath)
            res[img_name] = gpaths[indices[i].numpy()].tolist()
        with open(osp.join(save_dir, 'submit.json'), 'w', encoding='utf8') as f:
            json.dump(res, f)

        if visrank:
            self.visualize_ranked_results(
                distmat,
                self.datamanager.return_testdataset_by_name(dataset_name),
                width=self.datamanager.width,
                height=self.datamanager.height,
                save_dir=osp.join(save_dir, 'visrank_' + dataset_name),
                topk=visrank_topk
            )

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

    def visualize_ranked_results(self, distmat, dataset, width=128, height=256, save_dir='', topk=10):
        """Visualizes ranked results.

        Supports both image-reid and video-reid.

        For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
        saved in folders each containing a tracklet.

        Args:
            distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
            dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
                tuples of (img_path(s), pid, camid).
            width (int, optional): resized image width. Default is 128.
            height (int, optional): resized image height. Default is 256.
            save_dir (str): directory to save output images.
            topk (int, optional): denoting top-k images in the rank list to be visualized.
                Default is 10.
        """
        num_q, num_g = distmat.shape
        mkdir_if_missing(save_dir)

        print('# query: {}\n# gallery {}'.format(num_q, num_g))
        print('Visualizing top-{} ranks ...'.format(topk))

        query, gallery = dataset
        assert num_q == len(query)
        assert num_g == len(gallery)

        indices = np.argsort(distmat, axis=1)

        def _cp_img_to(src, dst, rank, prefix, matched=False):
            """
            Args:
                src: image path or tuple (for vidreid)
                dst: target directory
                rank: int, denoting ranked position, starting from 1
                prefix: string
                matched: bool
            """
            if isinstance(src, (tuple, list)):
                if prefix == 'gallery':
                    suffix = 'TRUE' if matched else 'FALSE'
                    dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
                else:
                    dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
                mkdir_if_missing(dst)
                for img_path in src:
                    shutil.copy(img_path, dst)
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
                shutil.copy(src, dst)

        for q_idx in range(num_q):
            qimg_path, _ = query[q_idx]
            qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            # qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            # qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones((height, num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3),
                                     dtype=np.uint8)
            grid_img[:, :width, :] = qimg

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                gimg_path, _ = gallery[g_idx]
                # invalid = (qpid == gpid) & (qcamid == gcamid)
                invalid = 0
                if not invalid:
                    # matched = gpid == qpid
                    # border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    # gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    # gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx + 1) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start: end, :] = gimg

                    rank_idx += 1
                    if rank_idx > topk:
                        break

            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

            if (q_idx + 1) % 100 == 0:
                print('- done {}/{}'.format(q_idx + 1, num_q))

        print('Done. Images have been saved to "{}" ...'.format(save_dir))
