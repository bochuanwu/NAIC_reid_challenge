#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2019/11/25 4:04 下午
# @author  : wuh-xmu
# @FileName: ensemble.py
# @Software: PyCharm
import json
from collections import defaultdict
import glob
import torch
import numpy as np
import os


def weightSum(paths, weights=None):
    results = [torch.load(path, map_location='cpu') for path in paths]

    anchor = results[0]
    anchor_query_path = anchor['query_path']
    anchor_query_path = list(map(os.path.basename, anchor_query_path))
    anchor_gallery_path = anchor['gallery_path']
    anchor_gallery_path = list(map(os.path.basename, anchor_gallery_path))
    anchor_dist_mat = anchor['dist_mat']

    anchor_query_path = np.asarray(anchor_query_path, np.str)
    anchor_gallery_path = np.asarray(anchor_gallery_path, np.str)

    query_idx = np.argsort(anchor_query_path)
    anchor_query_path = anchor_query_path[query_idx]

    gallery_idx = np.argsort(anchor_gallery_path)
    anchor_gallery_path = anchor_gallery_path[gallery_idx]

    anchor_dist_mat = anchor_dist_mat[query_idx, :]
    anchor_dist_mat = anchor_dist_mat[:, gallery_idx]

    dist_mats = []
    for result in results:
        query_path = result['query_path']
        gallery_path = result['gallery_path']
        dist_mat = result['dist_mat']
        query_path = np.asarray(query_path, np.str)
        gallery_path = np.asarray(gallery_path, np.str)
        query_idx = np.argsort(query_path)
        assert (query_path[query_idx] != anchor_query_path).sum() == 0
        gallery_idx = np.argsort(gallery_path)
        assert (gallery_path[gallery_idx] != anchor_gallery_path).sum() == 0
        dist_mat = dist_mat[query_idx, :]
        dist_mat = dist_mat[:, gallery_idx]
        dist_mats.append(dist_mat)

    res = np.zeros(anchor_dist_mat.shape)
    if weights is not None:
        for dist_mat, weight in zip(dist_mats, weights):
            res += weight * dist_mat
    else:
        for dist_mat in dist_mats:
            res += dist_mat
        res /= len(dist_mats)
        print('...')
    save_dict = {'dist_mat': res,
                 'query_path': anchor_query_path.tolist(),
                 'gallery_path': anchor_gallery_path.tolist()}

    torch.save(save_dict, './final_distmat.pth')
    _, indices = torch.topk(torch.from_numpy(res), k=200, dim=1, largest=False)
    sumbit = defaultdict(list)
    anchor_query_path = np.asarray(anchor_query_path, np.str)
    anchor_gallery_path = np.asarray(anchor_gallery_path, np.str)
    for i, qpath in enumerate(anchor_query_path):
        img_name = os.path.basename(qpath)
        sumbit[img_name] = anchor_gallery_path[indices[i].numpy()].tolist()
    with open(os.path.join('.', 'final_submit.json'), 'w', encoding='utf8') as f:
        json.dump(sumbit, f)


if __name__ == '__main__':
    model_results = [
        'model_results_B/duallattv2_ensembele_mgn_feat512_sepbn_mgnv3.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam_b.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam_feat512_128_b.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam_feat512.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam_feat512_b.pth',
        'model_results_B/mgn_ibn_bnneck_eraParam_feat512_sepbn_fc.pth',
        'model_results_B/mgnv2_512.pth',
        'model_results_B/resibn50_pa_ca_dropblock_tta_b.pth',
        'model_results_B/SPCBv2.pth',
        'model_results_B/StackPCBv2_ibn_bnneck.pth'

    ]
    weights = [0.15, 0.12, 0.05, 0.12, 0.05, 0.12, 0.05, 0.12, 0.05, 0.12, 0.05]


    weightSum(model_results, weights)
