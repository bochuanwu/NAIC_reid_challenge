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

    dist_mats = [anchor_dist_mat]
    for result in results[1:]:
        res = np.zeros(shape=anchor_dist_mat.shape)
        query_path = result['query_path']
        gallery_path = result['gallery_path']
        dist_mat = result['dist_mat']

        for ii, qpath in enumerate(query_path):
            qidx = anchor_query_path.index(os.path.basename(qpath))
            for jj, gpath in enumerate(gallery_path):
                gidx = anchor_gallery_path.index((os.path.basename(gpath)))
                res[qidx, gidx] = dist_mat[ii, jj]

        dist_mats.append(res)

    res = np.zeros(anchor_dist_mat.shape)
    if weights is not None:
        for dist_mat, weight in zip(dist_mats, weights):
            res += weight * dist_mat
    else:
        for dist_mat in dist_mats:
            res += dist_mat
        res /= len(dist_mats)
        print('...')
    torch.save(res, './final_distmat.pth')
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
    # model_results = glob.glob('model_results/*')
    model_results = ['model_results/mgn_ibn_bnneck_eraParam_8614.pth',
                     'model_results/mgn_ibn_bnneck_eraParam_feat512_sepbn_8688.pth',
                     # 'model_results/mgn_ibn_bnneck_eraParam_feat512_8634.pth',
                     'model_results/mgn_ibn_bnneck_eraParam_feat512.pth',
                     'model_results/spcbv2_8675.pth',
                     'model_results/dual_ser101_ensemble_8638.pth',
                     'model_results/StackPCBv2_ibn_bnneck_862.pth']

    print(model_results)
    weightSum(model_results)
