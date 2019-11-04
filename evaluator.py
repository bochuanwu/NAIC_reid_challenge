# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import heapq
from reranking import re_ranking as re_ranking_func

class Evaluator:
    def __init__(self, model, norm=False):
        self.model = model
        self.norm = norm

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
                 eval_flip = False, ranks=200, re_ranking=False):
        self.model.eval()
        qf, q_paths = [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, _ = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)
            q_paths+=paths

        qf = torch.cat(qf, 0)
        if True == self.norm:
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_paths = [], []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, _ = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
            g_paths +=paths

        gf = torch.cat(gf, 0)
        if True == self.norm:
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        # q_g_dist = q_g_dist.cpu().numpy()

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist)).cpu().numpy()
        else:
            distmat = q_g_dist.cpu().numpy()

        print(distmat.shape, len(q_paths), len(g_paths))
        q_paths = np.array(q_paths)
        g_paths = np.array(g_paths)

        #generate compare results
        clusters = {}
        for i in range(distmat.shape[0]):
            print(i)
            temp_vector = np.squeeze(distmat[i,])
            temp = temp_vector.tolist()

            y = list(map(temp.index, heapq.nsmallest(ranks, temp)))

            if len(y) != len(set(y)):
                m=[]
                for j in range(ranks):
                    m.append(temp.index(min(temp)))
                    temp[temp.index(min(temp))] = 1000000
                print(m)
                clusters[q_paths[i]] = g_paths[np.array(m)].tolist()
                continue

            clusters[q_paths[i]] = g_paths[np.array(y)].tolist()

        print("results: ", clusters)

        return clusters

    def _parse_data(self, inputs):
        imgs, _, image_path = inputs
        return imgs.cuda(), image_path

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()