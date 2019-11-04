# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import heapq

class Evaluator:
    def __init__(self, model, norm=False):
        self.model = model
        self.norm = norm

    def evaluate(self, queryloader, galleryloader, ranks=200):
        self.model.eval()
        qf, q_paths = [], []
        for i, inputs0 in enumerate(queryloader):
            inputs, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            qf.append(feature0)
            q_paths+=paths

        qf = torch.cat(qf, 0)
        if True == self.norm:
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_paths = [], []
        for i, inputs0 in enumerate(galleryloader):
            inputs, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
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

        q_g_dist = q_g_dist.cpu().numpy()

        print(q_g_dist.shape, len(q_paths), len(g_paths))
        q_paths = np.array(q_paths)
        g_paths = np.array(g_paths)

        #generate compare results
        clusters = {}
        for i in range(q_g_dist.shape[0]):
            temp_vector = np.squeeze(q_g_dist[i,])
            temp = temp_vector.tolist()

            y = np.array(list(map(temp.index, heapq.nsmallest(ranks, temp_vector))))

            clusters[q_paths[i]] = g_paths[y].tolist()

        print("results: ", clusters)

        return clusters

    def _parse_data(self, inputs):
        imgs, _, image_path = inputs
        return imgs.cuda(), image_path

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()