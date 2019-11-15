# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import heapq
from reranking import re_ranking as re_ranking_func

class Evaluator:
    def __init__(self, model, pcb_model=None, norm=False, eval_flip=False, re_ranking=False, concate=False):
        self.model = model
        self.norm = norm
        self.eval_flip = eval_flip
        self.re_ranking = re_ranking
        self.pcb_model = pcb_model
        self.concate = concate

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader, ranks=200, k1=20, k2=6, lambda_value=0.3):
        self.model.eval()
        if self.concate:
            self.pcb_model.eval()
        qf, q_paths = [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, _, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if self.eval_flip:
                inputs, _, _ = self._parse_data(inputs1)
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
            inputs, _, paths = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if self.eval_flip:
                inputs, _, _ = self._parse_data(inputs1)
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

        if self.re_ranking:
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

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value)).cpu().numpy()
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

        # print("results: ", clusters)

        return clusters

    def validation(self, queryloader, galleryloader, queryFliploader, galleryFliploader,re_ranking=False,
                   ranks=[1], k1=20, k2=6, lambda_value=0.3):
        self.model.eval()
        if self.concate:
            self.pcb_model.eval()
        qf, q_pids = [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, _ = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if self.eval_flip:
                inputs, _, _ = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)
            q_pids.extend(list(map(int, pids)))

        qf = torch.cat(qf, 0)
        if True == self.norm:
            qf = torch.nn.functional.normalize(qf, dim=1, p=2)
        q_pids = torch.Tensor(q_pids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids = [], []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, _ = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if self.eval_flip:
                inputs, _, _ = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
            g_pids.extend(list(map(int, pids)))

        gf = torch.cat(gf, 0)
        if True == self.norm:
            gf = torch.nn.functional.normalize(gf, dim=1, p=2)
        g_pids = torch.Tensor(g_pids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")
        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

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

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist, k1=k1, k2=k2, lambda_value=lambda_value))
        else:
            distmat = q_g_dist

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
            print("tencent score: {}".format((cmc[r-1] + mAP) / 2))
        print("------------------")
        return (cmc[0] + mAP) / 2


    def _parse_data(self, inputs):
        imgs, pids, image_path = inputs
        return imgs.cuda(), pids, image_path

    def _forward(self, inputs):
        if self.concate:
            with torch.no_grad():
                feature0 = self.model(inputs)
                feature1 = self.pcb_model(inputs)
                # print(feature0.size(), feature1.size())
                feature = torch.cat((feature0, feature1), 1)
        else:
            with torch.no_grad():
                feature = self.model(inputs)
        return feature.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, max_rank=200):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = (g_pids[indices] == q_pids.view([num_q, -1]))

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][:]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()