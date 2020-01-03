# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
from metrics import compute_distance_matrix


def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def rank_loss(dist_mat, labels, margin, alpha, tval):
    """
    Args:
      dist_mat: pytorch Tensor, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_pos[torch.arange(N), torch.arange(N)] = 0
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    dist_ap = dist_mat[is_pos].contiguous().view(N, -1)
    dist_an = dist_mat[is_neg].contiguous().view(N, -1)

    ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=1e-6)
    ap_pos_num = ap_is_pos.size(1)
    ap_pos_val_sum = torch.sum(ap_is_pos, dim=1, keepdim=True)
    loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

    an_is_pos = torch.lt(dist_an, alpha).float()
    an_less_alpha = dist_an * an_is_pos
    alpha_ = alpha * an_is_pos
    tval_ = tval * an_is_pos
    an_weight = torch.exp(tval_ * (-1 * an_less_alpha + alpha_))
    an_weight_sum = torch.sum(an_weight, dim=1, keepdim=True) + 1e-5
    an_dist_lm = alpha_ - an_less_alpha
    an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight), dim=1, keepdim=True)
    loss_an = torch.div(an_ln_sum, an_weight_sum)
    loss = loss_ap + loss_an
    return loss.sum() * 1. / N


class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, margin=None, alpha=None, tval=None, metric='euclidean'):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        self.metric = metric

    def __call__(self, global_feat, labels, normalize_feature=True):

        if self.metric == 'euclidean':
            if normalize_feature:
                global_feat = normalize_rank(global_feat, axis=-1)
            dist_mat = euclidean_dist_rank(global_feat, global_feat)
        else:
            dist_mat = compute_distance_matrix(global_feat, global_feat, self.metric)
        total_loss = rank_loss(dist_mat, labels, self.margin, self.alpha, self.tval)

        return total_loss
