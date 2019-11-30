from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

from metrics import distance


def l2_norm(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
    Returns:
        x (torch.Tensor): same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def hard_example_mining(dist_mat, labels):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat (torch.Tensor): pair wise distance between samples, shape [N, N]
        labels: (torch.LongTensor): with shape [N]
    Returns:
        dist_ap (torch.Tensor): distance(anchor, positive); shape [N]
        dist_an (torch.Tensor): distance(anchor, negative); shape [N]
    NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
    """

    torch.set_printoptions(threshold=5000)
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
        metric (str, optional): method for pair distance. Default is 'euclidean'.
    """

    def __init__(self, margin=0.3, metric='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        if margin > 0.:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize=False):
        """
         Args:
             inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
             targets (torch.LongTensor): ground truth labels with shape (num_classes).
             normalize (bool, optional): Normalizing to unit length along the specified dimension.
        """
        if normalize:
            inputs = l2_norm(inputs, axis=-1)
        # Compute pairwise distance, replace by the official when merged
        if self.metric == 'euclidean':
            dist_mat = distance.euclidean_distance(inputs, inputs)
        elif self.metric == 'cosine':
            dist_mat = distance.cosine_distance(inputs, inputs)
        else:
            raise ValueError(
                'Unknown distance metric: {}. '
                'Please choose either "euclidean" or "cosine"'.format(self.metric)
            )
        dist_ap, dist_an = hard_example_mining(dist_mat, targets)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin > 0.:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss
