from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .angular_penalty_softmax_loss import AngularPenaltySMLoss
from .of_penalty_loss import OFPenalty
from .ranked_loss import RankedLoss
from .ranked_clu_loss import CRankedLoss
from .arcface_loss import ArcMarginProduct
from .multi_similarity_loss import MultiSimilarityLoss
from .focal_loss import FocalLossWithOHEM

def DeepSupervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
