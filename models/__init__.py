from __future__ import absolute_import

from .osnet_v2 import *
from .resnet import *
from .resnet_ibn_a import *
from .resnext_ibn_a import *

from .seresnet_ibn_a import *
from .resnet_gn import *
from .m_layers_net import mlayer_seresnet101
from .dualatt_seresnet_v2 import duallatt_seresnet101v2
from .resnet_v2_sn import resnetv2sn101
from .ensemble_dualatt_seresnet_v2 import ensemble_duallatt_seresnet
from .mgn import *
from .stack_pcb import *
from .stack_pcbv2 import *
from .mgnv2 import *
from .mgnv3 import *

__model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'resnet50_bnneck': resnet50_bnneck,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet152_ibn_a': resnet152_ibn_a,
    'resnetv2sn101': resnetv2sn101,
    # lightweight models

    # reid-specific models
    'osnet_x1_0': osnet_x1_0,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ibn_x4_0': osnet_ibn_x4_0,

    'seresnet101_ibn_a': seresnet101_ibn_a,
    'resnext101_ibn_a_32x4d': resnext101_ibn_a_32x4d,
    'resnet101_ibgn': resnet101_ibgn,
    'mlayer_seresnet101': mlayer_seresnet101,
    'duallatt_seresnet101v2': duallatt_seresnet101v2,

    'spcb_resnet50_512': spcb_resnet50_512,
    'spcbv2_resnet50_256': spcbv2_resnet50_256,
    'spcbv2_resnet50_512': spcbv2_resnet50_512,
    'spcbv2_resnet50_2048': spcbv2_resnet50_2048,
    'mgn_resnet50_256': mgn_resnet50_256,
    'mgn_resnet50_512': mgn_resnet50_512,
    'mgnv2_resnet50_256': mgnv2_resnet50_256,
    'mgnv2_resnet50_512': mgnv2_resnet50_512,
    'mgnv2_resnet50_1024': mgnv2_resnet50_1024,
    'mgnv2_resnet101_256': mgnv2_resnet101_256,
    'mgnv2_resnet101_512': mgnv2_resnet101_512,
    'mgnv2_resnet101_1024': mgnv2_resnet101_1024,
    'mgnv3_resnet50_256': mgnv3_resnet50_256,
    'mgnv3_resnet50_512': mgnv3_resnet50_512,
    'mgnv3_resnet101_256': mgnv3_resnet101_256,
    'mgnv3_resnet101_512': mgnv3_resnet101_512,
    # ensemble
    'ensemble_duallatt_seresnet': ensemble_duallatt_seresnet
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss='softmax', pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError('Unknown model: {}. Must be one of {}'.format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )
