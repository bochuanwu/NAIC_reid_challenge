from __future__ import absolute_import


from .kaiming_init import *
from .resnet_ibn_a import *
from .resnext_ibn_a import *
from .dilated_resnet_ibn_a import *
from .seresnet_ibn_a import *

from .dualatt_seresnet_v2 import duallatt_seresnet101v2
from .stack_pcbv2 import *
from .mgnv2 import *
from .mgnv2_r101 import *
from .mgnv4 import *
from .mgnv4_r101 import *
from .mgnv4_r101v2 import *
from .mgnv4_r152 import *
from .mgnv5 import *
from .mgnv6 import *
from .dilated_mgn import *
from .strong_baseline import *


__model_factory = {
    # image classification models
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet152_ibn_a': resnet152_ibn_a,
    'dilated_resnet50_ibn_a': dilated_resnet50_ibn_a,
    'dilated_resnet101_ibn_a': dilated_resnet101_ibn_a,
    # lightweight models

    # reid-specific models

    'sb_ibn50': sb_ibn50,

    'seresnet101_ibn_a': seresnet101_ibn_a,
    'resnext101_ibn_a_32x4d': resnext101_ibn_a_32x4d,
    'duallatt_seresnet101v2': duallatt_seresnet101v2,

    'spcbv2_resnet50_256': spcbv2_resnet50_256,
    'spcbv2_resnet50_512': spcbv2_resnet50_512,
    'spcbv2_resnet50_2048': spcbv2_resnet50_2048,
    'mgnv2_resnet50_256': mgnv2_resnet50_256,
    'mgnv2_resnet50_512': mgnv2_resnet50_512,
    'mgnv2_resnet50_1024': mgnv2_resnet50_1024,
    'mgnv2_woibn_resnet50_512': mgnv2_woibn_resnet50_512,
    'mgnv2_resnet101_512': mgnv2_resnet101_512,
    'mgnv4_resnet50_512': mgnv4_resnet50_512,
    'mgnv4_resnet101_512': mgnv4_resnet101_512,
    'mgnv4_seresnet101_512':mgnv4_seresnet101_512,
    'mgnv4_resnext101_512': mgnv4_resnext101_512,
    'mgnv4_resnet101v2_512': mgnv4_resnet101v2_512,
    'mgnv4_resnet152_512': mgnv4_resnet152_512,
    'mgnv5_resnet101_256': mgnv5_resnet101_256,
    'mgnv5_seresnet101_256':mgnv5_seresnet101_256,
    'mgnv5_seresnet101_512': mgnv5_seresnet101_512,
    'mgnv5_resnext101_32x4d_256': mgnv5_resnext101_32x4d_256,
    'mgnv5_resnext101_32x4d_512': mgnv5_resnext101_32x4d_512,

    'mgnv6_resnet101_256': mgnv6_resnet101_256,
    'mgnv6_seresnet101_256': mgnv6_seresnet101_256,
    'mgnv6_seresnet101_512': mgnv6_seresnet101_512,
    'mgnv6_resnext101_32x4d_256': mgnv6_resnext101_32x4d_256,
    'mgnv6_resnext101_32x4d_512': mgnv6_resnext101_32x4d_512,

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
