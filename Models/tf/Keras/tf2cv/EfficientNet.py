from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kk

from .commons import conv1x1_block, conv7x7_block, groupconv3x3_block


class EfficientBone(kk.Layer):
    pass


def EfficientNet(
        include_fc: bool = True,
        n_fc: int = 1000,
        load_pretrained: bool = True,
        **kwargs
        ):
    pass
