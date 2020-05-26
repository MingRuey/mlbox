import math
import sys
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as kk

from .commons import conv1x1_block, conv7x7_block, groupconv3x3_block


PRETRAINED_PATH = Path(__file__).joinpath("pretrained")


class ResNetHead(kk.Layer):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(ResNetHead, self).__init__(**kwargs)
        self.conv = conv7x7_block(
            name="conv",
            in_channels=in_channels, out_channels=out_channels,
            strides=2, kernel_regularizer=keras.regularizers.l2(0.0001)
        )
        self.pool = kk.MaxPool2D(
            name="pool", pool_size=(3, 3),
            strides=2, padding="same"
        )

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.pool(x)
        return x


class ResNeXtBlock(kk.Layer):
    """A single ResNeXtblock"""

    def __init__(
            self,
            in_channels: int, out_channels: int, strides,
            cardinality: int, bottleneck_channel: int,
            **kwargs
            ):
        super(ResNeXtBlock, self).__init__(**kwargs)

        group_width = cardinality * bottleneck_channel
        self.conv1 = conv1x1_block(
            name="conv1x1-1",
            in_channels=in_channels, out_channels=group_width,
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )
        self.conv2 = groupconv3x3_block(
            name="groupconv3x3",
            in_channels=group_width, out_channels=group_width,
            groups=cardinality, strides=strides,
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )
        self.conv3 = conv1x1_block(
            name="conv1x1-2",
            in_channels=group_width, out_channels=out_channels,
            activation=None,
            kernel_regularizer=keras.regularizers.l2(0.0001)
        )

        is_identity = (strides == 1) and (in_channels == out_channels)
        self.identity = None if is_identity else \
            conv1x1_block(
                name="identity_conv",
                in_channels=in_channels, out_channels=out_channels,
                strides=strides, activation=None,
                kernel_regularizer=keras.regularizers.l2(0.0001)
            )
        self.activ = kk.ReLU()

    def call(self, x, training=None):
        if self.identity:
            identity = self.identity(x, training=training)
        else:
            identity = x
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = x + identity
        x = self.activ(x)
        return x


class ResNeXt(kk.Layer):
    """
    ResNeXt model from
    'Aggregated Residual Transformations for Deep Neural Networks,'
    http://arxiv.org/abs/1611.05431.
    """

    HEAD_CHANNELS = 64
    # from eq 4 in paper, the logics of d is actually
    # C * (out_channel * 2d + 9dd) equal to parameters of resnet,
    # which is 17 * out_channel / 16;
    # so when out_channels doubled as stage increases,
    # bottleneck channels should increase in some way.
    # Here we simply double it.
    CARDINALITY = 32
    BOTTLENECK_WIDTH = [4, 8, 16, 32]
    CHANNELS = [256, 512, 1024, 2048]

    def __init__(
            self,
            n_blocks: List[int],
            in_channels: int = 3,
            n_fc: int = 1000,
            include_fc: bool = True,
            **kwargs
            ):
        """
        Args:
            in_channels (int):
                number of channels of input image.
            n_blocks (List of int):
                specify number of residual blocks for each conv stage.
            n_fc (int):
                number of nodes in final fully connected layer,
                this is ignored if include_fc set to False.
            include_fc (bool):
                if include the fully connected layer,
                or only get feature map.
        """
        super().__init__(**kwargs)

        self.head = ResNetHead(
            name="init_block",
            in_channels=in_channels,
            out_channels=self.HEAD_CHANNELS,
        )

        # compose residual blocks
        self.convs = []
        in_channels = self.HEAD_CHANNELS
        for stage_idx, n_block in enumerate(n_blocks):
            for block_idx in range(n_block):
                out_channels = self.CHANNELS[stage_idx]
                bottleneck_channels = self.BOTTLENECK_WIDTH[stage_idx]
                strides = 2 if (stage_idx != 0) and (block_idx == 0) else 1

                self.convs.append(ResNeXtBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    cardinality=self.CARDINALITY,
                    bottleneck_channel=bottleneck_channels,
                    name="stage{}-unit{}".format(stage_idx + 1, block_idx + 1)
                ))
                in_channels = out_channels

        self.include_fc = include_fc
        if include_fc:
            self.pool = kk.GlobalAveragePooling2D()
            self.output1 = kk.Dense(
                units=n_fc, input_dim=in_channels, name="output1",
                kernel_regularizer=keras.regularizers.l2(0.0001)
            )

    def call(self, x, training=None):
        x = self.head(x)
        for conv in self.convs:
            x = conv(x, training=training)
        if self.include_fc:
            x = self.pool(x)
            x = self.output1(x)
        return x


def ResNeXt50(
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        include_fc: bool = True,
        n_fc: int = 1000,
        load_pretrained: bool = True,
        **kwargs
        ):

    PRETRAINED = str(PRETRAINED_PATH.joinpath("ResNeXt50"))

    inputs = keras.Input(shape=input_shape, name="input1")
    outputs = ResNeXt(
        name="ResNeXt50", n_blocks=[3, 4, 6, 3],
        in_channels=input_shape[-1],
        include_fc=include_fc, n_fc=n_fc,
        **kwargs
    )(inputs)
    return keras.Model(inputs, outputs)
