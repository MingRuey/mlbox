import math

import tensorflow as tf
import tensorflow.keras.layers as kk

from .activations import get_activation_layer


def _expand2tuple(size):
    """Convert size (for kernel, stride, padding .etc) to tuple if necessary"""
    if not isinstance(size, tuple):
        return (size, size)
    return size


class ConvBlock(kk.Layer):
    """Conv block with optional BN and activation

    Args:
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple/list of 2 int
            Convolution window size.
        strides : int or tuple/list of 2 int
            Strides of the convolution.
        padding : int or tuple/list of 2 int
            Padding value for convolution layer.
        groups : int, default 1
            Number of groups.
        use_bias : bool, default False
            Whether the layer uses a bias vector.
        use_bn : bool, default True
            Whether to use BatchNorm layer.
        bn_eps : float, default 1e-5
            Small float added to variance in Batch norm.
        activation : function or str or None, default 'relu'
            Activation function or name of activation function.
        kernel_initializer, kernel_regularizer, ...
            checkout tensorflow.keras.layers.Conv2D document
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 strides=1,
                 padding=0,
                 use_bias: bool = False,
                 use_bn: bool = True,
                 bn_eps: bool = 1e-5,
                 activation="relu",
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        kernel_size = _expand2tuple(kernel_size)
        strides = _expand2tuple(strides)
        padding = _expand2tuple(padding)

        use_pad = (padding[0] > 0) or (padding[1] > 0)

        self.pad = kk.ZeroPadding2D(padding=padding) if use_pad else None
        self.conv = kk.Conv2D(
            name="conv", filters=out_channels, use_bias=use_bias,
            kernel_size=kernel_size, strides=strides, padding="valid",
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            kernel_constraint=kernel_constraint
        )
        self.bn = kk.BatchNormalization(name="bn", epsilon=bn_eps) if use_bn else None
        self.activate = get_activation_layer(activation) if activation else None

    def call(self, x, training=None):
        if self.pad:
            x = self.pad(x)
        x = self.conv(x, training=training)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activate(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  strides=1,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  kernel_initializer="glorot_uniform",
                  kernel_regularizer=None,
                  kernel_constraint=None,
                  **kwargs):
    """
    1x1 version of the standard convolution block.

    Args:
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        strides : int or tuple/list of 2 int, default 1
            Strides of the convolution.
        use_bias : bool, default False
            Whether the layer uses a bias vector.
        use_bn : bool, default True
            Whether to use BatchNorm layer.
        bn_eps : float, default 1e-5
            Small float added to variance in Batch norm.
        activation : function or str or None, default 'relu'
            Activation function or name of activation function.
        kernel_initializer, kernel_regularizer, ...
            checkout tensorflow.keras.layers.Conv2D document
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint,
        **kwargs
    )


def conv7x7_block(in_channels,
                  out_channels,
                  strides=1,
                  padding=3,
                  use_bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation="relu",
                  kernel_initializer="glorot_uniform",
                  kernel_regularizer=None,
                  kernel_constraint=None,
                  **kwargs):
    """
    7x7 version of the standard convolution block.

    Args:
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        strides : int or tuple/list of 2 int, default 1
            Strides of the convolution.
        padding : int or tuple/list of 2 int, default 3
            Padding value for convolution layer.
        use_bias : bool, default False
            Whether the layer uses a bias vector.
        use_bn : bool, default True
            Whether to use BatchNorm layer.
        bn_eps : float, default 1e-5
            Small float added to variance in Batch norm.
        activation : function or str or None, default 'relu'
            Activation function or name of activation function.
        kernel_initializer, kernel_regularizer, ...
            checkout tensorflow.keras.layers.Conv2D document
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_constraint=kernel_constraint,
        kernel_regularizer=kernel_regularizer,
        **kwargs
    )


class GroupConv(kk.Layer):
    """Group Conv with optional BN and activation

    Note: if groups == 1, normal Conv2D is used.

    Args:
        groups: int, the number of groups for group convolutions
        **kwargs: same arguments as Conv2D, checkout doc for Conv2D
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 strides,
                 padding,
                 groups: int = 1,
                 use_bias: bool = False,
                 use_bn: bool = True,
                 bn_eps: bool = 1e-5,
                 activation="relu",
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super(GroupConv, self).__init__(**kwargs)
        kernel_size = _expand2tuple(kernel_size)
        strides = _expand2tuple(strides)
        padding = _expand2tuple(padding)

        use_pad = (padding[0] > 0) or (padding[1] > 0)
        self.pad = kk.ZeroPadding2D(padding=padding) if use_pad else None

        if groups == 1:
            self.conv = kk.Conv2D(
                name="conv", filters=out_channels, use_bias=use_bias,
                kernel_size=kernel_size, strides=strides, padding="valid",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                kernel_constraint=kernel_constraint
            )
        else:
            assert (in_channels % groups == 0)
            assert (out_channels % groups == 0)
            self.groups = groups
            self.convs = []
            for i in range(groups):
                self.convs.append(kk.Conv2D(
                    name="convgroup{}".format(i + 1),
                    filters=(out_channels // groups),
                    kernel_size=kernel_size,
                    strides=strides,
                    padding="valid",
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint
                ))

        self.bn = kk.BatchNormalization(name="bn", epsilon=bn_eps) if use_bn else None
        self.activate = get_activation_layer(activation) if activation else None

    def call(self, x, training=None):
        if self.pad:
            x = self.pad(x)

        if self.groups == 1:
            x = self.conv(x, training=training)
        else:
            yy = []
            xx = tf.split(x, num_or_size_splits=self.groups, axis=-1)
            for xi, convi in zip(xx, self.convs):
                yy.append(convi(xi, training=training))
            x = tf.concat(yy, axis=-1)
        if self.bn:
            x = self.bn(x, training=training)
        if self.activate:
            x = self.activate(x)
        return x


def groupconv3x3_block(in_channels,
                       out_channels,
                       groups,
                       strides=1,
                       padding=1,
                       use_bias=False,
                       use_bn=True,
                       bn_eps=1e-5,
                       activation="relu",
                       kernel_initializer="glorot_uniform",
                       kernel_regularizer=None,
                       kernel_constraint=None,
                       **kwargs):
    """
    3x3 version of the standard convolution block.

    Args:
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        groups: int
            Number of groups in Group convolution
        strides : int or tuple/list of 2 int, default 1
            Strides of the convolution.
        padding : int or tuple/list of 2 int, default 1
            Padding value for convolution layer.
        use_bias : bool, default False
            Whether the layer uses a bias vector.
        use_bn : bool, default True
            Whether to use BatchNorm layer.
        bn_eps : float, default 1e-5
            Small float added to variance in Batch norm.
        activation : function or str or None, default 'relu'
            Activation function or name of activation function.
        kernel_initializer, kernel_regularizer, ...
            checkout tensorflow.keras.layers.Conv2D document
    """
    return GroupConv(
        in_channels=in_channels,
        out_channels=out_channels,
        groups=groups,
        kernel_size=3,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        kernel_constraint=kernel_constraint,
        **kwargs
    )
