from abc import ABC, abstractmethod
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, BatchNormalization, Activation,
    Reshape, Dense, Add
)
from tensorflow.keras.regularizers import l2


class IdentityBlock(Layer):

    def __init__(self, in_filters: int, out_filters: int):
        super().__init__()
        self._in_filters = in_filters
        self._out_filters = out_filters

    def build(self, build_shape):
        self._blocks = [
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=self._in_filters, kernel_size=1, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            ),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=self._in_filters, kernel_size=3, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            ),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=self._out_filters, kernel_size=1, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            )
        ]

    def call(self, x):
        inputs = x
        for layer in self._blocks:
            x = layer(x)
        outputs = Add()([x, inputs])
        return outputs


class ConvBlock(Layer):

    def __init__(self, in_filters: int, out_filters: int):
        super().__init__()
        self._in_filters = in_filters
        self._out_filters = out_filters

    def build(self, build_shape):
        self._blocks = [
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=self._in_filters, kernel_size=1, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            ),
            BatchNormalization(),
            Activation("relu"),
            Conv2DTranspose(
                filters=self._in_filters,
                kernel_size=3, strides=2, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            ),
            BatchNormalization(),
            Activation("relu"),
            Conv2D(
                filters=self._out_filters,
                kernel_size=1, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            )
        ]

        self._upsample = Conv2DTranspose(
                filters=self._out_filters,
                kernel_size=3, strides=2, padding="same",
                kernel_initializer="he_uniform",
                kernel_regularizer=l2(1e-4)
            )

    def call(self, x):
        shortcut = self._upsample(x)
        for layer in self._blocks:
            x = layer(x)
        outputs = Add()([x, shortcut])
        return outputs


class Stage(Layer):

    def __init__(
            self, in_filters: int, out_filters: int,
            num_of_resblock: int
            ):
        super().__init__()
        self._in_filters = in_filters
        self._out_filters = out_filters
        self._n = num_of_resblock

    def build(self, build_shape):
        self._layers = []
        for block in range(self._n):
            if block == 0:
                self._layers.append(
                    ConvBlock(
                        in_filters=self._in_filters,
                        out_filters=self._out_filters
                    )
                )
            else:
                self._layers.append(
                    IdentityBlock(
                        in_filters=self._in_filters,
                        out_filters=self._out_filters
                    )
                )

    def call(self, x):
        for layer in self._layers:
            x = layer(x)
        return x


class Decoder(Layer):
    """Decoder reconstructs x' from a input distribution (mean, stddev)"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def call(self, x):
        """x is a tensor of shape (None, latent size)"""
        pass


class ResNetDecoder(Decoder):
    """Resnet50 decoder"""

    def __init__(self, image_shape: tuple):
        super().__init__()
        img_h, img_w, *_ = image_shape
        if img_h % 32 != 0 or img_w % 32 != 0:
            msg = "image shape must be multiple of 32, get {}"
            raise ValueError(msg.format(image_shape))
        self._img_shp = (img_h, img_w)

    def build(self, build_shape):
        h, w = self._img_shp
        h = h // 32
        w = w // 32
        self._feat_dense = Dense(
            h * w,
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(1e-4)
        )
        self._feat_reshape = Reshape((h, w, 1))
        self._feat_conv = Conv2D(
            filters=2048,
            kernel_size=1, padding="same",
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(1e-4)
        )
        self._stage4 = Stage(
            in_filters=2048, out_filters=512, num_of_resblock=3
        )
        self._stage3 = Stage(
            in_filters=1024, out_filters=256, num_of_resblock=6
        )
        self._stage2 = Stage(
            in_filters=512, out_filters=128, num_of_resblock=4
        )
        self._stage1 = Stage(
            in_filters=256, out_filters=64, num_of_resblock=3
        )
        self._final_conv = Conv2DTranspose(
            filters=3, kernel_size=3, strides=2, padding="same",
            kernel_initializer="he_uniform",
            kernel_regularizer=l2(1e-4),
            name="decoder_output"
        )

    def call(self, x):
        """x is a tensor of shape (None, latent size)"""
        feat = self._feat_dense(x)
        feat = self._feat_reshape(feat)
        feat = self._feat_conv(feat)

        x = self._stage4(feat)
        x = self._stage3(x)
        x = self._stage2(x)
        x = self._stage1(x)
        x = self._final_conv(x)
        return x
