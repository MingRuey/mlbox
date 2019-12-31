import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Dense, BatchNormalization, LeakyReLU, ReLU,
    Reshape, Conv2DTranspose, Conv2D, Flatten
)


def _batch_relu(inputs: tf.Tensor):
    """helper function for creating bn with relu activation"""
    bn = BatchNormalization()(inputs)
    leaky = ReLU()(bn)
    return leaky


def _batch_leaky(inputs: tf.Tensor):
    """helper function for creating bn with leaky relu activation"""
    bn = BatchNormalization()(inputs)
    leaky = LeakyReLU()(bn)
    return leaky


class Generator:

    def __init__(self, image_shape: tuple):
        if (image_shape[0] % 16 != 0) or (image_shape[1] % 16 != 0):
            msg = "Image height/width must be multiple of 16, got: {}"
            raise ValueError(msg.format(image_shape))
        h, w, *_ = image_shape
        self._h, self._w = h // 16, w // 16

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        dense = Dense(self._h * self._w * 1024)(inputs)
        dense = Reshape((self._h, self._w, 1024))(dense)
        dense = _batch_leaky(dense)

        conv1 = Conv2DTranspose(
            filters=512, kernel_size=(5, 5), strides=(2, 2),
            padding="same", use_bias=False
        )(dense)
        conv1 = _batch_relu(conv1)

        conv2 = Conv2DTranspose(
            filters=256, kernel_size=(5, 5), strides=(2, 2),
            padding="same", use_bias=False
        )(conv1)
        conv2 = _batch_relu(conv2)

        conv2 = Conv2DTranspose(
            filters=128, kernel_size=(5, 5), strides=(2, 2),
            padding="same", use_bias=False,
        )(conv2)
        conv2 = _batch_relu(conv2)

        conv3 = Conv2DTranspose(
            filters=3, kernel_size=(5, 5), strides=(2, 2),
            padding="same", use_bias=False,
            activation="tanh"
        )(conv2)

        return conv3


class Discriminator:

    def __init__(self):
        pass

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        conv1 = Conv2D(
            filters=128, kernel_size=(5, 5), strides=(2, 2),
            padding='same'
        )(inputs)
        conv1 = _batch_leaky(conv1)

        conv2 = Conv2D(
            filters=256, kernel_size=(5, 5), strides=(2, 2),
            padding='same'
        )(conv1)
        conv2 = _batch_leaky(conv2)

        conv3 = Conv2D(
            filters=512, kernel_size=(5, 5), strides=(2, 2),
            padding='same'
        )(conv2)
        conv3 = _batch_leaky(conv3)

        conv4 = Conv2D(
            filters=1024, kernel_size=(5, 5), strides=(2, 2),
            padding='same'
        )(conv3)
        conv4 = _batch_leaky(conv4)

        dense = Flatten()(conv4)
        dense = Dense(1)(dense)
        return dense


class ResDiscriminator:

    def __init__(self):
        pass

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        resnet = keras.applications.ResNet50V2(
            include_top=False, weights=None,
            input_tensor=inputs, input_shape=None,
            pooling=None
        )
        res_output = resnet.output
        dense = Flatten()(res_output)
        dense = Dense(1)(dense)
        return dense


if __name__ == "__main__":
    pass
