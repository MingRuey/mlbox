import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from .DCGAN import Generator, Discriminator


class GANomalyG:

    def __init__(self, image_shape):
        self._encoder = Discriminator(return_flatten=True)
        self._gen = Generator(image_shape=image_shape)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        encoded = self._encoder(inputs)
        generated = self._gen(encoded)
        return generated


class GANomalyD:

    def __init__(self):
        self._disc = Discriminator(return_flatten=True)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        latent = self._disc(inputs)
        dense = Dense(1)(latent)
        return latent, dense


if __name__ == "__main__":
    pass
