import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator
from MLBOX.Models.TF.Keras.UNet import UNET


class SkipGANomalyG:

    def __init__(self, *args, **kwargs):
        self._unet = UNET(*args, **kwargs)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self._unet.forward(input)


class SkipGANomalyD:

    def __init__(self):
        self._disc = Discriminator(return_flatten=True)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        latent = self._disc(inputs)
        dense = Dense(1)(latent)
        return latent, dense


if __name__ == "__main__":
    pass
