"""
Implement following papers:

GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training
"""
from typing import List

import tensorflow as tf
import tensorflow.keras as keras

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator
from MLBOX.Models.TF.Keras.DCGAN import _batch_relu, _batch_leaky


class AdvLoss:
    """Adversarial loss for Discriminator and Generator"""
    pass


class ContextLoss:
    """Contextual loss for Generator reconstruction"""
    pass


class EncoderLoss:
    """Loss on similarities between latent vectors in Generator"""
    pass


class GANomalyG:

    def __init__(self, latent_size: int, image_shape: tuple):
        self._shape = image_shape
        self.decoder = Generator(image_shape)
        self.encoder = Discriminator(n_out=latent_size)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        # note that self.encoder always create new tensor on __call__
        # so latent_i and latent_o have difference parametrizations
        latent_i = self.encoder(inputs)
        generated = self.decoder(latent_i)
        latent_o = self.encoder(generated)
        return generated, latent_i, latent_o


class GANomalyD(Discriminator):

    def __init__(self):
        super().__init__(n_out=1)


if __name__ == "__main__":
    pass
