from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D


class Encoder(Layer):
    """Encoder yield the (mean, stddev) for distribution p(z|x)"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def call(self, x):
        """x is the input image"""
        pass


class ResNetEncoder(Encoder):
    """Resnet50 encoder"""

    def __init__(
            self,
            latent_size: int,
            load_pretrained: bool = False
            ):
        super().__init__()
        self._latent_size = int(latent_size)
        self._load_pretrained = load_pretrained

    def build(self, input_shape):
        self._res50enc = keras.applications.ResNet50V2(
            include_top=False,
            weights="imagenet" if self._load_pretrained else None,
            input_shape=input_shape[1:],
            pooling=None
        )
        self._mean_conv = Conv2D(
            filters=self._latent_size, kernel_size=(1, 1),
            padding='same', kernel_initializer="he_normal"
        )
        self._mean = GlobalAveragePooling2D(name="encoder_mean")

        self._logvar_conv = Conv2D(
            filters=self._latent_size, kernel_size=(1, 1),
            padding='same', kernel_initializer="he_normal",
        )
        self._logvar = GlobalAveragePooling2D(name="encoder_logvar")

    def call(self, x):
        """x is the input image"""
        feature_map = self._res50enc(x)

        mean_conv = self._mean_conv(feature_map)
        mean = self._mean(mean_conv)

        logvar_conv = self._logvar_conv(feature_map)
        logvar = self._logvar(logvar_conv)
        return K.stack([mean, logvar], axis=-1)


def latent_loss(mean, logvar):
    beta = 0.01
    latent_loss = -0.5 * (1 + logvar - K.square(mean) - K.exp(logvar))
    latent_loss = K.sum(latent_loss, axis=-1)  # sum over latent dimension
    latent_loss = K.mean(latent_loss, axis=0)  # avg over batch
    latent_loss = beta * latent_loss
    return latent_loss


class SampleLayer(Layer):
    """SampleLayer conducts the reparameterization trick"""

    def __init__(self):
        super().__init__()

    def call(self, x):
        """x is a tensor of shape (None, latent_size, 2)

        Note, the first element of last dimension is mean,
        the second is log var
        """
        z_mean = x[..., 0]
        z_log_var = x[..., 1]
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]

        epsilon = K.random_normal(
            shape=(batch, dim),
            mean=0.0, stddev=1.0
        )

        # loss = latent_loss(z_mean, z_log_var)
        # self.add_loss(loss)
        # self.add_metric(loss, aggregation="mean", name="latent_loss")
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
