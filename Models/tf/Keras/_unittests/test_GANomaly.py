import pytest

import tensorflow as tf
from tensorflow.keras import Input, Model

from MLBOX.Models.tf.Keras.GANomaly import GANomalyG, GANomalyD


class TestGANomalyG:

    def test_netG_output(self):
        """G should return (generated image, internal latent, output latent"""
        image_shape = (416, 416, 3)
        latent_size = 100

        inputs = Input(image_shape)
        gnet = GANomalyG(latent_size, image_shape)(inputs)
        gnet = Model(inputs=inputs, outputs=gnet)

        prior = tf.random.normal([1, *image_shape])
        generated, latent_i, latent_o = gnet.predict(prior)

        assert generated[0, ...].shape == image_shape
        assert latent_i[0, ...].shape == (latent_size,)
        assert latent_o[0, ...].shape == (latent_size,)

    def test_netD_output(self):
        """D should return single probability"""
        image_shape = (416, 416, 3)

        inputs = Input(image_shape)
        dnet = GANomalyD()(inputs)
        dnet = Model(inputs=inputs, outputs=dnet)

        img = tf.ones(shape=(1, 416, 416, 3))
        pred = dnet.predict(img)
        pred = pred[0, ...]
        assert pred.shape == (1,)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
