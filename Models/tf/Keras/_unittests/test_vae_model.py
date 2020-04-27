import os
import sys

import pytest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model

from MLBOX.Models.TF.Keras.modules.ResDecoder import ResNetDecoder
from MLBOX.Models.TF.Keras.betaVAE import ResNetEncoder, SampleLayer


class TestDecoderEncoder:

    # @pytest.mark.skip(reason="stable")
    def test_resnet_encoder(self):
        shape = (512, 512, 3)
        latent_size = 100

        inputs = Input(shape=shape)
        outputs = ResNetEncoder(
            latent_size=latent_size,
            load_pretrained=False
        )(inputs)

        model = Model(inputs=inputs, outputs=outputs)

        image = tf.ones(shape=(1, *shape))
        outputs = model.predict(image)

        assert outputs.shape == (1, latent_size, 2)

    # @pytest.mark.skip(reason="stable")
    def test_resnet_decoder(self):
        shape = (512, 512, 3)
        latent_size = 100

        inputs = Input(shape=(latent_size,))
        reconstruct = ResNetDecoder(image_shape=shape)(inputs)

        model = Model(inputs=inputs, outputs=reconstruct)

        latent_variable = tf.ones(shape=(1, latent_size))
        pred = model.predict(latent_variable)

        assert pred.shape == (1, *shape)


class TestSampleLayer:

    def test_sample_layer(self):
        latent_size = 100

        latent = Input(shape=(latent_size, 2))
        reparameterization = SampleLayer()(latent)
        model = Model(inputs=latent, outputs=reparameterization)

        pseudo_latent = tf.zeros(shape=(1, latent_size, 2))
        pred = model.predict(pseudo_latent)
        print(pred)
        print(pred.shape)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
