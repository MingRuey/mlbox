import os
import sys

import pytest
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, Model
import matplotlib.pyplot as plt

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator
from MLBOX.Models.TF.Keras.DCGAN import ResDiscriminator


class TestGenerator:

    def test_generator(self):
        image_shape = (416, 416, 3)

        inputs = Input((100,))
        generator = Generator(image_shape=image_shape)(inputs)
        generator = Model(inputs=inputs, outputs=generator)

        prior = tf.random.normal([1, 100])
        pred = generator.predict(prior)
        pred = pred[0, ...]
        assert pred.shape == image_shape


class TestDiscriminator:

    def test_discriminator(self):
        image_shape = (416, 416, 3)

        inputs = Input(image_shape)
        discriminator = Discriminator()(inputs)
        discriminator = Model(inputs=inputs, outputs=discriminator)

        img = tf.ones(shape=(1, 416, 416, 3))
        pred = discriminator.predict(img)
        pred = pred[0, ...]
        assert pred.shape == (1,)

    def test_res_discriminator(self):
        image_shape = (416, 416, 3)

        inputs = Input(image_shape)
        discriminator = ResDiscriminator()(inputs)
        discriminator = Model(inputs=inputs, outputs=discriminator)
        img = tf.ones(shape=(1, 416, 416, 3))
        pred = discriminator.predict(img)
        pred = pred[0, ...]
        assert pred.shape == (1,)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
