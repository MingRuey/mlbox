import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
from tensorflow.keras import Input, Model  # noqa: E402
from tensorflow.keras.layers import Dense, Conv2D  # noqa: E402
from tensorflow.keras.initializers import Constant  # noqa: E402
from tensorflow.keras.losses import MeanSquaredError  # noqa: E402
from tensorflow.keras.optimizers import SGD  # noqa: E402

from MLBOX.Models.tf.Keras.WGAN import WeightsClip, WGenerator, WDiscriminator  # noqa: E402


class TestWeightsClip:

    def test_weights_clip(self):
        """WeightsClip should restrict values after optimizer updates"""
        constraint = WeightsClip(min_value=-0.001, max_value=0.001)
        initializer = Constant(1.0)

        layer = Dense(
            10, kernel_constraint=constraint, bias_constraint=constraint,
            kernel_initializer=initializer, bias_initializer=initializer
        )

        with tf.GradientTape() as tape:
            data = tf.random.normal(shape=(1, 5))
            pred = layer(data)
            true_lb = tf.ones_like(pred)
            loss = MeanSquaredError()(true_lb, pred)

        # before update, weights should equal to initial values
        for weight in layer.weights:
            assert np.allclose(weight.numpy(), 1.0)

        optimizer = SGD(learning_rate=0.0001)
        grad = tape.gradient(loss, layer.trainable_variables)
        optimizer.apply_gradients(zip(grad, layer.trainable_variables))

        # after update, weights should within constraint ranges
        for weight in layer.weights:
            assert np.all(-0.001 <= weight.numpy())
            assert np.all(weight.numpy() <= 0.001)


class TestWGAN:

    def test_generator_output(self):
        """Generator should yield image with range [0, 1]"""
        input_shape = (64, 64, 3)
        latent_size = 5

        inputs = Input(latent_size)
        generator = WGenerator(image_shape=input_shape)(inputs)
        generator = Model(inputs=inputs, outputs=generator)

        prior = tf.random.normal([1, latent_size])
        pred = generator.predict(prior)
        assert np.all(pred <= 1)
        assert np.all(0 <= pred)

    def test_discriminator_weight_updates(self):
        """Discriminator weights should be restrcited into constraint range"""

        inputs = Input((64, 64, 3))
        discriminator = WDiscriminator()(inputs)
        discriminator = Model(inputs=inputs, outputs=discriminator)

        with tf.GradientTape() as tape:
            data = tf.random.normal((1, 64, 64, 3))
            pred = discriminator(data, training=True)

            true_lb = tf.ones_like(pred)
            loss = MeanSquaredError()(true_lb, pred)

        optimizer = SGD(learning_rate=1.0)
        grad = tape.gradient(loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(grad, discriminator.trainable_variables))

        # after update, weights should within constraint ranges
        for weight in discriminator.trainable_variables:
            assert np.all(-0.01 <= weight.numpy())
            assert np.all(weight.numpy() <= 0.01)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
