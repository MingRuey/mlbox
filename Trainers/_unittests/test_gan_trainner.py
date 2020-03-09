import pathlib

import pytest
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import SGD, Adam

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator
from MLBOX.Trainers.GANtrainner import KerasGANTrainner


LATENT_SIZE = 10
INPUT_SHAPE = (32, 32, 3)


def _get_adam():
    return Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-04,
        amsgrad=True,
        clipnorm=1.0,
        clipvalue=1.0
    )


def DCGAN():
    inputs = Input((LATENT_SIZE,))
    generator = Generator(image_shape=INPUT_SHAPE)(inputs)
    generator = Model(inputs=inputs, outputs=generator)

    inputs = Input(INPUT_SHAPE)
    discriminator = Discriminator()(inputs)
    discriminator = Model(inputs=inputs, outputs=discriminator)

    return generator, discriminator


class MockDataset:

    DATADIR = "/rawdata2/tensorflow_datasets/"

    def get_dataset(self, epoch: int, batchsize: int):
        tf_flowers = tfds.load(
            name="tf_flowers", data_dir=self.DATADIR, batch_size=batchsize
        )

        tf_flowers = tf_flowers["train"]
        tf_flowers = tf_flowers.map(self.mapping)
        return tf_flowers

    @property
    def data_count(self):
        return 3670

    @staticmethod
    def mapping(example):
        img = example["image"]
        img = tf.image.resize(img, INPUT_SHAPE[:2])
        img = img / 255
        noise = tf.random.normal((tf.shape(img)[0], LATENT_SIZE))
        return img, noise


def test_gan_trainner():
    dataset = MockDataset()
    generator, discriminator = DCGAN()

    trainner = KerasGANTrainner(
        generator=generator,
        discriminator=discriminator,
        gen_optimizer=_get_adam(),
        disc_optimizer=_get_adam(),
        out_dir="."
    )

    trainner.train(
        database=dataset, batch_size=4, max_epoch=5
    )


if __name__ == "__main__":
    test_gan_trainner()
