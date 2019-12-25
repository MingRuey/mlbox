import os
import sys
import logging
import pathlib
import numpy as np

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[logging.FileHandler("{}.log".format(file))]
    )

import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense   # noqa: E402
from tensorflow.keras.losses import BinaryCrossentropy  # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402

from MLBOX.Database.dataset import DataBase  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainers.TF.Keras_Callbacks import ModelLogger, TrainRecord  # noqa: E402


def disc_loss_fn(true_pred: tf.Tensor, fake_pred: tf.Tensor):
    """Loss function for discriminator

    Args:
        true_pred (tf.Tensor): the predictions of discriminator of true images
        fake_pred (tf.Tensor): the predictions of discriminator of fake images
    """
    true_lb = 0.9 * tf.ones_like(true_pred)
    fake_lb = tf.zeros_like(fake_pred)

    lb = tf.stack([true_lb, fake_lb], axis=-1)
    pred = tf.stack([true_pred, fake_pred], axis=-1)
    return keras.losses.binary_crossentropy(lb, pred)


def gen_loss_fn(fake_pred: tf.Tensor):
    lb = tf.ones_like(fake_pred)
    return keras.losses.binary_crossentropy(lb, fake_pred)


class KerasGANTrainner:

    def __init__(
            self,
            generator: keras.Model, discriminator: keras.Model,
            gen_optimizer: keras.optimizers.Optimizer,
            disc_optimizer: keras.optimizers.Optimizer,
            out_dir: str,
            gen_metrics: keras.metrics.Metric = None,
            disc_metrics: keras.metrics.Metric = None,
            ):
        """Create a trainner for GAN architecture

        Args:
            generator (keras.Model): generator model
            discriminator (keras.Model): discriminator model
            gen_optimizer, disc_optimizer (keras.optimers.Optimizer):
                the optimizer for training generator/discriminator
            gen_metrics, disc_metrics (keras.metrics.Metric, optional):
                Metrics for generator and discrimnator to monitor.
                Defaults to None.
        """
        self._gen = generator
        self._gen_optimzier = gen_optimizer

        self._disc = discriminator
        self._disc_optimzier = disc_optimizer

        if not pathlib.Path(str(out_dir)).is_dir():
            msg = "Invalid output dir, got {}"
            raise ValueError(msg.format(out_dir))
        self.out_dir = str(out_dir)
        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

    def train(
            self,
            database: DataBase,
            batch_size: int = 8,
            max_epoch: int = 200,
            ):

        dataset = database.get_dataset(epoch=max_epoch, batchsize=batch_size)
        dataset = iter(dataset)
        steps_per_epoch = database.data_count // batch_size

        for epoch in range(max_epoch):
            print("epoch {} starts".format(epoch))
            for batch in range(steps_per_epoch):
                gt_imgs, noises = next(dataset)
                gen_loss, disc_loss = self._train_step(gt_imgs, noises)
                msg = "    batch {} - Generator loss: {}; Descriminator loss: {}"
                print(msg.format(batch, gen_loss, disc_loss))
            print("epoch {} ends".format(epoch))

    @tf.function
    def _train_step(self, gt_imgs: tf.Tensor, noises: tf.Tensor):
        """[summary]

        Args:
            images (tf.Tensor): [description]
            noise (tf.Tensor): [description]
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self._gen(noises, training=True)

            true_pred = self._disc(gt_imgs, training=True)
            fake_pred = self._disc(fake_imgs, training=True)
            pred = tf.concat([true_pred, fake_pred], axis=-1)

            true_lb = 0.9 * tf.ones_like(true_pred)
            fake_lb = tf.zeros_like(fake_pred)
            disc_lb = tf.concat([true_lb, fake_lb], axis=-1)

            gen_lb = tf.ones_like(fake_pred)

            disc_loss = BinaryCrossentropy(from_logits=True)(disc_lb, pred)
            gen_loss = BinaryCrossentropy(from_logits=True)(gen_lb, fake_pred)

        disc_grad = disc_tape.gradient(
            disc_loss,
            self._disc.trainable_variables
        )
        self._disc_optimzier.apply_gradients(
            zip(disc_grad, self._disc.trainable_variables)
        )

        gen_grad = gen_tape.gradient(
            gen_loss,
            self._gen.trainable_variables
        )
        self._gen_optimzier.apply_gradients(
            zip(gen_grad, self._gen.trainable_variables)
        )
        return disc_loss, gen_loss
