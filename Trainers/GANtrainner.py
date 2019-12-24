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
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402

from MLBOX.Datbase.dataset import DataBase  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainers.TF.Keras_Callbacks import ModelLogger, TrainRecord  # noqa: E402


def disc_loss(true_pred: tf.Tensor, fake_pred: tf.Tensor):
    """Loss function for discriminator

    Args:
        true_pred (tf.Tensor): the predictions of discriminator of true images
        fake_pred (tf.Tensor): the predictions of discriminator of fake images
    """
    true_lb = 0.9 * tf.ones_like(true_pred)
    fake_lb = tf.zeors_like(fake_pred)

    lb = tf.stack([true_lb, fake_lb], axis=-1)
    pred = tf.stack([true_pred, fake_pred], axis=-1)
    return keras.losses.BinaryCrossentropy(lb, pred)


def gen_loss(fake_pred: tf.Tensor):
    lb = tf.ones_like(fake_pred)
    return keras.losses.BinaryCrossentropy(lb, fake_pred)


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
        self._gen_loss = gen_loss

        self._disc = discriminator
        self._disc_optimzier = disc_optimizer
        self._disc_loss = disc_loss

        if not pathlib.Path(str(out_dir)).is_dir():
            msg = "Invalid output dir, got {}"
            raise ValueError(msg.format(out_dir))
        self.out_dir = str(out_dir)
        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

    def train(
            self,
            database: Database,
            batch_size: int = 8,
            max_epoch: int = 200,
            ):

        dataset = database.get_datase(epoch=max_epoch, batch_size=batch_size)
        steps_per_epoch = database.data_count // batch_size,

        epoch = 0
        for imgs, noises in dataset:
            epoch += 1
            batch = 0
            print("epoch {} starts".format(epoch))
            for batch in steps_per_epoch:
                batch += 1
                gen_loss, disc_loss = _train_step(imgs, noises)
                msg = "    batch {} - Generator loss: {}; Descriminator loss: {}"
                print(msg.format(batch, gen_loss, disc_loss))
            print("epoch {} ends".format(epoch))

    @tf.function
    def _train_step(self, gt_imgs: tf.Tensor, noise: tf.Tensor):
        """[summary]

        Args:
            images (tf.Tensor): [description]
            noise (tf.Tensor): [description]
        """
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_imgs = self._gen(noise, training=True)

            real_out = self._disc(gt_imgs, training=True)
            fake_out = self._disc(fake_imgs, training=True)

            gen_loss = gen_loss(fake_out)
            disc_loss = disc_loss(real_out, fake_out)

        self._gen_optimzier.minimize(gen_loss, self._gen.trainable_variables)
        self._disc.minimize(gen_loss, self._disc.trainable_variables)
        return gen_loss, disc_loss
