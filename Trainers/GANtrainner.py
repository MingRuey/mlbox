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

from ..Database.core.database import Dataset  # noqa: E402
from .TF.Callbacks import ModelLogger, TrainRecord  # noqa: E402
from .TF.Metrics import SSIM  # noqa: E402
from .TF.Loss import SSIMLoss  # noqa: E402


class KerasGANTrainner:

    def __init__(
            self,
            generator: keras.Model, discriminator: keras.Model,
            gen_optimizer: keras.optimizers.Optimizer,
            gen_loss: keras.losses.Loss,
            disc_optimizer: keras.optimizers.Optimizer,
            disc_loss: keras.losses.Loss,
            out_dir: str,
            gen_metrics: keras.metrics.Metric = None,
            disc_metrics: keras.metrics.Metric = None,
            disc_update_per_batch: int = 1
            ):
        """Create a trainner for GAN architecture

        Args:
            generator (keras.Model): generator model
            discriminator (keras.Model): discriminator model
            gen_optimizer, disc_optimizer (keras.optimers.Optimizer):
                the optimizer for training generator/discriminator
            gen_loss, disc_loss (keras.losses.Loss):
                the loss fcuntion for generator/discriminator
            gen_metrics, disc_metrics (keras.metrics.Metric, optional):
                Metrics for generator and discrimnator to monitor.
                Defaults to None.
            disc_update_per_batch (int):
                number of iteration per batch on discriminator
        """
        self._gen = generator
        self._gen_optimzier = gen_optimizer
        self._gen_loss = gen_loss
        self._constrcut_loss = SSIMLoss(max_val=1)
        self._gen_name = "Gen_{:03d}_loss-{:.5f}_construct-{:.5f}.h5"

        self._disc = discriminator
        self._disc_optimzier = disc_optimizer
        self._disc_loss = disc_loss
        self._disc_update_per_batch = disc_update_per_batch
        self._disc_name = "Disc_{:03d}_{:.5f}.h5"

        if not pathlib.Path(str(out_dir)).is_dir():
            msg = "Invalid output dir, got {}"
            raise ValueError(msg.format(out_dir))

        self.out_dir = str(out_dir)
        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

    def train(
            self,
            database: Dataset,
            batch_size: int = 8,
            max_epoch: int = 200,
            load_best: bool = True,
            ):

        if load_best:
            gen_weights = list(pathlib.Path(self.out_dir).glob("Gen_*.h5"))
            if gen_weights:
                gen_weights = gen_weights[0]
                filename = gen_weights.name
                self._gen.load_weights(str(gen_weights))
                print("Generator load pretrain weights from {}".format(filename))

            disc_weights = list(pathlib.Path(self.out_dir).glob("Disc_*.h5"))
            if disc_weights:
                disc_weights = disc_weights[0]
                filename = disc_weights.name
                self._disc.load_weights(str(disc_weights))
                print("Discriminator load pretrain weights from {}".format(filename))

        dataset = database.to_tfdataset(epoch=max_epoch, batch=batch_size)
        dataset = iter(dataset)
        steps_per_epoch = database.data_count // batch_size

        for epoch in range(max_epoch):
            print("epoch {} starts".format(epoch))
            epoch_disc_loss = keras.metrics.Mean()
            epoch_gen_loss = keras.metrics.Mean()
            epoch_construct_loss = keras.metrics.Mean()
            epoch_adv_loss = keras.metrics.Mean()

            for batch in range(steps_per_epoch):
                gt_imgs, noises = next(dataset)
                disc_loss, gen_loss, adv_loss, construct_loss = \
                    self._train_step(gt_imgs, noises)

                epoch_gen_loss(gen_loss)
                epoch_disc_loss(disc_loss)
                epoch_adv_loss(adv_loss)
                epoch_construct_loss(construct_loss)

                msg = "    batch {:03d} - Gen loss: {:.5f} (adv: :{:.5f}, construct: {:.5f}); Disc loss: {:.5f}"
                print(msg.format(batch, gen_loss, adv_loss, construct_loss, disc_loss), end="\r")

            msg = "epoch {} ends, Gen loss: {} (adv: {}, construct: {}), Disc loss: {}"
            print(msg.format(
                epoch, epoch_gen_loss.result(),
                epoch_adv_loss.result(), epoch_construct_loss.result(),
                epoch_disc_loss.result()
            ))

            file_disc = os.path.join(
                self.tmp_dir, self._disc_name.format(epoch, epoch_disc_loss.result())
            )
            self._disc.save_weights(file_disc)

            file_gen = os.path.join(
                self.tmp_dir, self._gen_name.format(epoch, epoch_gen_loss.result(), epoch_construct_loss.result())
            )
            self._gen.save_weights(file_gen)

    @tf.function
    def _train_step(self, gt_imgs: tf.Tensor, noises: tf.Tensor):
        """Single train step - optimizer minimize a batch over two models"""
        batch = gt_imgs.shape[0]

        # update discriminator
        for _ in range(self._disc_update_per_batch):
            with tf.GradientTape() as disc_tape:
                fake_imgs = self._gen(noises, training=True)

                true_pred = self._disc(gt_imgs, training=True)
                fake_pred = self._disc(fake_imgs, training=True)
                pred = tf.concat([true_pred, fake_pred], axis=-1)

                true_lb = tf.ones_like(true_pred)
                fake_lb = -1 * tf.ones_like(fake_pred)
                disc_lb = tf.concat([true_lb, fake_lb], axis=-1)

                disc_loss = self._disc_loss(disc_lb, pred)

            disc_grad = disc_tape.gradient(
                disc_loss,
                self._disc.trainable_variables
            )
            self._disc_optimzier.apply_gradients(
                zip(disc_grad, self._disc.trainable_variables)
            )

        # update generator
        with tf.GradientTape() as gen_tape:
            fake_imgs = self._gen(noises, training=True)
            fake_pred = self._disc(fake_imgs, training=True)

            gen_lb = tf.ones_like(fake_pred)
            adv_loss = self._gen_loss(gen_lb, fake_pred)
            construct_loss = self._constrcut_loss(fake_imgs, gt_imgs)
            gen_loss = adv_loss + construct_loss

        gen_grad = gen_tape.gradient(
            gen_loss, self._gen.trainable_variables
        )
        self._gen_optimzier.apply_gradients(
            zip(gen_grad, self._gen.trainable_variables)
        )
        return (
            disc_loss / (2*batch),
            gen_loss / batch, adv_loss / batch, construct_loss / batch
        )
