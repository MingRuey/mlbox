import os
import sys
import logging
import pathlib
import numpy as np
from collections import namedtuple

file = os.path.basename(__file__)
file = pathlib.Path(file).stem
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(message).1000s ',
    handlers=[logging.FileHandler("{}.log".format(file))]
    )

import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402

from MLBOX.Database.dataset import DataBase  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainers.TF.Callbacks import ModelLogger, TrainRecord  # noqa: E402
from MLBOX.Trainers.TF.Metrics import SSIM  # noqa: E402
from MLBOX.Trainers.TF.Loss import SSIMLoss  # noqa: E402


LossCollections = namedtuple(
    "LossCollections",
    ["disc", "gen", "adv", "construct", "latent"]
)


def format_losses(losses: LossCollections):
    string = "Gen: {:.5f} (adv: {:.5f}, construct: {:.5f}, latent: {:.5f}); Disc: {:.5f}"
    return string.format(
        losses.gen, losses.adv, losses.construct, losses.latent, losses.disc
    )


class SkipGANomalyTrainer:

    def __init__(
            self,
            generator: keras.Model, discriminator: keras.Model,
            gen_optimizer: keras.optimizers.Optimizer,
            disc_optimizer: keras.optimizers.Optimizer,
            adv_loss: keras.losses.Loss,
            latent_loss: keras.losses.Loss,
            construct_loss: keras.losses.Loss,
            out_dir: str,
            disc_update_per_batch: int = 1
            ):
        """Create a trainner for GAN architecture

        Args:
            generator (keras.Model): generator model
            discriminator (keras.Model): discriminator model
            gen_optimizer, disc_optimizer (keras.optimers.Optimizer):
                the optimizer for training generator/discriminator
            adv_loss (keras.losses.Loss):
                common loss function of adversarial training for
                both generator/discriminator
            construct_loss (keras.losses.Loss):
                reconstruction loss for generator
            latent_loss (keras.losses.Loss):
                latent loss for generator
            disc_update_per_batch (int):
                number of iteration per batch on discriminator
        """
        self._adv_loss = adv_loss

        self._gen = generator
        self._gen_optimzier = gen_optimizer
        self._construct_loss = construct_loss
        self._latent_loss = latent_loss
        self._gen_name = "Gen_{:03d}_{:.5f}.h5"

        self._disc = discriminator
        self._disc_optimzier = disc_optimizer
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
            train_db: DataBase,
            valid_normals: DataBase,
            valid_defects: DataBase,
            batch_size: int = 8,
            max_epoch: int = 200,
            load_best: bool = True,
            ):
        """Train on SkipGANomaly

        Args:
            train_db (DataBase):
                the training database, all images should be normal samples.
            valid_normals:
                the validation database,
                all images are considered normal for acc metric.
            valid_normals:
                the validation database,
                all images are considered defect for acc metric.
            batch_size (int, optional): Defaults to 8.
            max_epoch (int, optional): Defaults to 200.
            load_best (bool, optional): Defaults to True.
        """

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

        trainset = train_db.get_dataset(epoch=max_epoch, batchsize=batch_size)
        trainset = iter(trainset)
        steps_per_epoch = train_db.data_count // batch_size

        n_steps = valid_normals.data_count // batch_size
        d_steps = valid_defects.data_count // batch_size
        valid_n = valid_normals.get_dataset(epoch=max_epoch, batchsize=batch_size)
        valid_n = iter(valid_n)
        valid_d = valid_defects.get_dataset(epoch=max_epoch, batchsize=batch_size)
        valid_d = iter(valid_d)

        for epoch in range(max_epoch):
            print("epoch {} starts".format(epoch))
            epoch_disc = keras.metrics.Mean()
            epoch_gen = keras.metrics.Mean()
            epoch_construct = keras.metrics.Mean()
            epoch_adv = keras.metrics.Mean()
            epoch_latent = keras.metrics.Mean()

            for batch in range(steps_per_epoch):
                gt_imgs = next(trainset)
                losses = self._train_step(gt_imgs)

                epoch_disc(losses.disc)
                epoch_gen(losses.gen)
                epoch_adv(losses.adv)
                epoch_construct(losses.construct)
                epoch_latent(losses.latent)

                msg = "    batch {:03d} - {}"
                print(msg.format(batch, format_losses(losses)), end="\r")

            msg = "epoch {} ends - {}"
            epoch_loss = LossCollections(
                disc=epoch_disc.result(), gen=epoch_gen.result(),
                adv=epoch_adv.result(), construct=epoch_construct.result(),
                latent=epoch_latent.result()
            )
            print(msg.format(epoch, format_losses(epoch_loss)))

            # validations
            n_construct, n_latent = self._validate(valid_n, n_steps)
            d_construct, d_latent = self._validate(valid_d, d_steps)
            print("    -- normal construct: {}, latnet: {}".format(n_construct, n_latent))
            print("    -- defect construct: {}, latnet: {}".format(d_construct, d_latent))

            file_disc = os.path.join(
                self.tmp_dir, self._disc_name.format(epoch, epoch_disc.result())
            )
            self._disc.save_weights(file_disc)

            file_gen = os.path.join(
                self.tmp_dir, self._gen_name.format(epoch, epoch_gen.result())
            )
            self._gen.save_weights(file_gen)

    def _validate(self, dataset: tf.data.Dataset, steps: int):
        epoch_construct = keras.metrics.Mean()
        epoch_latent = keras.metrics.Mean()
        for _ in range(steps):
            gt_imgs = next(dataset)
            construct_loss, latent_loss = self._validate_step(gt_imgs)
            epoch_construct(construct_loss)
            epoch_latent(latent_loss)
        return epoch_construct.result(), epoch_latent.result()

    @tf.function
    def _train_step(self, gt_imgs: tf.Tensor):
        """Single train step - optimizer minimize a batch over two models"""
        batch = gt_imgs.shape[0]

        # update discriminator
        for _ in range(self._disc_update_per_batch):
            with tf.GradientTape() as disc_tape:
                fake_imgs = self._gen(gt_imgs, training=True)

                _, fake_pred = self._disc(fake_imgs, training=True)
                _, true_pred = self._disc(gt_imgs, training=True)
                pred = tf.concat([true_pred, fake_pred], axis=-1)

                true_lb = tf.ones_like(true_pred)
                fake_lb = -1 * tf.ones_like(fake_pred)
                disc_lb = tf.concat([true_lb, fake_lb], axis=-1)

                disc_loss = self._adv_loss(disc_lb, pred)

            disc_grad = disc_tape.gradient(
                disc_loss, self._disc.trainable_variables
            )
            self._disc_optimzier.apply_gradients(
                zip(disc_grad, self._disc.trainable_variables)
            )

        # update generator
        with tf.GradientTape() as gen_tape:
            fake_imgs = self._gen(gt_imgs, training=True)
            fake_latent, fake_pred = self._disc(fake_imgs, training=True)
            true_latent, _ = self._disc(gt_imgs, training=True)

            construct_loss = self._construct_loss(gt_imgs, fake_imgs)
            latent_loss = self._latent_loss(true_latent, fake_latent)

            gen_lb = tf.ones_like(fake_pred)
            adv_loss = self._adv_loss(gen_lb, fake_pred)

            gen_loss = construct_loss + latent_loss + adv_loss

        gen_grad = gen_tape.gradient(
            gen_loss, self._gen.trainable_variables
        )
        self._gen_optimzier.apply_gradients(
            zip(gen_grad, self._gen.trainable_variables)
        )
        return LossCollections(
            disc=disc_loss / (2*batch), gen=gen_loss / batch,
            adv=adv_loss / batch, construct=construct_loss / batch,
            latent=latent_loss / batch
        )

    @tf.function
    def _validate_step(self, gt_imgs: tf.Tensor):
        fake_imgs = self._gen(gt_imgs, training=False)
        fake_latent, fake_pred = self._disc(fake_imgs, training=False)
        true_latent, _ = self._disc(gt_imgs, training=False)

        construct_loss = self._construct_loss(gt_imgs, fake_imgs)
        latent_loss = self._latent_loss(true_latent, fake_latent)
        return construct_loss, latent_loss
