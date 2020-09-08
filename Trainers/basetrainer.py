import os
import sys
import logging
import pathlib
import numpy as np

import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense   # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402

from ..Database.core.database import Dataset  # noqa: E402
from .TF.Callbacks import ModelLogger, TrainRecord  # noqa: E402


class KerasBaseTrainer:

    def __init__(
            self,
            model: keras.Model,
            loss: keras.losses.Loss,
            optimizer: keras.optimizers.Optimizer,
            out_dir: str,
            metrics: keras.metrics.Metric = None
            ):
        """Create a KerasBaseTrainner instance

        Args:
            model (keras.Model): the target model to be trained
            loss (keras.losses.Loss): the loss to optimizer
            optimizer (keras.optimizers.Optimizer):  the optimizer to use
            out_dir (str): the directory to store the trained model
            metrics (keras.metrics.Metric):
                The metrics to monitor. Defaults to None.
        """
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self._model = model

        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics

        if not pathlib.Path(str(out_dir)).is_dir():
            raise ValueError("Invalid output dir")
        self.out_dir = str(out_dir)
        self.tmp_dir = pathlib.Path(out_dir).joinpath("tmp")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = str(self.tmp_dir)

    def train(
            self,
            train_db: Dataset,
            vali_db: Dataset,
            validation_freq: int = 1,
            lr_decay_factor: float = 0.5,
            batch_size: int = 8,
            max_epoch: int = 200,
            early_stop_patience: int = 20,
            load_best: int = True
            ):
        """

        Args:
            train_db (Dataset): database of training set
            vali_db (Dataset): database of validation set
            batch_size (int, optional): Defaults to 8.
            max_epoch (int, optional):
                The maximum epoch for training. Defaults to 200.
            early_stop_patience (int, optional):
                The waiting epoch for early stop machanism. Defaults to 20.
            load_best (int, optional):
                To load the best model under output_dir or not.
                Defaults to True.
        """

        init_epoch = 0
        if load_best:
            weights = list(pathlib.Path(self.out_dir).glob("*.h5"))
            if weights:
                weights = weights[0]
                filename = weights.name
                init_epoch = int(filename.split("_")[1])
                self._model.load_weights(str(weights))
                print("load pretrain weights from {}".format(filename))
                print("Re-train from epoch: {}".format(init_epoch))

        self._model.fit(
            x=train_db.to_tfdataset(
                epoch=max_epoch, batch=batch_size
            ),
            epochs=max_epoch,
            steps_per_epoch=train_db.count // batch_size,
            validation_data=vali_db.to_tfdataset(
                epoch=max_epoch, batch=batch_size
            ),
            validation_steps=vali_db.count // batch_size,
            callbacks=[
                ModelLogger(
                    train_record=TrainRecord(),
                    temp_model_folder=self.tmp_dir,
                    best_model_folder=self.out_dir,
                    monitor='val_loss', verbose=1, mode='min',
                    save_weights_only=True
                    ),
                ReduceLROnPlateau(
                    factor=lr_decay_factor,
                    patience=4,
                    min_delta=0.1,
                    cooldown=0,
                    min_lr=1e-6,
                    monitor='val_loss', verbose=1, mode='min',
                    ),
                TensorBoard(
                    log_dir=self.tmp_dir
                    ),
                EarlyStopping(
                    monitor='val_loss',
                    mode="min",
                    patience=early_stop_patience,
                    verbose=1
                )
            ],
            validation_freq=validation_freq
        )
