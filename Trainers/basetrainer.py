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
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense   # noqa: E402
from tensorflow.keras.optimizers import SGD, Adam  # noqa: E402
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping  # noqa: E402
from tensorflow.keras.callbacks import ReduceLROnPlateau  # noqa: E402
from MLBOX.Scenes.SimpleSplit import SimpleSplit   # noqa: E402
from MLBOX.Trainers.TF.Keras_Callbacks import ModelLogger, TrainRecord  # noqa: E402
# from MLBOX.Trainers.TF.Keras_Callbacks import LearningRateDecaySchedule  # noqa: E402


class KerasBaseTrainner:

    def __init__(self, model, loss, optimizer, out_dir, metrics=None):
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
            train_db,  # should already config parser
            vali_db,  # should already config parser
            lr_decay_factor=0.5,
            batch_size=8,
            min_epoch=40,
            max_epoch=200,
            early_stop_patience=20,
            load_best=True
            ):

        init_epoch = 0
        if load_best:
            weights = list(pathlib.Path(self.out_dir).glob("*.h5"))
            if weights:
                weights = weights[0]
                filename = weights.name
                ini_epoch = int(filename.split("_")[1])
                self._model.load_weights(str(weights))
                print("load pretrain weights from {}".format(filename))
                print("Re-train from epoch: {}".format(init_epoch))

        self._model.fit(
            x=train_db.get_dataset(epoch=max_epoch, batchsize=batch_size),
            epochs=max_epoch,
            steps_per_epoch=train_db.data_count // batch_size,
            validation_data=vali_db.get_dataset(epoch=max_epoch, batchsize=batch_size),
            validation_steps=vali_db.data_count // batch_size,
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
                    patience=early_stop_patience // 2,
                    min_delta=1e-4,
                    cooldown=2,
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
            validation_freq=1
        )
