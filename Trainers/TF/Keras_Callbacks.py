import os
import logging

import yaml
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import LearningRateScheduler


MODLE_NAME = "model_{:03d}_{:.5f}.h5"


class TrainRecord:

    def __init__(self):
        self._logs = {}

        self.optimizer_settings = {}
        self.train_loss = 0.0
        self.validation_loss = 0.0
        self.best_model_file = ""

    @property
    def logs(self):
        """Return logs conforms to Keras callback convention"""
        return self._logs.copy()

    def update_log(self, logs: dict):
        """Update the training information via KerasCallback logs"""
        if logs:
            for key, val in logs.items():
                record = self._logs.setdefault(key, [])
                record.append(float(val))

    def update_record(self, train_loss, vali_loss, best_model):
        self.train_loss = float(train_loss)
        self.validation_loss = float(vali_loss)
        self.best_model_file = str(best_model)

    def dump_record(self, output_file):
        """Dump optimizer and training logs into output_file in yaml format"""
        info = {
            "best_model": {
                "best_model_file": self.best_model_file,
                "training_loss": self.train_loss,
                "validation_loss": self.validation_loss,
            },
            "optimizer_setting": self.optimizer_settings,
            "logs": self._logs
        }

        with open(str(output_file), "w") as f:
            yaml.safe_dump(info, f, default_flow_style=False)


class ModelLogger(Callback):
    """Save the model without optimize states after every epoch.

    Args:
        train_record:
            the TrainingRecord of ModeSpec, the record will be updated inplace.
        temp_model_folder:
            folder to store model without best performance.
        best_model_folder:
            folder to store best model.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model will be save to `best_model_folder`,
            and last best model will be move to `temp_model_folder`.
            if `save_best_only=False`,
            the operation of best model is same.
            but if model didn't improve from latest best quantity of monitor,
            model will be save to `temp_model_folder`.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
        **kwargs:
            NOT USED.
            For caller can add additional kwargs without getting errors.
    """

    def __init__(
            self,
            train_record: TrainRecord,
            temp_model_folder, best_model_folder,
            monitor='val_loss',
            verbose=0,
            save_best_only: bool = False,
            save_weights_only: bool = False,
            mode='auto',
            period=1,
            **kwargs
            ):
        super(ModelLogger, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.temp_dir = temp_model_folder
        self.best_dir = best_model_folder
        self.best_model_name = ""
        self.record = train_record

        if mode not in ['auto', 'min', 'max']:
            msg = 'ModelCheckpoint mode {} is unknown, fallback to auto mode.'
            self._logging(msg.format(mode))
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def _logging(self, msg):
        """Helper for logging message depends on the verbose setting"""
        if self.verbose > 0:
            logging.info(msg)

    def _save_model(self, temp_or_best: str, model_name: str, message=""):
        """Helper for saving model to temp folder """
        if temp_or_best == "temp":
            folder = self.temp_dir
        else:
            folder = self.best_dir

        file_path = os.path.join(folder, model_name)
        if self.save_weights_only:
            self.model.save_weights(file_path, overwrite=True)
        else:
            save_model(self.model, file_path, include_optimizer=False)

        self._logging("saving model to {}".format(file_path))

    def on_epoch_end(self, epoch, logs=None):
        if (self.model.stop_training and epoch == 0):
            return

        logs = logs or {}
        trainloss = logs.get('loss', np.inf)
        validloss = logs.get('val_loss', np.inf)

        now_epoch = epoch + 1
        now_model_name = MODLE_NAME.format(now_epoch, validloss)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            current = logs.get(self.monitor)
            if current is None:
                if self.save_best_only:
                    msg = "Save best model only with {} available, skipping."
                    self._logging(msg.format(self.monitor))
                else:
                    self._save_model(
                        temp_or_best="temp",
                        model_name=now_model_name
                    )
            else:
                if self.monitor_op(current, self.best):
                    msg = "Epoch {}: {} improved from {} to {}."
                    self._logging(msg.format(
                        epoch + 1, self.monitor, self.best, current,
                    ))
                    if self.best_model_name is not "":
                        if self.save_best_only:
                            os.remove(os.path.join(self.best_dir, self.best_model_name))
                        else:
                            os.rename(
                                os.path.join(self.best_dir, self.best_model_name),
                                os.path.join(self.temp_dir, self.best_model_name)
                            )
                    self.best = current
                    self.best_model_name = now_model_name
                    self.record.update_record(
                        train_loss=trainloss, vali_loss=validloss,
                        best_model=now_model_name
                    )
                    self._save_model(
                        temp_or_best="best",
                        model_name=now_model_name,
                        message=msg
                    )
                else:
                    msg = "Epoch {}: {} did not improve from {}"
                    self._logging(
                        msg.format(epoch + 1, self.monitor, self.best)
                    )
                    if not self.save_best_only:
                        self._save_model(
                            temp_or_best="temp",
                            model_name=now_model_name
                        )

        self.record.update_log(logs)
        self.record.dump_record(os.path.join(self.best_dir, "logs.yml"))


class EarlyStopper(Callback):
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        wake_up: number of epochs to wake up EarlyStopper.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 wake_up=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopper, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.wake_up = wake_up

        if mode not in ['auto', 'min', 'max']:
            msg = "EarlyStopping mode {} is unknown, fallback to auto mode."
            logging.info(msg.format(mode))
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 > self.wake_up:
            logging.info('Early stopping mode on')
            current = self.get_monitor_value(logs)
            if current is None:
                return

            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                    if self.restore_best_weights:
                        if self.verbose > 0:
                            msg = "Restore model weights from the end of " + \
                                "the best epoch"
                            logging.info(msg)
                        self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            msg = "Epoch {}: early stopping"
            logging.log(self.stopped_epoch + 1)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            msg = "Early stopping conditioned on metric `{}` " + \
                "which is not available. Available metrics are: %s"
            logging.info(
                msg.format(
                    self.monitor,
                    ','.join(list(logs.keys()))
                ))
        return monitor_value


class TimeMeasure(Callback):
    def __init__(self):
        super(TimeMeasure, self).__init__()
        self.time = 0.0
        self.total_time = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self.total_time = 0.0

    def on_batch_begin(self, batch, logs=None):
        self.time = time.time()

    def on_batch_end(self, batch, logs=None):
        logging.info(" - time: {:.3f}s".format(time.time()-self.time), end="")
        self.total_time += time.time()-self.time

    def on_epoch_end(self, epoch, logs=None):
        logging.info(" - epoch time: {:.3f}s".format(self.total_time), end="")


class LearningRateDecaySchedule:
    """options of learning rate decay policy
    """
    @staticmethod
    def step_decay_by_epoch(decay=0.1, epochs_to_decay=2):
        def schedule(epoch, lr):
            if epoch > 1 and epoch % epochs_to_decay == 0:
                lr = lr * decay
            return lr
        return LearningRateScheduler(schedule, verbose=1)
