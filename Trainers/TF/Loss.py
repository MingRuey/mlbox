import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.losses import Loss


class PerceptualLoss(Loss):

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        pass


class SSIMLoss(Loss):

    def __init__(
            self, max_val, k1=0.01, k2=0.03,
            filter_size=11, filter_sigma=1.5
            ):
        """Init SSIM loss with ssim parameters

        Args:
            same as tf.image.ssim
        """
        self._params = {
            "max_val": max_val, "k1": k1, "k2": k2,
            "filter_size": filter_size, "filter_sigma": filter_sigma
        }

    def __call__(self, y_true, y_pred):
        return tf.reduce_mean(
            1.0 - tf.image.ssim(y_true, y_pred, **self._params)
        )
