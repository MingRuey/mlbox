import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.metrics import Metric


class SSIM(Metric):
    """The mean structural similarity index between two images

    Note: it assumes the in
    """

    def __init__(self, name="SSIM", **kwargs):
        super().__init__(name=name, **kwargs)
        self.ssim = self.add_weight(name="ssim", initializer="zeros")
        self.batch = self.add_weight(name="batch", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true, y_pred, max_val=1)
        batch_dim = tf.shape(y_true)[0]
        batch_dim = tf.cast(batch_dim, tf.float32)
        self.ssim.assign_add(tf.reduce_sum(ssim))
        self.batch.assign_add(batch_dim)

    def result(self):
        return self.ssim / self.batch
