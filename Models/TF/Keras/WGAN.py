import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.constraints import Constraint

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator


class WeightsClip(Constraint):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {
            'min_value': self.min_value,
            'max_value': self.max_value
        }


class WGenerator(Generator):
    pass


class WDiscriminator(Discriminator):
    """Basically a DCGAN except wrapping weight-clipping constraint"""

    def __init__(self):
        pass

    # @contraint_wrapper(WeightsClip(-0.01, 0.01))
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        pass


if __name__ == "__main__":
    WDiscriminator()
    Discriminator()
