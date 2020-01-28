import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, Lambda

from MLBOX.Models.TF.Keras.wrappers.layer_wrappers import ConstraintWrapper
from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator


class WeightsClip(Constraint):

    def __init__(self, min_value, max_value):
        if min_value > max_value:
            msg = "min_value {} larger than max_value"
            raise ValueError(msg.format(min_value, max_value))
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {
            'min_value': self.min_value,
            'max_value': self.max_value
        }


class WasserteinLoss:

    def __call__(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


class WGenerator(Generator):

    @staticmethod
    def rescale(x: tf.Tensor) -> tf.Tensor:
        return 0.5 * (x + 1.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        raw_ouput = super().__call__(inputs)
        output = Activation("tanh")(raw_ouput)
        output = Lambda(WGenerator.rescale)(output)
        return output


class WDiscriminator(Discriminator):
    """Basically a DCGAN except wrapping weight-clipping constraint"""

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        wrapper = ConstraintWrapper(
            constraint=WeightsClip(-0.01, 0.01),
            targets={Conv2D, Dense, BatchNormalization}
        )
        wrapped_call = wrapper(super().__call__)
        return wrapped_call(inputs)


if __name__ == "__main__":
    WDiscriminator()
    Discriminator()
