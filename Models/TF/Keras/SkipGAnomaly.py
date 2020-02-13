import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input

from MLBOX.Models.TF.Keras.DCGAN import Generator, Discriminator
from MLBOX.Models.TF.Keras.UNet import UNET


class SkipGANomalyG(UNET):

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        return self.forward(inputs)


class SkipGANomalyD(Discriminator):
    pass


if __name__ == "__main__":
    pass
