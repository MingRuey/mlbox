import tensorflow as tf

from MLBOX.Database.core.parsers import ParserFMT
from MLBOX.Database.core.features import ImageFeature


class MNIST(ParserFMT):

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64)
    }

    def parse_example(self, example):
        example = tf.io.parse_single_example(
            example, features=self.features
        )
        image = tf.image.decode_image(example["image"])
        return {"image": image, "label": example["label"]}
