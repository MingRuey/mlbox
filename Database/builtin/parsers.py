from typing import Dict
import tensorflow as tf

from MLBOX.Database.core.parsers import ParserFMT, Feature
from MLBOX.Database.core.features import ImageFeature, IntLabel


class _ClassficationFMT(ParserFMT):

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse_example(self, example):
        example = tf.io.parse_single_example(
            example, features=self.features
        )
        example["image"] = tf.image.decode_image(example["image"])
        return example


class MNIST(_ClassficationFMT):
    pass


class CIFAR10(_ClassficationFMT):
    pass


class CIFAR100(CIFAR10):

    features = CIFAR10.features.copy()
    features["coarse_label"] = tf.io.FixedLenFeature([], tf.int64)


class IMAGENET(ParserFMT):

    features = [
        ImageFeature(resize_shape=(256, 256)), IntLabel(1000)
    ]

    def parse_example(self, example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = super().parse_example(example)
        example.pop("image_id")
        example.pop("image_type")
        image = example.pop("image_content")
        label = example.pop("classes")
        example["image"] = image
        example["label"] = label
        return example
