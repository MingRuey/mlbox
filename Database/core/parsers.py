from typing import Dict, List

import tensorflow as tf

from MLBOX.Database.core.features import Feature
from MLBOX.Database.core.features import ImageFeature


class ParserFMT:

    @property
    def features(self) -> List[Feature]:
        """List of Feature that the format contains"""
        raise NotImplementedError()

    def parse_example(self, example) -> Dict[str, tf.Tensor]:
        """Convert tf.train.Example to dict of tf.tensor"""
        encoded = {}
        for feat in self.features:
            encoded.update(feat.encoded_features)

        example = tf.io.parse_single_example(
            example.SerializeToString(), features=encoded
        )

        outputs = {}
        for feat in self.features:
            outputs.update(feat._parse_from(
                **{key: example[key] for key in feat.encoded_features}
            ))
        return outputs

    def to_example(self, **kwargs) -> tf.train.Example:
        """Convert data to tf.train.Example for writing database

        Args:
            **kwrags: {feature: value} map defined by Format.features

        Returns:
            tf.train.Example
        """
        outputs = {}
        for feat in self.features:
            outputs.update(feat._create_from(
                **{key: kwargs[key] for key in feat.create_keys}
            ))

        example = tf.train.Example(
            features=tf.train.Features(feature=outputs)
        )
        return example


class ImageParser(ParserFMT):

    def __init__(self):
        self._img = ImageFeature()

    @property
    def features(self):
        return [self._img]
