from abc import abstractmethod, ABC
from typing import Dict, List
import inspect

import tensorflow as tf

from .features import Feature
from .features import ImageFeature


class ParserFMT(ABC):

    @property
    def info(self) -> str:
        feat_info = "{} - create keys: {}; encoded_features: {}"
        outputs = "\n".join(
            feat_info.format(
                feat.name,
                set(inspect.signature(feat._create_from).parameters.keys()),
                feat.encoded_features.keys()
            ) for feat in self.features
        )
        return outputs

    @property
    @abstractmethod
    def features(self) -> List[Feature]:
        """List of Feature that the format contains"""
        raise NotImplementedError()

    def parse_example(self, example: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Convert tf.train.Example to dict of tf.tensor"""
        encoded = {}
        for feat in self.features:
            encoded.update(feat.encoded_features)

        example = tf.io.parse_single_example(example, features=encoded)

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
            feat_signature = inspect.signature(feat._create_from)
            feat_inputs = {}
            for key, param in feat_signature.parameters.items():
                val = kwargs.get(key, inspect.Parameter.empty)
                if val == inspect.Parameter.empty:
                    if param.default != inspect.Parameter.empty:
                        val = param.default
                    else:
                        msg = "Missing data field {} of feature: {}"
                        raise ValueError(msg.format(feat.name, key))
                feat_inputs[key] = val
            outputs.update(feat._create_from(**feat_inputs))

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
