import inspect
import imghdr
from pathlib import Path
from typing import Set, Dict, Union

import tensorflow as tf


def _tffeature_int64(value):
    value = [value] if isinstance(value, int) else value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _tffeature_float(value):
    value = [value] if isinstance(value, float) else value
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _tffeature_bytes(value):
    if isinstance(value, str):
        value = value.encode()
    value = [value] if isinstance(value, bytes) else value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class Feature:

    @property
    def name(self):
        """Get the name of features"""
        return self.__class__.__name__

    @property
    def encoded_features(self) -> Dict[str, Union[tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature]]:
        """Get features encoded into tf.train.Example

        Keys should match the outputs of _create_from and inputs of _parse_from
        """
        raise NotImplementedError()

    @property
    def create_keys(self) -> Set[str]:
        """Get the variables needed for creating tf.train.Example"""
        return set(inspect.signature(self._create_from).parameters.keys())

    def _parse_from(self,  **kwrags) -> Dict[str, tf.Tensor]:
        """Subclass implement the parser function for decoding tf.train.example

        Args:
            **kwargs:
                the features for parsing,
                should match keys of encoded_features
        """
        raise NotImplementedError()

    def _create_from(self, **kwargs) -> Dict[str, tf.train.Feature]:
        """Subclass should implement this to specify whats are needed to
        create this feature, and how to convert data to tf.train.Feature

        Args:
            **kwargs: the data needed for creating the feature

        Returns:
            a tf.train.Feature instance created from input data.
            the returned features should match encoded_features
        """
        raise NotImplementedError()


class ImageFeature(Feature):

    encoded_features = {
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'image_type': tf.io.FixedLenFeature([], tf.string),
        'image_content': tf.io.FixedLenFeature([], tf.string)
    }

    # Implicitly assuming keys appear in ImageFeature.features
    def _parse_from(self, image_id, image_type, image_content):
        img = tf.image.decode_image(image_content, channels=0)
        img = tf.cast(img, tf.float32)
        return {
            'image_id': image_id,
            'image_type': image_type,
            'image_content': img
        }

    def _create_from(self, image: str):
        """Build image feature from file"""
        image = Path(image)
        if not image.is_file():
            raise ValueError("Invalid image path: {}".format(image))

        image_id = image.stem
        image_type = imghdr.what(str(image))
        if not image_type:
            raise ValueError("Unrecognized image type: {}".format(image))

        with open(str(image), 'rb') as f:
            image_bytes = f.read()

        features = {
            'image_id': _tffeature_bytes(image_id),
            'image_type': _tffeature_bytes(image_type),
            'image_content': _tffeature_bytes(image_bytes),
        }
        return features
