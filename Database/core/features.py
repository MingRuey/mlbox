import inspect
import imghdr
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Set, Dict, Union, Tuple

import tensorflow as tf


def _tffeature_int64(value):
    value = [value] if isinstance(value, int) else value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _tffeature_float(value):
    value = [value] if isinstance(value, float) else value
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _tffeature_bytes(value):
    if isinstance(value, str):
        value = [value.encode()]
    elif isinstance(value, list):
        value = [val.encode() for val in value if isinstance(val, str)]
    elif isinstance(value, bytes):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class Feature(ABC):

    @property
    def name(self):
        """Get the name of features"""
        return self.__class__.__name__

    @property
    @abstractmethod
    def encoded_features(self) -> Dict[str, Union[tf.io.FixedLenFeature, tf.io.FixedLenSequenceFeature]]:
        """Get features encoded into tf.train.Example

        Keys should match the outputs of _create_from and inputs of _parse_from
        """
        raise NotImplementedError()

    @property
    def create_keys(self) -> Set[str]:
        """Get the variables needed for creating tf.train.Example"""
        return set(inspect.signature(self._create_from).parameters.keys())

    @abstractmethod
    def _parse_from(self,  **kwrags) -> Dict[str, tf.Tensor]:
        """Subclass implement the parser function for decoding tf.train.example

        Args:
            **kwargs:
                the features for parsing,
                should match keys of encoded_features
        """
        raise NotImplementedError()

    @abstractmethod
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


class _SimpleLabel(Feature):
    """Simple feature for creating homogenious dtype label"""

    def __init__(self, dtype):
        """
        Args:
            dtype:
                the type of label,
                must be one of tf.string, tf.int64, tf.float32
        """
        allowed = [tf.string, tf.int64, tf.float32]
        if dtype not in allowed:
            msg = "dtype must be one of {}, got {}"
            raise ValueError(msg.format(allowed, dtype))

        self._dtype = dtype
        self._features = {
            "classes": tf.io.FixedLenSequenceFeature(
                [], dtype, allow_missing=True
            )
        }

    @property
    def encoded_features(self):
        return self._features


class IntLabel(_SimpleLabel):
    """Simple integer(s) as label"""

    def __init__(self, n_class: int):
        """
        Args:
            n_class (int): total number of classes
        """
        super().__init__(tf.int64)
        self.n_class = int(n_class)

    def _parse_from(self, classes):
        label = tf.one_hot(classes, depth=self.n_class)
        label = tf.reduce_sum(label, axis=0)
        return {"classes": label}

    def _create_from(self, label: List[int]):
        return {"classes": _tffeature_int64(label)}


class StrLabel(_SimpleLabel):
    """Simple (byte) sting(s) as label"""

    def __init__(self):
        super().__init__(tf.string)

    def _parse_from(self, classes):
        return {"classes": classes}

    def _create_from(self, label: List[str]):
        return {"classes": _tffeature_bytes(label)}


class FloatLabel(_SimpleLabel):
    """Simple float labels"""

    def __init__(self):
        super().__init__(tf.float32)

    def _parse_from(self, classes):
        return {"classes": classes}

    def _create_from(self, label: List[float]):
        return {"classes": _tffeature_float(label)}


class ImageFeature(Feature):
    """Encode image content from file"""

    encoded_features = {
        'image_id': tf.io.FixedLenFeature([], tf.string),
        'image_type': tf.io.FixedLenFeature([], tf.string),
        'image_content': tf.io.FixedLenFeature([], tf.string)
    }

    def __init__(
            self,
            resize_shape: Tuple[int, int] = None,
            method: str = "bilinear"
            ):
        """Create ImageFeature with resize options

        Args:
            resize_shape:
                a tuple of (height, width) specifying
                uniform resize applied to images
            method:
                the method of resizing,
                must be one of "bilinear", "lanczos3", "lanczos5",
                "bicubic", "gaussian", "nearest", "area", "mitchellcubic".
                If resizing is None, this is ignored.
        """
        if resize_shape is not None:
            availiable_methods = [
                getattr(tf.image.ResizeMethod, attr)
                for attr in dir(tf.image.ResizeMethod) if attr.isupper()
            ]
            if str(method).lower() not in availiable_methods:
                msg = "Unrecognized method {}, availiable ones: {}"
                raise ValueError(msg.format(method, availiable_methods))

        self._shp = resize_shape
        self._method = str(method).lower()

    # Implicitly assuming keys appear in ImageFeature.encoded_features
    def _parse_from(self, image_id, image_type, image_content):
        img = tf.image.decode_image(image_content, channels=0)
        img = tf.cast(img, tf.float32)

        if self._shp is not None:
            img = tf.image.resize(
                img, self._shp, self._method, antialias=True
            )

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
