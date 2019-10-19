from abc import abstractmethod
import imghdr
import pathlib
import yaml
from scipy.io import wavfile
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


class DataFormat(yaml.YAMLObject):
    yaml_tag = "!DataFormat"

    feature_types = {
        tf.io.FixedLenFeature,
        tf.io.FixedLenSequenceFeature,
        tf.io.VarLenFeature,
    }

    valid_extensions = set()
    features = {}

    def __getitem__(self, key):
        return self.features[key]

    def __iter__(self):
        return iter(self.features)

    def __len__(self):
        return len(self.features)

    def __eq__(self, other):
        if not isinstance(other, DataFormat):
            return False
        return self.features == other.features

    def update(self, dct):
        for key, item in dct.items():

            if not isinstance(key, str):
                key = str(key)

            if type(item) not in self.feature_types:
                msg = "Item must be one of {}, got {}"
                raise ValueError(
                    msg.format(self.feature_types, type(item))
                    )

            self.features[key] = item

    @classmethod
    def to_yaml(cls, dumper, fmt):
        features_repr = {k: str(v) for k, v in fmt.features.items()}
        return dumper.represent_mapping(cls.yaml_tag, features_repr)

    @classmethod
    def from_yaml(cls, loader, node):
        features_repr = loader.construct_mapping(node)
        features = {k.value: eval("tf.io." + v.value) for k, v in node.value}
        return cls(features)

    def to_tfexample(self, file) -> tf.train.Example:
        """Convert data to tf.train.Example for writing database

        Args:
            file: a string specify file path of image

        Returns:
            tf.train.Example
        """
        raise NotImplementedError()

    def get_parser(self):
        """Get a function which parses tf.train.Example into tensors for training

        Return:
           func(example: tf.train.Example)
           -> tuple of tensors (Image Data, Image ID, Image Label)
        """
        raise NotImplementedError()


class IMGFORMAT(DataFormat):
    """Format for classification on image"""

    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'extension': tf.io.FixedLenFeature([], tf.string),
        'encoded': tf.io.FixedLenFeature([], tf.string),
        'class':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    valid_extensions = {'.jpg', '.bmp'}

    def __init__(self, img_label_map=None):
        """
        Args:
            img_label_map:
                a dictionary that maps img_id into int or a list of int.
                which specify the label(s) of the image.
                If set to None,
                no class label is included in the tf.train.Example
        """
        self._label_map = None if not img_label_map else img_label_map.copy()

    @staticmethod
    def load_from_file(file: str):
        """Load image from file and return image id and content in bytes

        Args:
            img_file: a string specify file path of image

        Return:
            a tuple of bytes (image id, file extension, content in bytes)

        Raise:
            OSError: when file not exist
            TypeError: when file exist but not recognized as image
        """
        path = pathlib.Path(file)
        if not path.is_file():
            raise OSError("Invalid file path")

        image_id = path.stem
        image_type = imghdr.what(str(path))
        if not image_type:
            raise TypeError("Unrecognized image type")

        with open(str(file), 'rb') as f:
            image_bytes = f.read()

        return (
            bytes(image_id, 'utf8'),
            bytes(image_type, 'utf8'),
            image_bytes,
            )

    def to_tfexample(self, img_file_path):
        """Load image and return a tf.train.Example object

        Args:
            img_file_path:
                a string specify the path of image
        """
        img_id, img_type, img_bytes = IMGFORMAT.load_from_file(img_file_path)

        fields = {
            'filename': _tffeature_bytes(img_id),
            'extension': _tffeature_bytes(img_type),
            'encoded': _tffeature_bytes(img_bytes),
            }

        if self._label_map:
            labels = self._label_map[img_id.decode('utf8')]
            labels = [labels] if isinstance(labels, int) else \
                [int(lb) for lb in labels]

            fields['class'] = _tffeature_int64(labels)

        return tf.train.Example(features=tf.train.Features(feature=fields))

    @staticmethod
    def get_parser(num_of_class, **kwargs):
        """Construct image classification tfexample parser

        Args:
            num_of_classe:
                Specify the number of classes to classify,
                the output label will be one-hoted into this size.
            **kwargs:
                NOT USED,
                only for igorning possible arguments passed into get parser.

        Returns:
            A parser function which takes tfexample as input and return tensors
            Note that label is one-hoted, i.e. tensor of shape (num_of_class, )
        """
        if num_of_class <= 0:
            raise ValueError("Number of class must > 0")
        num_of_class = int(num_of_class)

        def parse_tfexample(example):
            parsed_feature = tf.io.parse_single_example(
                example,
                features=IMGFORMAT.features
            )

            img_name = parsed_feature['filename']
            label = parsed_feature['class']
            label = tf.one_hot(label, num_of_class)
            label = tf.reduce_sum(label, axis=0)

            # parse image and set shape, for more info about shape:
            # https://github.com/tensorflow/tensorflow/issues/8551
            img = tf.image.decode_image(
                parsed_feature['encoded'],
                channels=3
                )
            img.set_shape([None, None, 3])
            img = tf.cast(img, tf.float32)

            return img, img_name, label

        return parse_tfexample


class TSFORMAT(DataFormat):
    """Format for classification on time series"""

    features = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'extension': tf.io.FixedLenFeature([], tf.string),
        'encoded':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        'length': tf.io.FixedLenFeature([], tf.int64),
        'rate': tf.io.FixedLenFeature([], tf.int64),
        'class':
            tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    valid_extensions = {'.wav'}

    def __init__(self, ts_label_map=None):
        """
        Args:
            ts_label_map:
                a dict that maps time series id into int or a list of int.
                which specify the label(s) of the time series
                If set to None,
                no class label is included in the tf.train.Example
        """
        self._label_map = None if not ts_label_map else ts_label_map.copy()

    def to_tfexample(self, ts_file_path):
        file = pathlib.Path(file)
        if not file.is_file():
            raise ValueError("Invalid file path")

        if file.suffix != ".wav":
            raise ValueError("Invalid extension")

        dataid = file.stem
        extension = file.suffix.strip(".")
        rate, array = wavfile.read(str(file))

        fields = {
            'filename': _tffeature_bytes(bytes(dataid, 'utf8')),
            'extension': _tffeature_bytes(bytes(extension, 'utf')),
            'encoded': _tffeature_int64(array),
            'length': _tffeature_int64(len(array)),
            'rate': _tffeature_int64(rate),
        }

        if self._label_map:
            labels = self._label_map[str(dataid)]
            labels = [labels] if isinstance(labels, int) else \
                [int(lb) for lb in labels]

            fields['class'] = _tffeature_int64(labels)

        return tf.train.Example(features=tf.train.Features(feature=fields))

    @staticmethod
    def get_parser():
        raise NotImplementedError()
