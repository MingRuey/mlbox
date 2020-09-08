from typing import Dict, Tuple
import tensorflow as tf

from ..core.parsers import ParserFMT, Feature
from ..core.features import ImageFeature, IntLabel, BoundingBox, Segmentation


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
        ImageFeature(resize_shape=(256, 256), channels=3),
        IntLabel(1000)
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


class CoCoObjDet(ParserFMT):

    def __init__(
            self,
            resize_shape: Tuple[int, int] = None,
            channels: int = 0,
            method: str = "bilinear",
            one_hot_class: bool = True,
            max_bbox_per_data: int = 93,
            ):
        """Create CoCo-Detection parser with resize options

        Args:
            resize_shape, channels, method:
                please refer to ImageFeature documentation
            max_bbox_per_data (int):
                please refer to BoundingBox documentation.
                Note:
                    in CoCo dataset,
                    image with most bounding boxes has 93 box annotations.
            one_hot_class (bool):
                whether to one-hoted class index of bounding box
        """
        self.nClass = 80
        self._bbox_feat = BoundingBox(
            n_class=self.nClass,
            max_bbox_per_data=max_bbox_per_data
        )

        self._shp = resize_shape
        self._method = str(method).lower()
        self._seg_feat = Segmentation()
        self._img_feat = ImageFeature(
            # turn-off resizing to get original image size
            # use that for resizing bounding boxes
            resize_shape=None,
            channels=channels,
            method=method
        )
        self._one_hot_class = bool(one_hot_class)

    @property
    def features(self):
        return [self._img_feat, self._bbox_feat]

    def parse_example(self, example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = super().parse_example(example)
        example.pop("image_id")
        example.pop("image_type")
        image = example.pop("image_content")
        boxes = example.pop("boxes")

        if self._shp is not None:
            old_shp = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
            new_shp = tf.cast(self._shp, dtype=tf.float32)
            image = tf.image.resize(
                image, self._shp, self._method, antialias=True
            )

            # resize bounding box
            y, x, h, w = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            y = new_shp[0] * y / old_shp[0]
            x = new_shp[1] * x / old_shp[1]
            h = new_shp[0] * h / old_shp[0]
            w = new_shp[1] * w / old_shp[1]
            coordinates = tf.stack([y, x, h, w], axis=1)
        else:
            coordinates = boxes[:, :4]

        # one-hot classes
        classes = boxes[:, 4]
        if self._one_hot_class:
            classes = tf.cast(classes, dtype=tf.int32)
            classes = tf.one_hot(classes, depth=self.nClass)
        else:
            classes = classes[..., tf.newaxis]

        boxes = tf.concat([coordinates, classes], axis=1)
        example["image"] = image
        example["boxes"] = boxes
        return example


class CoCoSeg(ParserFMT):

    def __init__(
            self,
            resize_shape: Tuple[int, int] = None,
            channels: int = 0,
            method: str = "bilinear",
            one_hot_class: bool = True
            ):
        """Create CoCo-Segmentation parser with resize options

        Args:
            resize_shape, channels, method:
                please refer to ImageFeature documentation
            one_hot_class (bool):
                whether to one-hoted class index of bounding box
        """
        self.nClass = 80
        self._shp = resize_shape
        self._seg_feat = Segmentation()
        self._img_feat = ImageFeature(
            resize_shape=resize_shape,
            channels=channels,
            method=method
        )
        self._one_hot_class = bool(one_hot_class)

    @property
    def features(self):
        return [self._img_feat, self._seg_feat]

    def parse_example(self, example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = super().parse_example(example)
        example.pop("image_id")
        example.pop("image_type")
        image = example.pop("image_content")
        seg = example.pop("segment_mask")

        if self._shp is not None:
            seg = seg[..., tf.newaxis]
            seg = tf.image.resize(
                seg, size=self._shp,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                antialias=False
            )
            seg = seg[..., 0]

        # one-hot classes
        if self._one_hot_class:
            seg = tf.one_hot(seg, depth=self.nClass)
        else:
            seg = seg[..., tf.newaxis]

        example["image"] = image
        example["segment_mask"] = seg
        return example
