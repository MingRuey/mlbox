import inspect
import imghdr
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Set, Dict, Union, Tuple

import tensorflow as tf
import numpy as np
import cv2


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


class BoundingBox(Feature):
    """BoundingBoxes feature in COCO-like format, but inverse x-y"""

    encoded_features = {
        "bbox_ymins": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "bbox_xmins": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "bbox_heights": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "bbox_widths": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),
        "bbox_classes": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        )
    }

    def __init__(
            self,
            n_class: int,
            max_bbox_per_data: int,
            ):
        """
        Args:
            n_class (int): total number of classes
            max_bbox_per_data (int):
                Number of bxoes per data instance.
                Note that this settings only affects parsing,
                number of encoded bounding boxes can still exceeds this number.
        """
        self.n_class = int(n_class)
        self.n_box = int(max_bbox_per_data)

        pad_constant = tf.constant([0, 0, 0, 0, -1], dtype=tf.float32)
        pad_constant = tf.expand_dims(pad_constant, axis=-1)
        self._pad_constant = pad_constant

    def _parse_from(
            self,
            bbox_ymins, bbox_xmins,
            bbox_heights, bbox_widths, bbox_classes
            ):
        stack = tf.stack(
            [bbox_ymins, bbox_xmins, bbox_heights, bbox_widths, bbox_classes],
            axis=0, name="bbox_parser_stack"
        )
        nEncoded = tf.shape(stack)[1]

        def pad(n_box=self.n_box):
            multiples = tf.stack([1, (self.n_box - nEncoded)], axis=0)
            padded = tf.tile(self._pad_constant, multiples=multiples)
            return tf.concat([stack, padded], axis=1)

        def crop(n_box=self.n_box):
            return stack[:, :self.n_box]

        crop_or_pad = tf.case(
            [
                (tf.less(nEncoded, self.n_box), pad),
                (tf.greater(nEncoded, self.n_box), crop)
            ],
            default=lambda: stack,
            name="bbox_parse_padding_or_crop"
        )
        bboxes = tf.transpose(crop_or_pad, name="bbox_parser_transpose")
        return {"boxes": bboxes}

    def _create_from(self, boxes):
        """Create tffeatures for boxing boxes

        Args:
            boxes (List of Tuple(float, float, float, float, int)):
                List of bounding boxes in [ymin, xmin, height, width, class]
                (COCO-like but in the order of x-y is reversed)
        """
        ymins = []
        xmins = []
        heights = []
        widths = []
        classes = []
        for bbox in boxes:
            if len(bbox) != 5:
                msg = "Invalid box format, should be (y, x, h, w, class), got {}"
                raise ValueError(msg.format(bbox))

            ymin, xmin, h, w, cls_index = bbox
            if ymin < 0 or xmin < 0:
                msg = "Invalid box (ymin, xmin) labels must >= 0, get {}"
                raise ValueError(msg.format((ymin, xmin)))
            if h <= 0 or w <= 0:
                msg = "Invalid box (height, width), must > 0, get {}"
                raise ValueError(msg.format((h, w)))
            if cls_index > self.n_class - 1:
                msg = "Invalid class index: {} exceeds defined n_class {}"
                raise ValueError(msg.format(cls_index, self.n_class))

            ymins.append(ymin)
            xmins.append(xmin)
            heights.append(h)
            widths.append(w)
            classes.append(cls_index)

        features = {
            "bbox_ymins": _tffeature_float(ymins),
            "bbox_xmins": _tffeature_float(xmins),
            "bbox_heights": _tffeature_float(heights),
            "bbox_widths": _tffeature_float(widths),
            "bbox_classes": _tffeature_float(classes)
        }
        return features


class RLE:

    def __init__(self, bits: List[int], cls_idx: int):
        """Create RLE type segmentaion label

        Args:
            bits (List[int]): the uncompressed RLE format bits
            cls_idx (int): the class id of the segmentation
        """
        self._bits = list(bits)
        self._cls_idx = int(cls_idx)
        if int(cls_idx) == 0:
            msg = "class index of segmentation must >= 1"
            raise ValueError(msg)

    @property
    def bits(self) -> List[int]:
        return list(self._bits)

    @property
    def class_id(self) -> int:
        return self._cls_idx


class PolyGon:

    def __init__(self, pts: List[Tuple[int, int]], cls_idx: int):
        """Create PolyGon type segmnetaion label

        Args:
            pts (List[Tuple[int, int]]):
                A list of points (y, x) defining the PolyGon
            cls_idx (int):
                class id of the polygon
        """
        self._pts = list(pts)
        if int(cls_idx) == 0:
            msg = "class index of segmentation must >= 1"
            raise ValueError(msg)
        self._cls_idx = int(cls_idx)

    @property
    def points(self) -> List[Tuple[int, int]]:
        """A list of points (y, x) defining the PolyGon"""
        return list(self._pts)

    @property
    def class_id(self) -> int:
        return self._cls_idx


class Segmentation(Feature):

    encoded_features = {
        "segment_mask": tf.io.FixedLenFeature([], tf.string)
    }

    @staticmethod
    def _encode_image(array: np.ndarray):
        success, encoded = cv2.imencode(".bmp", array)
        return {"segment_mask": _tffeature_bytes(encoded.tobytes())}

    @staticmethod
    def _check_id(class_id: int):
        if class_id <= 0 or class_id > 255:
            msg = "Invalid segmentation class id: {}. {}"
            reason = "Segment mask use uint8 and 0 is background."
            raise ValueError(msg.format(class_id, reason))

    def _create_from_rle(
            self, rles: List[RLE], mask_shape: Tuple[int, int]
            ):
        img = np.zeros(mask_shape[0] * mask_shape[1], dtype=np.uint8)
        for rle in rles:
            self._check_id(rle.class_id)
            if not sum(rle.bits) == img.size:
                raise ValueError("invalid RLE bits for shape {}")

            val = 0
            idx = 0
            for length in rle.bits:
                if not val:
                    val = rle.class_id
                else:
                    img[idx:idx+length] = rle.class_id
                    val = 0
                idx += length
        img = img.reshape(mask_shape)
        return img

    def _create_from_polygons(
            self, polygons: List[PolyGon], mask_shape: Tuple[int, int]
            ):
        img = np.zeros(mask_shape, dtype=np.uint8)
        for poly in polygons:
            self._check_id(poly.class_id)

            pts = np.array(poly.points)
            pts = pts.astype(np.int)
            # cv2 points use (x, y) format instead of (y, x)
            pts = pts[:, ::-1]
            cv2.fillConvexPoly(img, pts, poly.class_id)
        return img

    def _create_from(
            self,
            mask_shape: Tuple[int, int],
            poly_or_rle: List[Union[PolyGon, RLE]]
            ):
        """Segmentaion label can be either 'polygon' or 'rle' format

        Note:
            if a pixel is labeled more than once,
            rle is usually prefered (since it's more accurate than polygon).
            but the behaviour is not guaranteed.

        Args:
            mask_shape: the image shape in (h, w)
            poly_or_rle:
                Segmentations specify by PolyGon or RLE.
        """
        mask_shape = mask_shape[:2]  # safeguard against (h, w, c) input
        polygons = [item for item in poly_or_rle if isinstance(item, PolyGon)]
        rles = [item for item in poly_or_rle if isinstance(item, RLE)]

        poly_img = self._create_from_polygons(polygons, mask_shape)
        rle_img = self._create_from_rle(rles, mask_shape)
        mask_img = np.where(rle_img, rle_img, poly_img)
        return self._encode_image(mask_img)

    def _parse_from(self, segment_mask):
        mask = tf.image.decode_image(
            segment_mask, channels=0, expand_animations=False
        )
        mask = mask[..., 0]
        return {"segment_mask": mask}


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
            channels: int = 0,
            method: str = "bilinear"
            ):
        """Create ImageFeature with resize options

        Args:
            resize_shape:
                a tuple of (height, width) specifying
                uniform resize applied to images
            channels:
                decoded image channels, 0 for auto-detecting.
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
        self._channels = channels
        self._method = str(method).lower()

    # Implicitly assuming keys appear in ImageFeature.encoded_features
    def _parse_from(self, image_id, image_type, image_content):
        img = tf.image.decode_image(
            image_content, channels=self._channels,
            expand_animations=False
        )
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
