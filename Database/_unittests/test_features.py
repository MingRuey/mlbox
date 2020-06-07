import os
import math
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import pytest  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E40

from MLBOX.Database.core.features import IntLabel, StrLabel, FloatLabel  # noqa: E402
from MLBOX.Database.core.features import ImageFeature, BoundingBox  # noqa: E402
from MLBOX.Database._unittests.configs import SAMPLE_FILES_DIR  # noqa: E402


def encode_decode(feat, **kwargs):
    encoded = feat._create_from(**kwargs)
    example = tf.train.Example(
        features=tf.train.Features(feature=encoded)
    )
    parsed = tf.io.parse_single_example(
        example.SerializeToString(), features=feat.encoded_features
    )
    parsed = feat._parse_from(**parsed)
    return encoded, parsed


class TestLabelFeature:

    @staticmethod
    def _encode_and_parse(feat, inputs):
        encoded = feat._create_from(**inputs)
        example = tf.train.Example(
            features=tf.train.Features(feature=encoded)
        )
        parsed = tf.io.parse_single_example(
            example.SerializeToString(), features=feat.encoded_features
        )
        parsed = feat._parse_from(**parsed)
        return encoded, parsed

    def test_simple_int_label(self):
        feat = IntLabel(n_class=5)
        inputs = {"label": [2, 4]}

        encoded, parsed = TestLabelFeature._encode_and_parse(feat, inputs)
        parsed = parsed["classes"].numpy()

        assert encoded.keys() == feat.encoded_features.keys()
        assert np.allclose(parsed, np.array([0., 0., 1., 0., 1.]))

    def test_simple_str_label(self):
        feat = StrLabel()
        inputs = {"label": ["meow", "woof"]}

        encoded, parsed = TestLabelFeature._encode_and_parse(feat, inputs)
        parsed = parsed["classes"].numpy()

        assert encoded.keys() == feat.encoded_features.keys()
        assert np.all(parsed == np.array([b"meow", b"woof"]))

    def test_simpel_float_label(self):
        feat = FloatLabel()
        inputs = {"label": [3.14, 2.718]}

        encoded, parsed = TestLabelFeature._encode_and_parse(feat, inputs)
        parsed = parsed["classes"].numpy()

        assert encoded.keys() == feat.encoded_features.keys()
        assert np.allclose(parsed, np.array([3.14, 2.718]))


class TestBBoxFeatures:

    @pytest.mark.parametrize(
        "bboxes", [
            [[1.0, 2.0, 3.0, 4], [-1.0, 2.0, 3.0, 4], [1.0, 2.0, 3.0, 4]],
            [[1.0, 2.0, -3.0, 4], [1.0, 2.0, 3.0, 4], [1.0, 2.0, 3.0, 4]],
            [[1.0, 2.0, 3.0, 4], [1.0, 2.0, 3.0, 4], [1.0, 2.0, -3.0, 4]],
            [[1.0, 2.0, 3.0, 4], [1.0, 2.0, 3.0, 11], [1.0, 2.0, 3.0, 4]],
            [[1.0, 2.0, 3.0, 4], [1.0, 2.0, 3.0, -1], [1.0, 2.0, 3.0, 4]],
        ]
    )
    def test_invalid_bbox_should_be_blocked(self, bboxes):
        """Invliad bbox should be blocked by _create_from"""
        feat = BoundingBox(n_class=10, max_bbox_per_data=10)
        with pytest.raises((ValueError, TypeError)):
            features = feat._create_from(bboxes=bboxes)

    def test_encode_decode_bboxes(self):
        """Check encodied ymin, xmin, height, width, class correct"""
        nCapacity = 10
        feat = BoundingBox(n_class=5, max_bbox_per_data=nCapacity)

        for nBox in range(1, 21):
            yx = np.random.randint(1000, size=(nBox, 2)) / 10
            hw = np.random.randint(1, 1000, size=(nBox, 2)) / 10
            classes = np.random.randint(5, size=(nBox, 1))
            boxes = np.concatenate((yx, hw, classes), axis=1)

            encoded, parsed = encode_decode(feat=feat, boxes=boxes)

            assert encoded.keys() == feat.encoded_features.keys()

            parsed = parsed["boxes"]
            assert parsed.shape == (nCapacity, 5)
            for idx in range(nCapacity):
                if idx <= nBox - 1:
                    y, x = yx[idx, :]
                    h, w = hw[idx, :]
                    cls_idx = float(classes[idx, 0])
                    bbox = parsed[idx, ...].numpy()
                    assert np.allclose(np.array((y, x, h, w, cls_idx)), bbox)
                elif idx < nCapacity - 1:
                    bbox = parsed[idx, ...].numpy()
                    assert np.allclose(np.array((0, 0, 0, 0, -1)), bbox)


class TestImageFeature:

    sample = str(SAMPLE_FILES_DIR.joinpath("Freyja.jpg"))

    @staticmethod
    def check_ssim(image: tf.Tensor, criteria: float = 0.99):
        """Check ssim with groud-truth sample"""
        gt = cv2.imread(TestImageFeature.sample)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        if image.shape != gt.shape:
            gt = cv2.resize(gt, (255, 255))
        gt = tf.convert_to_tensor(gt, dtype=tf.float32)
        ssim = tf.image.ssim(image, gt, max_val=255)
        assert ssim.numpy() > criteria

    @pytest.mark.parametrize(
        "resize", [False, True]
    )
    def test_encode_decode_in_eager_execution(self, resize):
        """Test encode-decode within eager execution mode

        Use ssim to check the decoded image
        """
        if resize:
            feat = ImageFeature(resize_shape=(255, 255))
        else:
            feat = ImageFeature()

        encoded, decoded = encode_decode(
            feat=feat, image=TestImageFeature.sample
        )
        assert encoded.keys() == feat.encoded_features.keys()

        # test parsed image similary with original image
        image_tensor = decoded["image_content"]
        if resize:
            TestImageFeature.check_ssim(image_tensor, criteria=0.95)
        else:
            TestImageFeature.check_ssim(image_tensor)

    @pytest.mark.parametrize(
        "resize", [False, True]
    )
    def test_decode_in_graph_mode(self, resize):
        """A regression test for decode image in graph mode"""

        @tf.function
        def encode_decode(feat, image):
            encoded = feat._create_from(image=image)
            example = tf.train.Example(
                features=tf.train.Features(feature=encoded)
            )
            parsed = tf.io.parse_single_example(
                example.SerializeToString(), features=feat.encoded_features
            )
            parsed = feat._parse_from(**parsed)
            return parsed

        if resize:
            feat = ImageFeature(resize_shape=(255, 255))
        else:
            feat = ImageFeature()

        decoded = encode_decode(feat, image=TestImageFeature.sample)
        image_tensor = decoded["image_content"]
        if resize:
            TestImageFeature.check_ssim(image_tensor, criteria=0.95)
        else:
            TestImageFeature.check_ssim(image_tensor)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
