import os
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import pytest  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E40

from MLBOX.Database.core.features import IntLabel, StrLabel, FloatLabel  # noqa: E402
from MLBOX.Database.core.features import ImageFeature  # noqa: E402
from MLBOX.Database._unittests.configs import SAMPLE_FILES_DIR  # noqa: E402


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


class TestImageFeature:

    encoded = None
    sample = str(SAMPLE_FILES_DIR.joinpath("Freyja.jpg"))

    def test_create_from_file(self):
        """
        the create_keys should match inputs of _create_from
        the keys returned by _create_from should match parse_features
        """
        feat = ImageFeature()

        features = (TestImageFeature.sample, )
        inputs = {}
        for key, feature in zip(feat.create_keys, features):
            inputs[key] = feature

        encoded = feat._create_from(**inputs)
        assert encoded.keys() == feat.encoded_features.keys()

        TestImageFeature.encoded = encoded

    def test_parse_from_features(self):
        """
        the parse_features should match inputs of _parse_from.
        the parsed tensor should match what had been encoded.
        we use ssim to check image equality
        """
        if self.encoded is None:
            msg = "Fail due to {} fails."
            raise RuntimeError(msg.format(self.test_create_from_file.__name__))

        feat = ImageFeature()

        example = tf.train.Example(features=tf.train.Features(
            feature=TestImageFeature.encoded
        ))
        parsed = tf.io.parse_single_example(
            example.SerializeToString(), features=feat.encoded_features
        )
        parsed = feat._parse_from(**parsed)
        image_tensor = parsed["image_content"]

        # test parsed image similary with original image
        ground_truth = cv2.imread(TestImageFeature.sample)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
        ssim = tf.image.ssim(image_tensor, ground_truth, max_val=255)
        assert ssim.numpy() > 0.99


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
