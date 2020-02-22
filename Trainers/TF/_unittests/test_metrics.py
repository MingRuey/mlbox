import pathlib

import os
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402
import tensorflow.keras as keras  # noqa: E402
import pytest  # noqa: E402

from MLBOX.Trainers.TF.Metrics import SSIM  # noqa: E402
from MLBOX.Trainers.TF._unittests.configs import SAMPLE_DIR  # noqa: E402


BW_SAMPLE = SAMPLE_DIR.joinpath("bw_sample.tiff")
assert BW_SAMPLE.is_file()

COLOR_SAMPLE = SAMPLE_DIR.joinpath("color_sample.jpg")
assert COLOR_SAMPLE.is_file()


def _get_identity_model(input_shape):
    inputs = keras.Input(input_shape)
    return keras.Model(inputs=inputs, outputs=inputs)


class TestSSIM:

    @pytest.mark.parametrize(
        "sample", [BW_SAMPLE, COLOR_SAMPLE]
    )
    def test_against_scikit_image(self, sample):
        """SSIM should yield simialr results as scikit-image implementation"""
        img = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
        img = img.astype("float32") / 255

        targets = [img, img * 0.95, img - np.min(img), img * 0.8]
        sk_ssims = [
            ssim(img, target, data_range=1.0, multichannel=True)
            for target in targets
        ]
        sk_ssims = sum(sk_ssims) / 4

        inputs = np.stack([img, img, img, img], axis=0)
        targets = np.stack(targets, axis=0)

        if img.ndim == 2:
            img = img[..., np.newaxis]
            inputs = inputs[..., np.newaxis]
            targets = targets[..., np.newaxis]

        model = _get_identity_model(img.shape)
        model.compile(loss="MSE", metrics=[SSIM()])

        _, ssim_values = model.evaluate(x=inputs, y=targets, batch_size=4)
        assert abs(ssim_values - sk_ssims) < 0.01


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
