import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402
import pytest  # noqa: E402

from MLBOX.Trainers.TF.Loss import SSIMLoss
from MLBOX.Trainers.TF._unittests.configs import SAMPLE_DIR  # noqa: E402


COLOR_SAMPLE = SAMPLE_DIR.joinpath("color_sample.jpg")
assert COLOR_SAMPLE.is_file()


class TestSSIM:

    def test_results_matches_tf(self):
        """The loss value returned should be negative scikit-image"""
        img = cv2.imread(str(COLOR_SAMPLE))[..., ::-1]
        imgs = [img, img, img, img]
        targets = [img, img * 0.95, img - np.min(img), img * 0.8]

        imgs_t = tf.convert_to_tensor(
            np.stack(imgs, axis=0), dtype=tf.float32
        )
        targets_t = tf.convert_to_tensor(
            np.stack(targets, axis=0), dtype=tf.float32
        )

        loss = SSIMLoss(max_val=255)(imgs_t, targets_t)
        ssims = np.stack(
            [
                ssim(
                    img, target,
                    gaussian_weights=True, K1=0.01, K2=0.03,
                    data_range=255, multichannel=True
                )
                for img, target in zip(imgs, targets)
            ],
            axis=0
        )
        neg_ssim = np.mean(1.0 - ssims)
        assert loss.numpy() - neg_ssim < 1e-4

        imgs_t = imgs_t / 255
        targets_t = targets_t / 255
        imgs = [img/255 for img in imgs]
        targets = [target/255 for target in targets]

        loss = SSIMLoss(max_val=1)(imgs_t, targets_t)
        ssims = np.stack(
            [
                ssim(
                    img, target,
                    gaussian_weights=True, K1=0.01, K2=0.03,
                    data_range=1, multichannel=True
                )
                for img, target in zip(imgs, targets)
            ],
            axis=0
        )
        neg_ssim = np.mean(1.0 - ssims)
        assert loss.numpy() - neg_ssim < 1e-4


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
