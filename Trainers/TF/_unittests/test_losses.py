import cv2
import numpy as np
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
import pytest

from MLBOX.Trainers.TF.Loss import SSIMLoss
from MLBOX.Trainers.TF._unittests.configs import SAMPLE_DIR  # noqa: E402


COLOR_SAMPLE = SAMPLE_DIR.joinpath("color_sample.jpg")
assert COLOR_SAMPLE.is_file()


class TestSSIM:

    def test_results_matches_tf(self):
        """The loss value returned should be negative scikit-image"""
        img = cv2.imread(str(COLOR_SAMPLE))[..., ::-1]
        img = tf.convert_to_tensor(img)
        img = tf.cast(img, tf.float32)
        imgs = [img, img, img, img]
        imgs = np.stack([img, img, img, img], axis=0)
        targets = [img, img * 0.95, img - np.min(img), img * 0.8]
        targets = np.stack(targets, axis=0)

        targets_t = tf.convert_to_tensor(targets)
        imgs_t = tf.convert_to_tensor(imgs)

        loss = SSIMLoss(max_val=255)(imgs_t, targets_t)
        neg_ssim = np.mean(1.0 - ssim(imgs, targets, multichannel=True))
        assert np.allclose(loss.numpy(), neg_ssim)

        imgs_t = imgs_t / 255
        imgs = targets_t / 255
        targets = imgs_t / 255
        targets_t = targets_t / 255

        loss = SSIMLoss(max_val=1)(imgs_t, targets_t)
        neg_ssim = np.mean(1.0 - ssim(imgs, targets, multichannel=True))
        assert np.allclose(loss.numpy(), neg_ssim)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
