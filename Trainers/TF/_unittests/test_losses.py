import cv2
import tensorflow as tf
import pytest

from MLBOX.Trainers.TF.Loss import SSIMLoss
from MLBOX.Trainers.TF._unittests.configs import SAMPLE_DIR  # noqa: E402


COLOR_SAMPLE = SAMPLE_DIR.joinpath("color_sample.jpg")
assert COLOR_SAMPLE.is_file()


class TestSSIM:

    def test_results_matches_tf(self):
        """The loss value returned should be negative of tf.image.ssim"""
        sample = cv2.imread(str(COLOR_SAMPLE))[..., ::-1]
        sample = tf.convert_to_tensor(sample)
        sample = tf.cast(sample, tf.float32)

        degrade = sample * 0.8
        loss = SSIMLoss(max_val=255)(degrade, sample)
        neg_ssim = -1 * tf.image.ssim(degrade, sample, max_val=255)

        assert tf.reduce_all(tf.equal(loss, neg_ssim)).numpy()

        sample = sample / 255
        degrade = sample * 0.8
        loss = SSIMLoss(max_val=1)(degrade, sample)
        neg_ssim = -1 * tf.image.ssim(degrade, sample, max_val=1)

        assert tf.reduce_all(tf.equal(loss, neg_ssim)).numpy()


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
