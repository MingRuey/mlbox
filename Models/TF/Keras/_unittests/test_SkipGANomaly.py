import os
import sys
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402
from tensorflow.keras import Input, Model  # noqa: E402
import pytest  # noqa: E402

from MLBOX.Models.TF.Keras.UNet import _crop_2d, _down_sample, _up_sample  # noqa: E402
from MLBOX.Models.TF.Keras.UNet import UNET, UNetPadType  # noqa: E402


class TestUNet:

    @pytest.mark.xfail(reason="Not implement test yet")
    @pytest.mark.parametrize(
        "pad_type", [UNetPadType.valid, UNetPadType.reflect, UNetPadType.zero]
    )
    def test_down_sample_layer_shape(self, pad_type):
        """Shape should become (h/2, w/2) or (h/2 - 2, w/2 -2) depends on the input"""
        raise NotImplementedError()

    @pytest.mark.xfail(reason="Not implement test yet")
    @pytest.mark.parametrize(
        "pad_type", [UNetPadType.valid, UNetPadType.reflect, UNetPadType.zero]
    )
    def test_up_sample_layer_shape(self, pad_type):
        raise NotImplementedError()

    def test_input_output_shape(self):
        """UNet input shape should match its output shape"""
        inputs = Input((252, 572, 3))
        unet = UNET(n_base_filter=64, n_down_sample=4, n_class=2, padding="valid")
        model = Model(inputs=inputs, outputs=unet.forward(inputs))

        img = tf.zeros((1, 252, 572, 3))
        result = model.predict(img)

        def valid_shape_cal(size):
            for _ in range(4):
                size = size / 2 - 2
            size = size - 2 - 2
            for _ in range(4):
                size = (size - 2) * 2
            return size

        assert result.shape == (1, valid_shape_cal(252), valid_shape_cal(572), 2)

        inputs = Input(shape=(256, 512, 3))
        unet = UNET(n_base_filter=64, n_down_sample=4, n_class=4, padding="reflect")
        model = tf.keras.Model(inputs=inputs, outputs=unet.forward(inputs))

        img = tf.zeros((1, 256, 512, 3))
        result = model.predict(img)
        assert result.shape == (1, 256, 512, 4)


if __name__ == "__main__":
    pytest.main(["-s", "-v", "-x", __file__])
