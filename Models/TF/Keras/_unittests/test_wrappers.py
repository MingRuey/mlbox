import os
import sys
import inspect
import unittest.mock as mock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pytest  # noqa: E402
from tensorflow.keras import Input  # noqa: E402
from tensorflow.keras.layers import Dense  # noqa: E402

from MLBOX.Models.TF.Keras.wrappers.layer_wrappers import ConstraintWrapper  # noqa: E402
from MLBOX.Models.TF.Keras.modules.ResDecoder import ResNetDecoder  # noqa: E402
from MLBOX.Models.TF.Keras.betaVAE import ResNetEncoder, SampleLayer  # noqa: E402


class MockedLayer:
    """As target for wrapper"""

    _mock = mock.MagicMock()
    return_value = None

    def __new__(cls, *args, kernel_constraint=None, **kwargs):
        if kernel_constraint is not None:
            kwargs["kernel_constraint"] = kernel_constraint
        cls.return_value = cls._mock(*args, **kwargs)
        cls.assert_called_with = cls._mock.assert_called_with
        cls.mock_calls = cls._mock.mock_calls
        cls.call_args_list = cls._mock.call_args_list
        return cls.return_value


class MockedConv(MockedLayer):

    _mock = mock.MagicMock()
    return_value = None


class MockedDense(MockedLayer):

    _mock = mock.MagicMock()
    return_value = None


class TestConstraintWrapper:

    def test_mock_single_object(self):
        """ConstraintWrapper should set the default 'kernel_constraint' kwrag"""

        @ConstraintWrapper(constraint="constraint", targets={MockedLayer})
        def target_func(inputs):
            dense1 = Dense(10)(inputs)
            wrapped = MockedLayer(3.14, activation="relu")
            wrapped = wrapped(dense1)
            return wrapped

        inputs = Input((240, 240, 3))
        wrapped = target_func(inputs)

        MockedLayer.assert_called_with(
            3.14, activation="relu",
            kernel_constraint="constraint"
        )
        layer_instance = MockedLayer.return_value
        layer_instance.assert_called()

        # check the function default is actually restored
        MockedLayer(2.718, activation="softmax")
        assert MockedLayer.mock_calls[-1] == mock.call(2.718, activation="softmax")

    def test_mock_multiple_layer_types(self):
        """ConstraintWrapper should set the default 'kernel_constraint' kwrag"""

        @ConstraintWrapper(
            constraint="constraint",
            targets={MockedConv, MockedDense}
        )
        def target_func(inputs):
            dense1 = Dense(10)(inputs)
            mocked = MockedConv(3.14, activation="relu")(dense1)
            mocked = MockedDense(2.71, activation="softmax")(mocked)
            return mocked

        inputs = Input((240, 240, 3))
        wrapped = target_func(inputs)

        MockedConv.assert_called_with(
            3.14, activation="relu",
            kernel_constraint="constraint"
        )
        conv_instance = MockedConv.return_value
        conv_instance.assert_called()

        MockedDense.assert_called_with(
            2.71, activation="softmax",
            kernel_constraint="constraint"
        )
        dense_instance = MockedDense.return_value
        dense_instance.assert_called_with(conv_instance.return_value)

        # check the function default is actually restored
        MockedConv(8.7, activation="tanh")
        MockedDense(87, activation="sine")
        assert MockedConv.mock_calls[-1] == mock.call(8.7, activation="tanh")
        assert MockedDense.mock_calls[-1] == mock.call(87, activation="sine")


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
