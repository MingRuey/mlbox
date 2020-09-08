import tensorflow as tf
import tensorflow.keras.layers as kk


class ReLU6(kk.Layer):
    """
    ReLU6 activation layer.
    """
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x)


class Swish(kk.Layer):
    """
    Swish activation function from 'Searching for Activation Functions,' https://arxiv.org/abs/1710.05941.
    """
    def call(self, x):
        return x * tf.nn.sigmoid(x)


class HSigmoid(kk.Layer):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSigmoid, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.relu6(x + 3.0) / 6.0


class HSwish(kk.Layer):
    """
    H-Swish activation function from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
    """
    def __init__(self, **kwargs):
        super(HSwish, self).__init__(**kwargs)

    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) / 6.0


class PReLU2(kk.Layer):
    """
    Parametric leaky version of a Rectified Linear Unit (with wide alpha).

    Parameters:
    ----------
    alpha : int, default 0.25
        Negative slope coefficient.
    """
    def __init__(self,
                 alpha=0.25,
                 **kwargs):
        super(PReLU2, self).__init__(**kwargs)
        self.active = kk.LeakyReLU(alpha=alpha)

    def call(self, x):
        return self.active(x)


def get_activation_layer(activation: str):
    """Create activation layer from string.

    Note:
        if activation is anything other than string,
        it will be directly returned as it is. This is convenient
        for creating 'activation' input argument for layers.

    Args:
        activation (str): function name of the activation layer

    Returns:
        keras layer object
    """
    if not isinstance(activation, str):
        return activation

    if activation == "relu":
        return kk.ReLU()
    elif activation == "relu6":
        return ReLU6()
    elif activation == "swish":
        return Swish()
    elif activation == "hswish":
        return HSwish()
    elif activation == "sigmoid":
        return tf.nn.sigmoid
    elif activation == "hsigmoid":
        return HSigmoid()
    else:
        msg = "Unrecognized activation function: {}"
        raise NotImplementedError(msg.format(activation))
