from enum import Enum
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import Cropping2D, Concatenate


class UNetPadType(Enum):
    valid = "valid"
    reflect = "reflect"
    zero = "zero"


def _conv3_relu(inputs: tf.Tensor, n_filter: int, padding="reflect", use_batchnorm=True) -> tf.Tensor:
    if padding == "reflect":
        padded = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        padding_kw = "valid"
    elif padding == "symmertric":
        padded = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        padding_kw = "valid"
    elif padding == "valid":
        padded = inputs
    elif padding == "zero":
        padded = inputs
        padding_kw = "same"
    else:
        raise ValueError("Unrecognized padding option: {}".format(padding))

    _kwargs = {
        "filters": n_filter, "kernel_size": 3, "padding": padding_kw,
        "activation": "relu", "kernel_initializer": 'he_uniform'
    }

    conv = Conv2D(**_kwargs)(padded)
    if use_batchnorm:
        conv = BatchNormalization(axis=-1)(conv)
    return conv


def _down_sample(
        inputs: tf.Tensor,
        n_filter: int,
        pad_type: UNetPadType
        ) -> tf.Tensor:
    conv1 = _conv3_relu(inputs, n_filter=n_filter, padding=pad_type.value)
    conv2 = _conv3_relu(bn1, n_filter=n_filter, padding=pad_type.value)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return pool, conv2


def _crop_2d(inputs: tf.Tensor, target_shape: tuple)-> tf.Tensor:
    """User tf.keras.layer.Cropping2D to corop inputs to match target_shape

    Args:
        inputs (tf.Tensor): tensor of shape (batch, height, width, channels)
        target_shape (tuple): the target shape in (heigt, width)

    Returns:
        the cropped tensor,
        of shape (batch, target height, target width, channels)
    """
    assert all(target_shape) >= 1
    b, h, w, c = inputs.shape
    assert h >= target_shape[0] and w >= target_shape[1]
    _top = (h  - target_shape[0]) // 2
    _bottom = h  - target_shape[0] - _top
    _left = (w - target_shape[1]) // 2
    _right = w  - target_shape[1] - _left
    cropped = Cropping2D(((_top, _bottom), (_left, _right)))(inputs)
    return cropped


def _up_sample(
        inputs: tf.Tensor,
        feat_to_concat: tf.Tensor,
        n_filter: int,
        pad_type: UNetPadType
        ) -> tf.Tensor:

    up = UpSampling2D(size=(2, 2))(inputs)
    up = _conv3_relu(up, n_filter=n_filter, padding="symmertric")

    if pad_type == UNetPadType.valid:
        _b, _h, _w, _c = up.shape
        _crop = _crop_2d(inputs=feat_to_concat, target_shape=(_h, _w))
        feat = Concatenate(axis=-1)([_crop, up])
    elif pad_type == UNetPadType.reflect or pad_type == UNetPadType.zero:
        feat = Concatenate(axis=-1)([feat_to_concat, up])
    else:
        raise ValueError("Unrecognized pad type: {}".format(pad_type))

    conv1 = _conv3_relu(feat, n_filter=n_filter, padding=pad_type.value)
    conv2 = _conv3_relu(conv1, n_filter=n_filter, padding=pad_type.value)
    return conv2


class UNET:

    def __init__(
            self,
            n_base_filter: int = 64,
            n_down_sample: int = 4,
            n_class: int = 2,
            padding: str = "reflect"
            ):
        """Init a UNET class

        Args:
            n_base_filter (int, optional):
                Number of filters at first feature map. Defaults to 64.
            n_down_sample (int, optional):
                Number of times to down sample 2X. Defaults to 4.
            n_class (int, optional):
                Number of class. Defaults to 2.
            padding:
                str, is either "reflect" or "valid"(=No padding).
                note that it's argued reflecting has better performance:
                https://www.kaggle.com/c/data-science-bowl-2018/discussion/54426

                also, valid padding will shrink the input dimension
        """
        self._nBaseFilter = n_base_filter
        self._nDownSample = n_down_sample
        self._nClass = n_class
        if padding.lower() not in ["reflect", "valid"]:
            raise ValueError("Unrecognized padding: {}".format(padding))

        if padding == "reflect":
            self._padType = UNetPadType.reflect
        else:
            self._padType = UNetPadType.valid

    def forward(self, inputs: tf.Tensor) -> tf.Tensor:
        # Down Sampling
        pool = inputs
        down_feats = []
        for index in range(1, self._nDownSample + 1):
            pool, feat = _down_sample(
                pool,
                n_filter=(2**index)*self._nBaseFilter,
                pad_type=self._padType
                )
            down_feats.append(feat)

        _bottom = _conv3_relu(
            pool,
            n_filter=(2**index)*self._nBaseFilter,
            padding=self._padType.value
        )
        _bottom = _conv3_relu(
            _bottom,
            n_filter=(2**index)*self._nBaseFilter,
            padding=self._padType.value
        )

        pool = _bottom
        for index in range(self._nDownSample, 0, -1):
            pool = _up_sample(
                pool, feat_to_concat=down_feats[index-1],
                n_filter=((2**index)*self._nBaseFilter)//2,
                pad_type=self._padType
            )

        output = Conv2D(
            filters=self._nClass, kernel_size=1,
            kernel_initializer='he_uniform'
        )(pool)
        return output


def unet_loss():

    _EPILSON = 12.0

    @tf.function
    def pixelwise_softmax(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate pixel-wsie cross entropy

        Args:
            y_true (tf.Tensor): tensor of shape (batch, height, width, 1)
            y_pred (tf.Tensor): tensor of shape (batch, height, width, nClass)

        Returns:
            tf.Tensor: tensor of shape (batch,)
        """
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_true = tf.squeeze(y_true, axis=-1)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        mini = tf.constant(value=0.0, dtype=tf.float32)
        maxi = tf.constant(value=_EPILSON, dtype=tf.float32)
        cross_entropy = tf.clip_by_value(cross_entropy, mini, maxi)
        return tf.reduce_mean(cross_entropy)

    return pixelwise_softmax


def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    dice_vec = (2. * intersection + smooth) / \
        (K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2]) + smooth)
    return 1 - K.mean(dice_vec)


def dice_coef(y_true, y_pred, thres=0.5, smooth=1e-2):
    y_pred = K.cast(y_pred > 0.5, 'float32')
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    dice_vec = (2. * intersection + smooth) / \
        (K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2]) + smooth)
    return K.mean(dice_vec)


if __name__ == "__main__":
    pass
