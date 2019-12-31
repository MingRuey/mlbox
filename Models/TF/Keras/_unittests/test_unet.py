import tensorflow as tf
from MLBOX.Models.TF.Keras.UNet import _crop_2d, _down_sample, _up_sample
from MLBOX.Models.TF.Keras.UNet import UNET, unet_loss

if __name__ == "__main__":
    inputs = tf.keras.layers.Input(shape=(572, 572, 3))
    unet = UNET(n_base_filter=64, n_down_sample=4, n_class=4, padding="valid")
    model = tf.keras.Model(inputs=inputs, outputs=unet.forward(inputs))

    img = tf.zeros((1, 572, 572, 3))
    result = model.predict(img)
    print(result.shape)

    inputs = tf.keras.layers.Input(shape=(256, 1600, 3))
    unet = UNET(n_base_filter=64, n_down_sample=4, n_class=4, padding="reflect")
    model = tf.keras.Model(inputs=inputs, outputs=unet.forward(inputs))

    img = tf.zeros((1, 256, 1600, 3))
    result = model.predict(img)
    print(result.shape)

    img = tf.zeros((1, 572, 572, 3))
    pool, feat = _down_sample(img, n_filter=64)
    print(pool.shape, feat.shape)
    print(_crop_2d(img, target_shape=(400, 400)).shape)
    img = tf.zeros((1, 56, 56, 512))
    feat = tf.zeros((1, 112, 112, 512))
    print(_up_sample(img, feat, n_filter=64).shape)
