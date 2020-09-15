import logging

import tensorflow as tf
import tensorflow.keras as keras

from MLBOX.Models import ResNeXt50
from MLBOX.Database import DBLoader
from MLBOX.Database.builtin.parsers import IMAGENET
from MLBOX.Trainers import KerasBaseTrainer


def random_rot(image):
    rot_flag = tf.random.uniform([1], 0, 4, dtype=tf.int32)
    rotated = tf.cond(
        rot_flag[0] == 0,
        true_fn=lambda: image,
        false_fn=lambda: tf.image.rot90(image, k=rot_flag[0])
    )
    return rotated


def random_resize_crop(image):
    rnd_area = tf.random.uniform([1], 0.1, 1.0) * 224 * 224
    rnd_ratio = tf.random.uniform([1], 0.8, 1.25)
    h = tf.round(tf.sqrt(rnd_area * rnd_ratio))[0]
    w = tf.round(tf.sqrt(rnd_area / rnd_ratio))[0]

    rnd_ratio_flag = tf.random.uniform([1], 0, 1)
    image = tf.cond(
        rnd_ratio_flag[0] < 0.5,
        true_fn=lambda: tf.image.random_crop(image, size=(h, w, 3)),
        false_fn=lambda: tf.image.random_crop(image, size=(w, h, 3)),
    )
    return image


class RemapImagenet(IMAGENET):

    def __init__(self, augmentation: bool = True):
        self._aug = bool(augmentation)

    def parse_example(self, example: tf.Tensor):
        example = super().parse_example(example)
        image = example.pop("image")
        if self._aug:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = random_rot(image)
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, 0.5, 1.5)
            image = tf.image.random_hue(image, max_delta=0.5)
            image = tf.image.random_saturation(image, 0.5, 1.5)
            image = random_resize_crop(image)

        image = tf.image.resize(
            image, (224, 224),
            preserve_aspect_ratio=False,
            antialias=False
        )
        label = example.pop("label")
        return image, label


if __name__ == "__main__":
    model = ResNeXt50(
        input_shape=(224, 224, 3),
        include_fc=True, n_fc=1000,
        load_pretrained=False
    )

    imagenet = DBLoader()
    imagenet.load_built_in("imagenet", parser=None)

    optimizer_params = {
        "learning_rate": 0.0025, "momentum": 0.9
    }
    hyper = {
        "SGD": str(optimizer_params),
        "lr_decay": 0.1,
        "loss": keras.losses.CategoricalCrossentropy(from_logits=True),
        "batch": 32,
    }
    logging.info("hyper info: {}".format(hyper))

    trainer = KerasBaseTrainer(
        model=model,
        loss=hyper["loss"],
        optimizer=keras.optimizers.SGD(**optimizer_params),
        out_dir="/archive/imagenet-pretrained",
        metrics=["acc"]
    )

    train_db = imagenet.train
    train_db._parser = RemapImagenet(True)
    val_db = imagenet.test
    val_db._parser = RemapImagenet(False)

    trainer.train(
        train_db=train_db,
        vali_db=val_db,
        validation_freq=1,
        lr_decay_factor=hyper["lr_decay"],
        batch_size=hyper["batch"],
        max_epoch=100,
        early_stop_patience=20,
        load_best=True
    )
