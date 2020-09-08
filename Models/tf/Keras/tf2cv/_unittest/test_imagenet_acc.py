import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import cv2
import numpy as np
import pytest

from MLBOX.Database import DBLoader
from MLBOX.Database.builtin.parsers import IMAGENET
from MLBOX.Models import ResNeXt50
from MLBOX.Models.tf.Keras.utils import log_model
from MLBOX.playground.script_train_imagenet import RemapImagenet


def count_trainable(model, name):
    trainalbe = np.sum(K.count_params(w) for w in model.trainable_weights)
    print(name, trainalbe)


class TestImageNetInference:

    def test_top1_accuracy(self):
        pass

    def test_examine_image(self, tmp_path):
        imagenet = DBLoader()
        imagenet.load_built_in("imagenet", parser=RemapImagenet())
        dataset = imagenet.test.to_tfdataset(batch=1, epoch=1)
        dataset = dataset.take(20)
        for idx, data in enumerate(dataset):
            image, label = data
            img = image[0, ..., ::-1].numpy()
            cv2.imwrite(str(tmp_path.joinpath("{:02d}.bmp".format(idx))), img)

    @pytest.mark.skip(reason="saving time")
    @pytest.mark.parametrize(
        "model", ["resnext"]
    )
    def test_inference_speed(self, model):
        if model == "resnet":
            model = resnet = tf.keras.applications.ResNet50V2(
                include_top=True, input_shape=(256, 256, 3),
                weights=None, classes=1000
            )
        elif model == "resnext":
            model = ResNeXt50(
                input_shape=(256, 256, 3),
                include_fc=True, n_fc=1000, load_pretrained=True
            )
            print(model.count_params())

        imagenet = DBLoader()
        imagenet.load_built_in(
            "imagenet", parser=RemapImagenet()
        )

        dataset = imagenet.test.to_tfdataset(batch=1, epoch=1)
        dataset = dataset.take(1000)
        start = time.time()
        model.predict(dataset)
        print(time.time() - start)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
