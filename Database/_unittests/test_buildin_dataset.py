import os
import json
import pathlib
import pytest

import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from MLBOX.Database.builtin import BUILT_INS  # noqa: E042
from MLBOX.Database import DBLoader  # noqa: E042
from MLBOX.Database.builtin.parsers import CoCoObjDet, CoCoSeg  # noqa: E402

from IMGBOX import Image, Mask


@pytest.mark.parametrize(
    "config", BUILT_INS, ids=[cfg.name for cfg in BUILT_INS]
)
class TestAllBuiltin:
    """check the json file information matches the actual tfrecords"""

    def test_existence(self, config):
        """the dataset with given name & version should exist at location"""
        name = config.name
        loc = pathlib.Path(config.location)
        for ver in config.versions:
            assert loc.is_dir()
            assert loc.joinpath(name).is_dir()
            assert loc.joinpath(name).joinpath(ver).is_dir()

    def test_loadable(self, config):
        """Can load all built-in datasets"""
        if config.name == "imagenet":
            pytest.skip(msg="count ImageNet takes too much time")

        db = DBLoader()
        assert db.info == ""
        assert db.train is None
        assert db.test is None

        db.load_built_in(config.name)
        assert db.train.count == config.train["count"]
        assert db.test.count == config.test["count"]
        assert db.info != ""


CLSFY = [
    cfg for cfg in BUILT_INS
    if cfg.info["type"] == "image classification"
]
OBJDET = [
    cfg for cfg in BUILT_INS
    if cfg.info["type"] == "object detection"
]


class TestImageDataset:

    @pytest.mark.parametrize(
        "config", CLSFY, ids=[cfg.name for cfg in CLSFY]
    )
    def test_classification_data_shape(self, config):
        """Check the tensor shape of classification dataset"""
        db = DBLoader()
        db.load_built_in(config.name)

        train_example = db.train.get_sample()
        test_example = db.train.get_sample()

        info = config.info["image"]
        if config.name == "imagenet":
            expected_shape = (256, 256, 3)
        else:
            expected_shape = info["height"], info["width"], info["channel"]

        for example in [train_example, test_example]:
            image = example["image"].numpy()
            label = example["label"].numpy()
            assert image.shape == expected_shape

            if config.name == "imagenet":
                assert label.shape == (1000,)
            else:
                assert label.shape == ()

    @pytest.mark.parametrize(
        "config", OBJDET, ids=[cfg.name for cfg in OBJDET]
    )
    def test_object_detection_data_shape(self, config):
        """Check the tensor shape of classification dataset"""
        db = DBLoader()
        db.load_built_in(config.name)

        train_example = db.train.get_sample()
        test_example = db.train.get_sample()

        img_info = config.info["image"]
        expected_img_shape = \
            img_info["height"], img_info["width"], img_info["channel"]

        box_info = config.info["boxes"]
        nClass = box_info["classes"]
        nBox = max(
            box_info["train"]["max boxes of single data"],
            box_info["test"]["max boxes of single data"],
        )
        expected_box_shape = nBox, 4 + nClass

        for example in [train_example, test_example]:
            image = example["image"].numpy()
            boxes = example["boxes"].numpy()
            for img_shp, expected in zip(image.shape, expected_img_shape):
                assert (expected is None) or img_shp == expected
            assert boxes.shape == expected_box_shape


class TestManualCoCoDet:

    CLSMAP_FILE = "/rawdata2/TFDataset/coco/coco-objdet/COCO-2017-ObjectDetection-1.0.0/" \
        + "CoCo-2017_object-detection_classes.json"

    _id2name = None

    @property
    def id2name(self):
        if self._id2name is None:
            with open(self.CLSMAP_FILE, "r") as f:
                content = json.load(f)
                id2name = {}
                for _, vals in content.items():
                    for name, ids in vals.items():
                        id2name[ids["id"]] = name
            self._id2name = id2name
        return self._id2name

    def draw_boxes(self, image: np.array, boxes: np.array):
        """Draw boxes onto the image in-place

        Args:
            image (np.array): image array of shape (None, None, 3)
            boxes (np.array):
                bound boxes of shape (nBox, 5),
                where each box is in format (y, x, h, w, cls)
        """
        for box in boxes:
            y, x, h, w, clsidx = box
            if clsidx >= 0:
                clsname = self.id2name[int(clsidx)]
                cv2.rectangle(
                    image, (x, y), (x + w, y + h), [0, 0, 255], 1
                )
                cv2.putText(
                    image, clsname, (x, y),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1.0, color=[0, 0, 255]
                )

    def manually_examine_bbox(
            self, nImage: int,  out_dir: pathlib.Path, resize_shape=None,
            ):
        parser = CoCoObjDet(one_hot_class=False, resize_shape=resize_shape)

        db = DBLoader()
        db.load_built_in("coco-objdet", parser=parser)

        ds = db.train.to_tfdataset(epoch=1, batch=1)
        for cnt, example in enumerate(ds):
            image = example["image"].numpy()[0, ...].astype(np.uint8)
            image = image[..., ::-1]
            image = np.array(image)

            boxes = example["boxes"].numpy()[0, ...]
            self.draw_boxes(image, boxes)
            cv2.imwrite(str(out_dir.joinpath("train_{}.bmp".format(cnt))), image)
            if cnt == nImage:
                break

        ds = db.test.to_tfdataset(epoch=1, batch=1)
        for cnt, example in enumerate(ds):
            image = example["image"].numpy()[0, ...].astype(np.uint8)
            image = image[..., ::-1]
            image = np.array(image)

            boxes = example["boxes"].numpy()[0, ...]
            self.draw_boxes(image, boxes)
            cv2.imwrite(str(out_dir.joinpath("test_{}.bmp".format(cnt))), image)
            if cnt == nImage:
                break

    def test_manuall_view_boxes(self, tmp_path):
        """Inspect bounding box manually without resizing"""
        self.manually_examine_bbox(10, tmp_path, resize_shape=None)
        print("export results to ", tmp_path)

    def test_manuall_view_boxes_with_resizing(self, tmp_path):
        """Inspect bounding box manually without resizing"""
        self.manually_examine_bbox(10, tmp_path, resize_shape=(500, 300))
        print("export results to ", tmp_path)


class TestManualCoCoSeg:

    CLSMAP_FILE = "/rawdata2/TFDataset/coco/coco-seg/COCO-2017-Segmentation-1.0.0/" \
        + "CoCo-2017_segmentation_classes.json"

    _id2name = None

    @property
    def id2name(self):
        if self._id2name is None:
            with open(self.CLSMAP_FILE, "r") as f:
                content = json.load(f)
                id2name = {}
                for _, vals in content.items():
                    for name, ids in vals.items():
                        id2name[ids["id"]] = name
            self._id2name = id2name
        return self._id2name

    def get_masked_image(self, image: np.array, mask: np.array) -> Image:
        """Create masked Image object"""
        op = Mask(color=(0, 0, 255))
        image = Image(image).to_gray()
        mask = Image(mask)
        return op.on(image, mask)

    def manually_examine_seg(
            self, nImage: int,  out_dir: pathlib.Path, resize_shape=None,
            ):
        parser = CoCoSeg(one_hot_class=False, resize_shape=resize_shape)

        db = DBLoader()
        db.load_built_in("coco-seg", parser=parser)

        ds = db.train.to_tfdataset(epoch=1, batch=1)
        for cnt, example in enumerate(ds):
            image = example["image"].numpy()[0, ...].astype(np.uint8)
            image = image[..., ::-1]
            image = np.array(image)

            mask = example["segment_mask"].numpy()[0, ..., 0]

            img = self.get_masked_image(image, mask)
            img.save(str(out_dir.joinpath("train_{}.bmp".format(cnt))))
            if cnt == nImage:
                break

        ds = db.test.to_tfdataset(epoch=1, batch=1)
        for cnt, example in enumerate(ds):
            image = example["image"].numpy()[0, ...].astype(np.uint8)
            image = image[..., ::-1]
            image = np.array(image)

            mask = example["segment_mask"].numpy()[0, ..., 0]

            img = self.get_masked_image(image, mask)
            img.save(str(out_dir.joinpath("test_{}.bmp".format(cnt))))
            if cnt == nImage:
                break

    def test_manuall_view_seg(self, tmp_path):
        """Inspect bounding box manually without resizing"""
        self.manually_examine_seg(10, tmp_path, resize_shape=None)
        print("export results to ", tmp_path)

    def test_manuall_view_seg_with_resizing(self, tmp_path):
        """Inspect bounding box manually without resizing"""
        self.manually_examine_seg(10, tmp_path, resize_shape=(500, 300))
        print("export results to ", tmp_path)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
