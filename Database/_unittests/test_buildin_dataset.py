import os
import json
import pathlib
import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from MLBOX.Database.builtin import BUILT_INS  # noqa: E042
from MLBOX.Database import DBLoader  # noqa: E042


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
        db = DBLoader()
        assert db.info == ""
        assert db.train is None
        assert db.test is None

        db.load_built_in(config.name)
        assert db.train.count == config.train["count"]
        assert db.test.count == config.test["count"]
        assert db.info != ""


_IMAGE_DATASETS = [
    cfg for cfg in BUILT_INS if cfg.name in
    ["mnist", "cifar100", "cifar10"]
]


@pytest.mark.parametrize(
    "config", _IMAGE_DATASETS, ids=[cfg.name for cfg in _IMAGE_DATASETS]
)
class TestImageDataset:

    def test_data_shape(self, config):
        """Check content of MNIST dataset"""
        db = DBLoader()
        db.load_built_in(config.name)

        train_example = db.train.get_sample()
        test_example = db.train.get_sample()

        info = config.info["image"]
        expected_shape = info["height"], info["width"], info["channel"]

        for exampel in [train_example, test_example]:
            image = train_example["image"].numpy()
            label = train_example["label"].numpy()
            assert image.shape == expected_shape
            assert label.shape == ()


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
