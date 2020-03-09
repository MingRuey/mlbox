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
class TestJsonInfo:
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

    def test_mnist(self, config):
        """Check content of MNIST dataset"""
        db = DBLoader()
        db.load_built_in("MNIST")

        ds_train = db.train.to_tfdataset(1, 1)
        ds_test = db.test.to_tfdataset(1, 1)
        ds_train = ds_train.take(1)
        ds_test = ds_test.take(1)

        train_example = next(iter(ds_train))
        test_example = next(iter(ds_test))

        for exampel in [train_example, test_example]:
            image = train_example["image"].numpy()
            label = train_example["label"].numpy()
            assert image.shape == (1, 28, 28, 1)
            assert label.shape == (1,)


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
