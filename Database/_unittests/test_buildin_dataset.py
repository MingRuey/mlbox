import json
import pathlib
import pytest

from MLBOX.Database.builtin import BUILT_INS
from MLBOX.Database import DBLoader


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


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
