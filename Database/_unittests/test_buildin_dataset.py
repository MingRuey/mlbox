import json
import pathlib
import pytest

from MLBOX.Database import builtin
from MLBOX.Database import DBLoader

BUILT_IN_PATH = pathlib.Path(list(builtin.__path__)[0])
assert BUILT_IN_PATH.is_dir()


class TestJsonInfo:
    """check the json file information matches the actual tfrecords"""

    @pytest.mark.parametrize(
        "config", [str(file) for file in BUILT_IN_PATH.glob("*.json")]
    )
    def test_existence(self, config):
        """the dataset with given name & version should exist at locatin"""
        with open(config, "r") as f:
            content = json.load(f)

        name = content["name"]
        version = content["version"]
        loc = pathlib.Path(content["location"])

        assert loc.is_dir()
        assert loc.joinpath(name).is_dir()
        assert loc.joinpath(name).joinpath(version).is_dir()

    def test_loadable(self):
        db = DBLoader.load("mnist")
        db = DBLoader.load("mnist", version="3.0.0")


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
