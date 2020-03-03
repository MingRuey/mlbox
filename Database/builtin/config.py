import json
import importlib
from collections import namedtuple


_CFG = namedtuple(
    "DatasetConfig", [
        "name", "versions", "location",
        "info", "parser", "train", "test"
    ]
)


class DatasetConfig(_CFG):

    @classmethod
    def from_json(cls, file: str):
        with open(file, "r") as f:
            content = json.load(f)

        name = content["name"]
        vers = content["versions"]
        loc = content["location"]
        info = content["info"]

        parser_str = content["parser"]
        parser = importlib.import_module("MLBOX.Database.builtin.parsers")
        parser = getattr(parser, parser_str)

        train = content["train"]
        test = content["test"]
        return super().__new__(
            cls,
            name=name, versions=vers, location=loc,
            info=info, parser=parser,
            train=train, test=test
        )
