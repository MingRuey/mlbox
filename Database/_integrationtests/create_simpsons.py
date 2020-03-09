"""
Use Simpson dataset on Kaggle as sample for building Image Dataset
https://www.yanlong.app/alexattia/the-simpsons-characters-dataset

The data is cleaned and organized under /rawdata2/toyset/Simpsons
"""
import json
import logging
from pathlib import Path

import pandas
import pytest
import tensorflow as tf

from MLBOX.Database import DBuilder, Feature, ParserFMT
from MLBOX.Database import DBLoader

DATA_DIR = "/rawdata2/toyset/Simpsons/imgs_train"
DATA_CSV = "/rawdata2/toyset/Simpsons/simpsons.csv"
CLS_DEFINITION = "/rawdata2/toyset/Simpsons/simpsons.json"


def _get_class_definition_dict() -> dict:
    with open(CLS_DEFINITION, "r") as f:
        def_map = json.load(f)

    def_map = def_map["classes"]
    def_map = {v: k for k, v in def_map.items()}
    return def_map


def _data_gener():
    def_map = _get_class_definition_dict()
    csv = pandas.read_csv(DATA_CSV)

    path = Path(DATA_DIR)
    for row in csv.itertuples():
        image_path = path.joinpath(row.id + ".jpg")
        yield {
            "image": str(image_path),
            "label": def_map[row.labels]
        }


class SimpsonFMT(ParserFMT):
    features = [Feature.ImageFeature(), Feature.StrLabel()]


def build_simpson(tmpdir):
    parser = SimpsonFMT()
    builder = DBuilder(name="simpsons", parser=parser)
    builder.build_tfrecords(
        generator=_data_gener(),
        output_dir=str(tmpdir), split="train",
        num_of_tfrecords=5
    )


def load_simpson(tmpdir):
    parser = SimpsonFMT()
    loader = DBLoader()
    loader.load(str(tmpdir), parser=parser)

    train = loader.train.to_tfdataset(epoch=1, batch=1)
    for example in train:
        print(example)
        break


if __name__ == "__main__":
    logging.basicConfig()
    # build_simpson(tmpdir="/tmp/pytest-of-mrchou")
    load_simpson(tmpdir="/tmp/pytest-of-mrchou")
