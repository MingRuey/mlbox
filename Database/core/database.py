import os
import re
from pathlib import Path
from typing import List
import tensorflow as tf

from MLBOX.Database.builtin import BUILT_INS
from MLBOX.Database.core.parsers import ParserFMT

OUTPUT_PARALLEL_CALL = 8
OUTPUT_BUFFER_TO_BATCH_RATIO = 16
TRAIN_TFRECORD_PATTERN = r".+-train.tfrecord-\d{5}-of-\d{5}"
TEST_TFRECORD_PATTERN = r".+-test.tfrecord-\d{5}-of-\d{5}"


def _fullpath_listdir(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


class Dataset:

    def __init__(self, tfrecords: List[str], parser: ParserFMT):
        """Create a dataset with tfrecord files and parser"""
        self._files = list(tfrecords)
        self._parser = parser
        self._data_mask = None

    @property
    def info(self):
        pass

    @property
    def count(self) -> int:
        """Number of data inside Dataset"""
        cnt = 0
        for example in tf.data.TFRecordDataset(self._files):
            cnt += 1
        return cnt

    # Both slice and split features are implemented via
    # an internal boolean mask over the dataset.
    # The same mechanism is used by tfds too.
    def __getitem__(self, val):
        """Make dataset support slice behaviour"""
        pass

    def split(self, ratio: float):
        """Split dataset randomly into two pile"""
        pass

    def to_tfdataset(
            self,
            batch: int, epoch: int,
            shuffle_n_batch: int = 100,
            reshuffle_per_epoch: bool = False,
            shuffle_seed: int = 42
            ):
        """Transform Dataset instance into tf.data.Dataset object

        Args:
            batch (int): batch size of the output data
            epoch (int): number of iteration over data
            shuffle_n_batch (int):
                the capacity of the shuffle buffer.
                specify in units of batch.
            reshuffle_per_epoch (bool):
                whether to shuffle data differently for every epoch
            shuffle_seed (int):
                the random seed for shuffle, default to 42.
        """
        dataset = tf.data.TFRecordDataset(self._files)
        dataset = dataset.map(
            self._parser,
            num_parallel_calls=OUTPUT_PARALLEL_CALL
        )
        dataset = dataset.shuffle(
            shuffle_n_batch * batch, seed=42,
            reshuffle_each_iteration=reshuffle_per_epoch
        )
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batch, drop_remainder=True)
        dataset = dataset.prefetch(OUTPUT_BUFFER_TO_BATCH_RATIO)
        return dataset


class _SlicedDataset(Dataset):
    """Created by slicing Dataset instance, which can not be further sliced"""

    def __getitem__(self, val):
        msg = "A sliced dataset object can not be sliced anymore"
        raise TypeError(msg)


class DBLoader:

    def __init__(self):
        self._train, self._test, self._info = None, None, ""

    @property
    def info(self):
        return self._info

    @property
    def train(self) -> Dataset:
        """Get the train set of the loaded database"""
        return self._train

    @property
    def test(self) -> Dataset:
        """Get the test set of the loaded database"""
        return self._test

    def load(self, directory: str, parser: ParserFMT):
        """Load custom built dataset

        Args:
            directory (str): the directory of the dataset
            parser (ParseFMT):
                the function defines how tf.train.Example is parsed
        """
        train_files = [
            f for f in _fullpath_listdir(directory)
            if re.search(TRAIN_TFRECORD_PATTERN, f)
        ]
        self._train = Dataset(train_files, parser) if train_files else None

        test_files = [
            f for f in _fullpath_listdir(directory)
            if re.search(TEST_TFRECORD_PATTERN, f)
        ]
        self._test = Dataset(test_files, parser) if test_files else None

    def load_built_in(
            self,
            name: str,
            version: str = "default",
            parser: ParserFMT = "default"
            ):
        """Load built-in datasets

        Args:
            name (str): the name of the built-in dataset, ex: "mnist"
            version (str):
                the version of the built-in dataset, ex: "3.0.0",
                if set to "default", the lastest is used.
            format (FormatBase):
                the format used to parse the database,
                if set to default, a default parser is used.
        """
        target = None
        for built_in in BUILT_INS:
            if built_in.name == name:
                target = built_in
                break

        if target is None:
            msg = "Database with name {} not found in build-in datasets"
            raise ValueError(msg.format(name))

        if version == "default":
            version = target.versions[0]
        elif version not in target.versions:
            msg = "Version {} not found in dataset {}, availiable ones: {}"
            raise ValueError(msg.format(version, target.name, target.versions))

        if parser == "default":
            parser = target.parser

        loc = str(Path(target.location).joinpath(name).joinpath(version))
        self.load(directory=loc, parser=parser)

        self._info = "Database {}(ver {})\ninfo: {}".format(
            name, version, target.info
        )


class DBBuilder:
    pass
