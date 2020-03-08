import os
import re
import logging
from concurrent import futures
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List

import tensorflow as tf

from MLBOX.Database.builtin import BUILT_INS
from MLBOX.Database.core.parsers import ParserFMT

OUTPUT_PARALLEL_CALL = 8
OUTPUT_BUFFER_TO_BATCH_RATIO = 16
TRAIN_TFRECORD_PATTERN = r".+-train.tfrecord-\d{5}-of-\d{5}"
TRAIN_TFRECORD_FMT = "{}-train.tfrecords-{:05d}-of-{:05d}"
TEST_TFRECORD_PATTERN = r".+-test.tfrecord-\d{5}-of-\d{5}"
TEST_TFRECORD_FMT = "{}-test.tfrecords-{:05d}-of-{:05d}"


def _fullpath_listdir(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


class _TFRWriter(Thread):

    def __init__(self, file, parse_fn):
        self._queue = Queue()
        self._file = str(file)
        self._parse_fn = parse_fn
        self._success_cnt = 0
        self._err_cnt = 0

    @property
    def input_queue(self):
        return self._queue

    @property
    def success(self) -> int:
        return self._success_cnt

    @property
    def error(self) -> int:
        return self._err_cnt

    def run(self):
        with tf.io.TFRecordWriter(self._file) as writer:
            while True:
                target = self._queue.get(block=True)
                if target is None:
                    if self.success == 0 and self.error == 0:
                        msg = "File {} receive empty item."
                        logging.warning(msg.format(self._file))
                    break
                try:
                    example = self._parse_fn(target)
                    writer.write(example.SerializeToString())
                except Exception as e:
                    msg = "Error while writing {}: {}."
                    logging.exception(msg.format(target, str(e)))
                    self._err_cnt += 1
                else:
                    self._success_cnt += 1


class Dataset:

    def __init__(self, tfrecords: List[str], parser: ParserFMT):
        """Create a dataset with tfrecord files and parser"""
        tfrecords = list(tfrecords)
        if not tfrecords:
            raise ValueError("Got empty tfrecords")
        self._files = tfrecords
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
            self._parser.parse_example,
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
            if built_in.name == name.lower():
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
            parser_cls = target.parser
            parser = parser_cls()

        loc = Path(target.location).joinpath(target.name).joinpath(version)
        self.load(directory=str(loc), parser=parser)

        self._info = "Database {}(ver {})\ninfo: {}".format(
            name, version, target.info
        )


class DBuilder:

    def __init__(self, name: str, parser: ParserFMT):
        """Create a builder object for building Database

        Args:
            name (str): the name of the database
            parser (ParserFMT): the parser defining the format of database
        """
        self._name = str(name)
        self._parser = parser

    def build_tfrecords(
            self, generator, output_dir: str, split: str,
            num_of_tfrecords: int = 255
            ):
        """Create training set from generator

        A single data point yield by generator is a dictionary,
        containing items specify by parser.

        Args:
            generator: the generator yields required data
            output_dir: the directory for storing tfrecord files
            split: be either "train" or "test"
            num_of_tfrecords:
                number of .tfrecord files created.
                data will be randomly distritubed over all files by hash.
        """
        out_dir = Path(output_dir)
        if not out_dir.is_dir():
            msg = "Invalid output directory: {}"
            raise ValueError(msg.format(output_dir))

        if split.lower() == "train":
            file_fmt = TRAIN_TFRECORD_FMT
        elif split.lower() == "test":
            file_fmt = TEST_TFRECORD_FMT
        else:
            msg = "Unrecognized split: {}; Must be 'train' or 'test'"
            raise ValueError(msg.format(split))

        if num_of_tfrecords > 1024:
            msg = "num_of_tfrecords too large, got {}. Suggest set it to < 255"
            raise ValueError(msg.format(num_of_tfrecords))

        success_sum = 0
        err_sum = 0
        threads = []
        for index in range(num_of_tfrecords):
            file = file_fmt.format(self._name, index, num_of_tfrecords)
            thread = _TFRWriter(file)
            thread.start()
            threads.append(thread)

        for item in generator:
            index = hash(item) % num_of_tfrecords
            threads[index].input_queue.put(item)

        for thread in threads:
            thread.input_queue.put(None)

        for thread in threads:
            thread.join()
            success_sum += thread.success
            err_sum += thread.error

        logging.info(' -- takes %s seconds' % (time.time() - start_time))
        logging.info(' -- %d success/%d errors' % (success_sum, err_sum))
        logging.info(' -- see log file')
