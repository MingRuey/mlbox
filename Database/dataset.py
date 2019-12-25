import time
import pathlib
import yaml
import logging
from concurrent import futures
from functools import partial
from typing import List
import numpy as np
import tensorflow as tf

from MLBOX.Database.formats import DataFormat, TSFORMAT, IMGFORMAT

NUM_OF_TFRECORDS = 255
OUTPUT_TFRECORDS = "tfdatabase_{:0=4}.tfrecord"
OUTPUT_PARALLEL_CALL = 8
OUTPUT_BUFFER_TO_BATCH_RATIO = 16


def _is_dir(directory, name="Directory"):
    path = pathlib.Path(directory)
    if path.is_dir():
        return path

    msg = "Invalid {}, got {}"
    raise ValueError(msg.format(name, directory))


def _is_file(file, name="File"):
    path = pathlib.Path(file)
    if path.is_file():
        return path

    msg = "Invalid {}, got {}"
    raise ValueError(msg.format(name, file))


def _glob_ext(directory, ext, mini=1, maxi=None, name="Directory"):
    path = _is_dir(directory, name=name)
    ext = str(ext)
    if not ext.startswith("."):
        ext = "." + ext

    globs = list(path.glob("*" + ext))
    if mini is not None and len(globs) < mini:
        msg = "{} must have at least {} files with extension {}, found {} under {}"
        raise ValueError(msg.format(name, mini, ext, len(globs), path))

    if maxi is not None and len(globs) > maxi:
        msg = "{} can have at most {} files with extension {}, found {} under {}"
        raise ValueError(msg.format(name, maxi, ext, len(globs), path))

    return globs


class TFRfile(tf.io.TFRecordWriter):

    def __init__(self, tffile, options=None):
        self._file = pathlib.Path(tffile)
        self._options = options
        self._ids = set()
        super().__init__(str(tffile), options)

    def add(self, files, reader):
        """Append the tfexample to the end of the file.

        Args:
            files:
                the targets to add, will treat items as string of file paths.
            reader:
                function takes file path as input and return tf.train.Example

        Return:
            tuple of (successes count, errors count)
        """
        files = list(files)
        success_count = 0
        err_count = 0
        for file in files:
            try:
                tfexample = reader(file)
                super().write(tfexample.SerializeToString())
            except ValueError as e:
                msg = "TFRfile -- Parse {} failed: {}"
                logging.exception(msg.format(file, str(e)))
                err_count += 1
            else:
                success_count += 1
        return success_count, err_count

    def remove(self, tfexample):
        raise NotImplementedError()

    def count(self):
        raise NotImplementedError()

    def get_ids(self, example_id):
        raise NotImplementedError()

    def __contains__(self):
        raise NotImplementedError()


class DataBase:
    """Dataset object holds a pack of tfrecord file"""

    yaml.SafeLoader.add_constructor(u"!DataFormat", DataFormat.from_yaml)

    def __init__(self, formats: DataFormat = None):
        """Initialize a dataset with formats"""
        self._files = []
        self._formats = formats

    @property
    def files(self) -> List[str]:
        """Get the tfreocrd files in dataset"""
        return self._files

    @property
    def file_count(self) -> int:
        """Get the total number of tfrecord files in dataset"""
        return len(self._files)

    @property
    def data_count(self) -> int:
        """Get the total number of data in dataset"""
        # TODO: this can be extremely slow
        cnt = 0
        for example in tf.data.TFRecordDataset(self.files):
            cnt += 1
        return cnt

    @property
    def formats(self) -> DataFormat:
        """Get the format dataset used to save tfrecords"""
        return self._formats

    def add_files(self, files: List[str]):
        """Add one or more tfrecord files

        Args:
            files: string or list of string of file paths
        """
        if isinstance(files, str):
            files = [files]
        valid_files = [str(file) for file in files if self._check_file(file)]
        self._files.extend(valid_files)

    @staticmethod
    def _check_file(file):
        file = pathlib.Path(file)
        if file.is_file() and file.name.endswith(".tfrecord"):
            return True

        msg = "Try to add invalid file into Database: {}"
        logging.warning(msg.format(file))
        return False

    def remove_files(self, files: List[str]):
        """Remove one or more tfrecord files

        Args:
            files: string or list of string of file name
        """
        raise NotImplementedError()

    def load_path(self, directory: str):
        """Init a Dataset with a dir containing tfrecords and metadata

        Load already exist files if meta is found,

        Args:
            directory: the directory that stores the dataset
        """
        # TODO: checkout build_database
        # yaml.SafeLoader.add_constructor(u"!DataFormat", DataFormat.from_yaml)
        # meta = _glob_ext(directory, ".yml", mini=1, maxi=1, name="Config file")
        # with open(str(meta[0]), "r") as f:
        #     fmt = yaml.safe_load(f)
        #     if fmt != self.formats:
        #         msg = "DataBase has format {} but try to load from {}"
        #         raise ValueError(msg.format(self.formats, fmt))

        tfrecords = _glob_ext(directory, ".tfrecord", mini=1, name="TFRecords")
        self.files.extend([str(file) for file in tfrecords])

    def build_database(self, input_dir: str, output_dir: str):
        """Build database from a tfexample generator and a path

        Args:
            input_dir:
                the directory contains data.
                only glob files with matched extension.
            output_dir:
                where to store tfrecord files.
        """
        output_dir = _is_dir(output_dir, "output_dir")

        # TODO: dump doesn't work,
        # subclass of DataFormat cannot be dumped properly
        #
        # with open(str(output_dir.joinpath("config.yml")), "w") as f:
        #     yaml.dump(self.formats, f)

        targets = []
        for ext in self.formats.valid_extensions:
            targets.extend(_glob_ext(input_dir, ext, mini=None, maxi=None))

        start_time = time.time()
        logging.info('Start writing tfrecords')

        def tfr_worker(files, fout):
            fout = pathlib.Path(fout)
            if fout.is_file():
                msg = "File {} already exist"
                logging.warning(msg.format(fout))
                return 0, len(files)

            if len(files) == 0:
                msg = "File {} receive empty image file list"
                logging.warning(msg.format(fout))
                return 0, 0

            with TFRfile(str(fout)) as tfr:
                return tfr.add(files=files, reader=self.formats.to_tfexample)

        success_sum = 0
        err_sum = 0
        with futures.ThreadPoolExecutor(max_workers=16) as exe:
            all_futures = []
            for index, files in enumerate(
                    np.array_split(targets, NUM_OF_TFRECORDS)):

                fout = OUTPUT_TFRECORDS.format(index)
                fout = output_dir.joinpath(fout)
                worker = partial(tfr_worker, fout=fout)
                all_futures.append(exe.submit(worker, files))

            for future in futures.as_completed(all_futures):
                success_count, err_count = future.result()
                success_sum += success_count
                err_sum += err_count

        logging.info(' -- takes %s seconds' % (time.time() - start_time))
        logging.info(' -- %d success/%d errors' % (success_sum, err_sum))
        logging.info(' -- see log file')

    @property
    def parser(self):
        if hasattr(self, "_parser"):
            return self._parser
        raise RuntimeError("Paser not configured: please config parser first")

    def config_parser(self, *args, **kwargs):
        """Configure parser for training

        Args:
            *args, **kwargs: arguments passed to get_parser function
        """
        self._parser = self.formats.get_parser(*args, **kwargs)

    def get_dataset(self, epoch: int, batchsize: int):
        """Get the tf.dataset class used by get_input_tensor"""
        dataset = tf.data.TFRecordDataset(self.files)
        dataset = dataset.map(
            self.parser,
            num_parallel_calls=OUTPUT_PARALLEL_CALL)
        dataset = dataset.shuffle(100 * batchsize)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(batchsize, drop_remainder=True)
        dataset = dataset.prefetch(OUTPUT_BUFFER_TO_BATCH_RATIO)
        return dataset

    def get_input_tensor(self, epoch: int, batchsize: int):
        """Get the input tensor from tfrecord files in databaseObj

        Args:
            epoch: the number of cycle that datasetObj iterates over its files
            batchsize: the number of data per tensor retrieve

        Returns:
            a tuple of data, label
            data:
                dictionary of tensors in the following form
                {'data': content of data, 'dataid': unique identifier of data}
            label:
                tensor of the data label
        """
        dataset = self.get_dataset(epoch, batchsize)
        for data, dataid, label in dataset:
            yield {'data': data, 'dataid': dataid}, label
