"""
This dataset is manually created.
The content simply consists of integers in seqential order
    train:
        tfrecord 00001 contains integers from [1, ..., 9]
        tfrecord 00002 contains integers from [11, ..., 19]
        ...
    test:
        same as train but the sign is the opposite
        i.e. tfrecord 00001 contains [-1, ..., -9], and so on.
"""

import pathlib
import tensorflow as tf

_FEATS = {"id": tf.io.FixedLenFeature([], tf.int64)}
TRAIN_TFRS = sorted([str(f) for f in pathlib.Path(__file__).parent.glob("*train*")])
TEST_TFRS = sorted([str(f) for f in pathlib.Path(__file__).parent.glob("*test*")])


def yield_example(fro: int, to: int):
    step = 1 if to > fro else -1
    for i in range(fro, to, step):
        value = tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
        yield tf.train.Example(
            features=tf.train.Features(feature={"id": value})
        )


def create_sample_records():
    path = pathlib.Path(__file__).parent

    train_fmt = "sample-train.tfrecord-{:05d}-of-{:05d}"
    test_fmt = "sample-test.tfrecord-{:05d}-of-{:05d}"

    for file_index in range(1, 6):
        file = str(path.joinpath(train_fmt.format(file_index, 5)))
        with tf.io.TFRecordWriter(file) as writer:
            start = (file_index - 1) * 10
            end = (file_index - 1) * 10 + 10
            for example in yield_example(start, end):
                writer.write(example.SerializeToString())

    for file_index in range(1, 6):
        file = str(path.joinpath(test_fmt.format(file_index, 5)))
        with tf.io.TFRecordWriter(file) as writer:
            start = -1 * (file_index - 1) * 10
            end = -1 * (file_index - 1) * 10 - 10
            for example in yield_example(start, end):
                writer.write(example.SerializeToString())


def parser(example):
    return tf.io.parse_single_example(example, features=_FEATS)


if __name__ == "__main__":
    create_sample_records()
