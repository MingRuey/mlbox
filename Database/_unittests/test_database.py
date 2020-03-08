import os
import re
import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from MLBOX.Database.core.database import Dataset  # noqa: E402
from MLBOX.Database.core.parsers import ParserFMT  # noqa: E402

from MLBOX.Database._unittests.sample_dataset.create_sample import parser  # noqa: E402
from MLBOX.Database._unittests.sample_dataset.create_sample import TRAIN_TFRS, TEST_TFRS  # noqa: E402


class SampleParser:

    def parse_example(self, example):
        return parser(example)


class TestDataset:

    def test_to_tfdataset(self):
        """The returned tf.data.dataset should match content of tfrecord"""
        ds = Dataset(tfrecords=TRAIN_TFRS, parser=SampleParser())
        ds = ds.to_tfdataset(1, 1)

        examples = [int(ex["id"].numpy()) for ex in ds]
        assert examples != [i for i in range(50)]  # items has benn shuffled
        assert set(examples) == set(i for i in range(50))

    def test_to_tfdataset_order(self):
        """Shuffle_n_batch = 1 should not shuffle anything"""
        ds = Dataset(tfrecords=TRAIN_TFRS, parser=SampleParser())
        ds = ds.to_tfdataset(1, 1, shuffle_n_batch=1)

        examples = [int(ex["id"].numpy()) for ex in ds]
        assert examples == [i for i in range(50)]

    def test_slicing(self):
        """Test the slicing behavior of the returned dataset"""
        pass

    def test_split(self):
        """Test splitting behavior of the Dataset"""
        pass

    def test_parser(self):
        """Test the given parser is acutally work on the returned dataset"""
        pass


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
