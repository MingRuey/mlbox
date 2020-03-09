import os
import re
import time
import unittest.mock as mock

import pytest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from MLBOX.Database.core.database import Dataset  # noqa: E402
from MLBOX.Database.core.database import _TFRWriter, DBuilder # noqa: E402
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

    def test_split_with_invalid_ratio(self):
        """split with invalid ratio should raise ValueError"""
        ds = Dataset(tfrecords=TRAIN_TFRS, parser=SampleParser())

        with pytest.raises(ValueError):
            ds1, ds2 = ds.split(ratio=0.0)

        with pytest.raises(ValueError):
            ds1, ds2 = ds.split(ratio=1.0)

    def test_splitted_can_not_be_split(self):
        """Dataset already been splitted should not be splitted anymore"""
        ds = Dataset(tfrecords=TRAIN_TFRS, parser=SampleParser())
        ds1, ds2 = ds.split(ratio=0.2)

        with pytest.raises(RuntimeError):
            _, __ = ds1.split(ratio=0.5)

        with pytest.raises(RuntimeError):
            _, __ = ds2.split(ratio=0.5)

    def test_split_data_count(self):
        """split Dataset should yields correct data count"""
        ds = Dataset(tfrecords=TRAIN_TFRS, parser=SampleParser())
        assert ds.count == 50

        ds1, ds2 = ds.split(ratio=0.2)
        assert ds1.count == 10
        assert ds2.count == 40

        cnt = 0
        for example in ds1.to_tfdataset(1, 1, shuffle_n_batch=1):
            cnt += 1
        assert cnt == 10

        cnt = 0
        for example in ds2.to_tfdataset(2, 3, shuffle_n_batch=1):
            cnt += 1
        assert cnt == 60


class TestDBuilder:

    def test_worker_thread(self, tmp_path):
        """Test worker thread correctly write queue content into file"""
        target = "MLBOX.Database.core.database.tf.io.TFRecordWriter"
        with mock.patch(target) as mock_writer:
            parse_fn = mock.MagicMock(side_effect=lambda x: x)
            file = str(tmp_path)
            thread = _TFRWriter(file=file, parse_fn=parse_fn)
            thread.start()

            inputs = [mock.MagicMock() for _ in range(4)]
            fail = "string lacks SerializeToString method"
            for sample in inputs:
                thread.input_queue.put({"x": sample})
            thread.input_queue.put({"x": fail})
            thread.input_queue.put(None)
            time.sleep(0.1)

            parse_fn.assert_has_calls(
                [mock.call(x=sample) for sample in inputs] + [mock.call(x=fail)]
            )

            mock_writer.assert_called_with(file)
            writer_instance = mock_writer.return_value.__enter__.return_value
            exs = [sample.SerializeToString.return_value for sample in inputs]
            writer_instance.write.assert_has_calls(
                [mock.call(ex) for ex in exs]
            )

            assert thread.success == 4
            assert thread.error == 1


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
