import pytest

from MLBOX.Database.dataset import DataBase


class TestDatabase:

    def test_merge_database(self):
        """Merge database should merge the internal tfrecord files"""
        pass


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
