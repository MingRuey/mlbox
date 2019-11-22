import random
from MLBOX.Database.dataset import DataBase

random.seed(84)   # the answers to life the universe and everything


class SimpleSplit:

    def __init__(self, database: DataBase, ratio_for_validation=0.2):
        self._ratio = ratio_for_validation

        train_files_count = int(len(database.files)*(1-self._ratio))

        files = database.files
        random.shuffle(files)

        train_files = files[:train_files_count]
        val_files = files[train_files_count:]

        self._train_db = DataBase(formats=database.formats)
        self._train_db.add_files(train_files)
        self._val_db = DataBase(formats=database.formats)
        self._val_db.add_files(val_files)

    def get_train_dataset(self):
        return self._train_db

    def get_vali_dataset(self):
        return self._val_db
