import logging
import os
import shutil
import uuid

import yaml

from ap.utils.general import batch_names, ensure_directory


class ModelDataManager:
    MAX_FILE_SIZE = 512 * 1024 ^ 2
    BATCH_SIZE = 10000

    def __init__(self, data_dir):
        """
        Создает дата менеджер.

        Parameters
        ----------
        data_dir - директория для хранения данных
        """
        self._data_dir = data_dir
        self._batches_dir = ensure_directory(os.path.join(data_dir, "batches"))
        self._new_batches_dir = ensure_directory(os.path.join(data_dir, "batches_new"))

        self._vw_dir = ensure_directory(os.path.join(data_dir, "vw"))
        self._new_vw_dir = ensure_directory(os.path.join(data_dir, "vw_new"))

        self._current_vw_name = os.path.join(self._new_vw_dir, "actual.txt")

        with open(os.path.join(data_dir, "classes.yaml"), "r") as f:
            self._class_ids = yaml.safe_load(f)

    def prepare_batches(self):
        """
        Досоздает батчи из новых данных и возвращает батч векторайзер.

        Returns
        -------
        artm.BatchVectorizer
        """
        import artm

        logging.info("Preparing batches")

        if os.path.exists(self._current_vw_name):
            self._close_current()

        for file in os.listdir(self._new_vw_dir):
            shutil.move(
                os.path.join(self._new_vw_dir, file), os.path.join(self._vw_dir, file)
            )
            vw_file = os.path.join(self._vw_dir, file)
            logging.info("Converting %s", vw_file)

            artm.BatchVectorizer(
                data_path=os.path.join(self._vw_dir, file),
                data_format="vowpal_wabbit",
                batch_size=self.BATCH_SIZE,
                target_folder=self._new_batches_dir,
            )

            logging.info("Merging batches")

            old_batches = os.listdir(self._batches_dir)
            if len(old_batches) == 0:
                for new_batch in os.listdir(self._new_batches_dir):
                    shutil.move(
                        os.path.join(self._new_batches_dir, new_batch),
                        os.path.join(self._batches_dir, new_batch),
                    )

            else:
                new_batches = sorted(os.listdir(self._new_batches_dir))

                for new_batch, new_batch_name in zip(
                    new_batches,
                    batch_names(
                        os.path.splitext(max(old_batches))[0], len(new_batches)
                    ),
                ):
                    shutil.move(
                        os.path.join(self._new_batches_dir, new_batch),
                        os.path.join(self._batches_dir, f"{new_batch_name}.batch"),
                    )

        logging.info("Creating batch vectorizer")
        return artm.BatchVectorizer(data_path=self._batches_dir, data_format="batches")

    def get_current_vw(self):
        """
        Возвращает текущий VopakWabbit файл.

        Returns
        -------
        Путь
        """
        if (
            os.path.exists(self._current_vw_name)
            and os.path.getsize(self._current_vw_name) > self.MAX_FILE_SIZE
        ):
            self._close_current()

        return self._current_vw_name

    def _close_current(self):
        shutil.move(
            self._current_vw_name, os.path.join(self._new_vw_dir, f"{uuid.uuid4()}.txt")
        )

    @property
    def class_ids(self):
        """
        Словарь class ids с весами.

        Returns
        -------
        Словарь class ids с весами.
        """
        return self._class_ids

    @property
    def dictionary(self):
        """
        Возвращает artm.Dictionary.

        Returns
        -------
        artm.Dictionary
        """
        import artm

        dictionary = artm.Dictionary("main_dict")
        dictionary.load_text(os.path.join(self._data_dir, "dictionary.txt"))
        return dictionary
