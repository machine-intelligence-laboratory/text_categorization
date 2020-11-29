import logging
import os
import shutil
import tempfile
import uuid

import yaml

from ap.utils.general import batch_names, ensure_directory
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class NoTranslationException(Exception):
    pass


class ModelDataManager:
    MAX_FILE_SIZE = 512 * 1024 ^ 2
    BATCH_SIZE = 10000

    def __init__(self, data_dir, train_conf, bpe_models):
        """
        Создает дата менеджер.

        Parameters
        ----------
        data_dir - директория для хранения данных
        """
        self._vw = VowpalWabbitBPE(bpe_models)

        self._train_conf = train_conf

        self._data_dir = data_dir
        self._batches_dir = ensure_directory(os.path.join(data_dir, "batches"))
        self._new_batches_dir = ensure_directory(os.path.join(data_dir, "batches_new"))

        self._vw_dir = ensure_directory(os.path.join(data_dir, "vw"))
        self._new_vw_dir = ensure_directory(os.path.join(data_dir, "vw_new"))

        self._current_vw_name = os.path.join(self._new_vw_dir, "actual.txt")

        self._class_ids_path = os.path.join(data_dir, "classes.yaml")
        with open(self._class_ids_path, "r") as f:
            self._class_ids = yaml.safe_load(f)

        self._new_class_ids_path = os.path.join(data_dir, "classes_new.yaml")
        if os.path.exists(self._new_class_ids_path):
            with open(self._new_class_ids_path, "r") as f:
                self._new_class_ids = yaml.safe_load(f)
        else:
            self._new_class_ids = {clsid: val for clsid, val in self._class_ids.items()}

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

            new_batch_vectorizer = artm.BatchVectorizer(
                data_path=os.path.join(self._vw_dir, file),
                data_format="vowpal_wabbit",
                batch_size=self.BATCH_SIZE,
                target_folder=self._new_batches_dir,
            )

            self._update_dictionary(new_batch_vectorizer.dictionary)

            self._merge_batches()

        logging.info("Creating batch vectorizer")
        return artm.BatchVectorizer(data_path=self._batches_dir, data_format="batches")

    def _merge_batches(self):
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
                batch_names(os.path.splitext(max(old_batches))[0], len(new_batches)),
            ):
                shutil.move(
                    os.path.join(self._new_batches_dir, new_batch),
                    os.path.join(self._batches_dir, f"{new_batch_name}.batch"),
                )

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

    def write_new_docs(self, docs):

        if not all(
            [
                any([f"@{lang}" in self._class_ids for lang in doc])
                for doc in docs.values()
            ]
        ):
            raise NoTranslationException()

        self._update_classes({lang for doc in docs.values() for lang in doc})
        data_file = self.get_current_vw()
        with open(data_file, "a" if os.path.exists(data_file) else "w") as f:
            self._vw.save_docs(f, docs)

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

    def _update_classes(self, new_classes):
        for cls in new_classes:
            self._new_class_ids[f"@{cls}"] = 1

        with open(self._new_class_ids_path, "w") as f:
            yaml.dump(self._new_class_ids, f)

    def _update_dictionary(self, new_dictionary):
        with tempfile.TemporaryDirectory(dir=self._data_dir) as tmp_dir:
            new_dict_dir = ensure_directory(os.path.join(tmp_dir, "new_dicts"))
            old_dict_dir = ensure_directory(os.path.join(tmp_dir, "old_dicts"))

            self._decompose_dicts(
                new_dict_dir,
                self._new_class_ids,
                new_dictionary,
                max_dictionary_size=self._train_conf["max_dictionary_size"],
            )
            self._decompose_dicts(old_dict_dir, self._class_ids, self.dictionary)

            for cls_id in self._class_ids:
                shutil.copy(
                    os.path.join(old_dict_dir, f"{cls_id[1:]}.txt"),
                    os.path.join(new_dict_dir, f"{cls_id[1:]}.txt"),
                )

            res = []
            for cls_id in self._new_class_ids:
                with open(os.path.join(new_dict_dir, f"{cls_id[1:]}.txt")) as f:
                    res.extend(f.readlines()[2:] if len(res) > 0 else f.readlines())

            with open(os.path.join(self._data_dir, "dictionary.txt"), "w") as f:
                f.write("".join(res))

            self._class_ids = {clsid: val for clsid, val in self._new_class_ids.items()}

    def _decompose_dicts(self, dir, cls_ids, dictionary, max_dictionary_size=None):
        for cls_id in cls_ids:
            filtered = dictionary
            inplace = False
            for other_id in cls_ids:
                if other_id != cls_id:
                    filtered = filtered.filter(
                        class_id=other_id,
                        max_df_rate=0.4,
                        min_df_rate=0.5,
                        inplace=inplace,
                    )
                    inplace = True
            if max_dictionary_size is not None:
                filtered.filter(max_dictionary_size=max_dictionary_size)
            filtered.save_text(os.path.join(dir, f"{cls_id[1:]}.txt"))
