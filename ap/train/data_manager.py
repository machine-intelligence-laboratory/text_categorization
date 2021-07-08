import logging
import os
import shutil
import tempfile
import uuid

import yaml
import joblib
import json
import numpy as np
from collections import Counter

from ap.utils.general import batch_names, ensure_directory
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class NoTranslationException(Exception):
    pass


class ModelDataManager:
    MAX_FILE_SIZE = 512 * 1024 ^ 2
    BATCH_SIZE = 10000

    def __init__(self, data_dir, train_conf, rubric_dir):
        """
        Создает дата менеджер.

        Parameters
        ----------
        data_dir - директория для хранения данных
        """
        self._train_conf = train_conf

        self._data_dir = data_dir
        self._batches_dir = ensure_directory(os.path.join(data_dir, "batches"))
        self._new_batches_dir = ensure_directory(os.path.join(data_dir, "batches_balanced"))

        self._current_vw_name = os.path.join(data_dir, "train_balanced.txt")
        
        self._class_ids_path = os.path.join(data_dir, "classes.yaml")
        with open(self._class_ids_path, "r") as f:
            self._class_ids = yaml.safe_load(f)
            
        self._new_class_ids_path = os.path.join(data_dir, "classes_new.yaml")
        if os.path.exists(self._new_class_ids_path):
            with open(self._new_class_ids_path, "r") as f:
                self._new_class_ids = yaml.safe_load(f)
        else:
            self._new_class_ids = {clsid: val for clsid, val in self._class_ids.items()}

        self._vw_dict = joblib.load(os.path.join(data_dir, "train_dict.joblib"), mmap_mode='r+')

        self._rubric_dir = rubric_dir

    def prepare_batches(self):
        """
        Досоздает батчи из новых данных и возвращает батч векторайзер.

        Returns
        -------
        artm.BatchVectorizer
        """
        import artm

        logging.info("Preparing batches")
            
        train_grnti = self.get_rubric_of_train_docs()
        docs_of_rubrics = {rubric: [] for rubric in set(train_grnti.values())}
        
        for doc_id, rubric in train_grnti.items():
            if doc_id in self._vw_dict:
                docs_of_rubrics[rubric].append(doc_id)
                
        balanced_doc_ids, train_dict = self.get_balanced_doc_ids(
                self._vw_dict, train_grnti, docs_of_rubrics
            )
        
        with open(self._current_vw_name, 'w') as file:
            file.writelines([self._vw_dict[doc_id].strip() + '\n'
                             for doc_id in balanced_doc_ids])

        _ = artm.BatchVectorizer(
            data_path=str(self._current_vw_name),
            data_format="vowpal_wabbit",
            target_folder=str(self._new_batches_dir),
        )
        
        logging.info("Creating batch vectorizer")
        return artm.BatchVectorizer(data_path=[self._new_batches_dir, self._batches_dir], data_weight=[1, 1])

    def get_rubric_of_train_docs(self):
        """
        Get dict where keys - document ids, value - numer of GRNTI rubric of document.

        Do not conteins rubric 'нет'.

        Returns
        -------
        train_grnti: dict
            dict where keys - document ids, value - numer of GRNTI rubric of document.
        """
        with open(os.path.join(self._rubric_dir, 'grnti_codes.json')) as file:
            articles_grnti_with_no = json.load(file)
        with open(os.path.join(self._rubric_dir, "elib_train_grnti_codes.json")) as file:
            elib_grnti_to_fix_with_no = json.load(file)
        with open(os.path.join(self._rubric_dir, "grnti_to_number.json")) as file:
            grnti_to_number = json.load(file)

        articles_grnti = {doc_id: rubric
                          for doc_id, rubric in articles_grnti_with_no.items()
                          if rubric != 'нет'}

        elib_grnti = {doc_id[:-len('.txt')]: rubric
                      for doc_id, rubric in elib_grnti_to_fix_with_no.items()
                      if rubric != 'нет'}

        train_grnti = dict()
        for doc_id in articles_grnti:
            rubric = str(grnti_to_number[articles_grnti[doc_id]])
            train_grnti[doc_id] = rubric
        for doc_id in elib_grnti:
            rubric = str(grnti_to_number[elib_grnti[doc_id]])
            train_grnti[doc_id] = rubric
        return train_grnti
    
    def get_balanced_doc_ids(
        self, train_dict, train_grnti, docs_of_rubrics,
    ):
        """
        Create train data balanced by rubrics.

        Returns balanced_doc_ids - list of document ids, balanced by rubric. Documents of
        all rubrics occures in balanced_doc_ids the same number of times,
        equal to average_rubric_size.
        Returns train_dict - dict where key - document id, value - document in
        vowpal wabbit format. Function change train_dict, multiplying token counters
        by number of occurrences of document id in balanced_doc_ids.

        Returns
        -------
        balanced_doc_ids: list
            list of document ids, balanced by rubric
        train_dict: dict
            dict where key - document id, value - document in vowpal wabbit format
        """
        average_rubric_size = int(len(train_grnti) / len(set(train_grnti.values())))
        balanced_doc_ids = []
        for rubric in set(train_grnti.values()):
            doc_ids_rubric = np.random.choice(docs_of_rubrics[rubric], average_rubric_size)
            balanced_doc_ids.extend(doc_ids_rubric)

            doc_ids_count = Counter(doc_ids_rubric)
            for doc_id, count in doc_ids_count.items():
                if count > 1:
                    new_line_dict = dict()
                    for line_lang in train_dict[doc_id].split(' |@')[1:]:
                        lang = line_lang.split()[0]
                        line_lang_dict = {
                            token_with_count.split(':')[0]: count *
                            int(token_with_count.split(':')[1])
                            for token_with_count in line_lang.split()[1:]
                        }
                        new_line_lang = ' '.join([lang] +
                                                 [':'.join([token, str(count)])
                                                  for token, count in line_lang_dict.items()])
                        new_line_dict[lang] = new_line_lang
                    new_line = ' |@'.join([doc_id] + list(new_line_dict.values()))
                    train_dict[doc_id] = new_line
        return balanced_doc_ids, train_dict
    
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

    def write_new_docs(self, vw_writer, docs):

        if not all(
            [
                any([f"@{lang}" in self._class_ids for lang in doc])
                for doc in docs.values()
            ]
        ):
            raise NoTranslationException()

        vw_writer.save_docs(self._vw_dict, docs)

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
