"""
Модуль для поддержания работы с данными
"""

import logging
from collections import Counter
import numpy as np
import yaml

import json
import typing
from pathlib import Path

from ap.train.metrics import set_metric
from ap.utils.bpe import load_bpe_models
from ap.utils.general import recursively_unlink
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class NoTranslationException(Exception):
    pass


class ModelDataManager:
    """
    Класс для поддержания работы с данными
    """
    # MAX_FILE_SIZE = 512 * 1024 ^ 2
    # BATCH_SIZE = 10000

    def __init__(self, data_dir: str, experiment_config: str):
        """
        Создает дата менеджер.

        Parameters
        ----------
        data_dir - директория для хранения данных
        """
        # self._train_conf = train_conf
        self._config_path = experiment_config
        with open(self._config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self._data_dir = data_dir

        self.train_grnti: typing.Dict[str, str] = self._get_rubric_of_train_docs()
        self.train_path = self.config["train_vw_path"]
        # self.train_dict: typing.Dict[str, str] = joblib.load(self._config["train_dict_path"])

        path_experiment = Path(self.config["path_experiment"])
        path_experiment.mkdir(parents=True, exist_ok=True)
        path_train_data = path_experiment.joinpath('train_data')
        self._path_to_batches = path_train_data.joinpath('batches_balanced')
        self._path_balanced_train = path_train_data.joinpath('train_balanced.txt')
        self._path_batches_wiki = self.config.get("path_wiki_train_batches", None)



        # self._batches_dir = ensure_directory(os.path.join(data_dir, "batches"))
        # self._new_batches_dir = ensure_directory(os.path.join(data_dir, "batches_balanced"))

        # self._current_vw_name = os.path.join(data_dir, "train_balanced.txt")

        # TODO: в добучении
        # старые модальности - вытащить из модели
        # новые - из конфига

        all_modalities_train = {**self.config["MODALITIES_TRAIN"],
                                **self.config["LANGUAGES_TRAIN"]}
        self.class_ids = all_modalities_train

        self.average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
        logging.info('Balanced learning is used: at each epoch ' +
                     'rubric-balanced documents are sampled from the training data.')
        logging.info(f'Each epoch uses {self.average_rubric_size} documents ' +
                     f'for each of {self.config["num_rubric"]} rubrics.')

        set_metric('average_rubric_size', self.average_rubric_size)
        set_metric('num_rubric', self.config["num_rubric"])

        # self._class_ids_path = os.path.join(data_dir, "classes.yaml")
        # with open(self._class_ids_path, "r") as file:
        #     self._class_ids = yaml.safe_load(file)

        # self._new_class_ids_path = os.path.join(data_dir, "classes_new.yaml")
        # if os.path.exists(self._new_class_ids_path):
        #     with open(self._new_class_ids_path, "r") as file:
        #         self._new_class_ids = yaml.safe_load(file)
        # else:
        #     self._new_class_ids = {class_id: val for class_id, val in self._class_ids.items()}

    def load_train_data(self):
        with open(self.train_path, encoding='utf-8') as file:
            train_vw = file.readlines()

        self.train_dict = {line.split()[0]: line for line in train_vw}

        docs_of_rubrics = {rubric: [] for rubric in set(self.train_grnti.values())}
        for doc_id, rubric in self.train_grnti.items():
            if doc_id in self.train_dict:
                docs_of_rubrics[rubric].append(doc_id)

        self._docs_of_rubrics: typing.Dict[str, list] = docs_of_rubrics

    def _get_rubric_of_train_docs(self):
        """
        Get dict where keys - document ids, value - number of GRNTI rubric of document.

        Do not contains rubric 'нет'.

        Returns
        -------
        train_grnti: dict
            dict where keys - document ids, value - number of GRNTI rubric of document.
        """
        with open(self.config["path_articles_rubrics_train_grnti"]) as file:
            articles_grnti_with_no = json.load(file)
        with open(self.config["path_elib_train_rubrics_grnti"]) as file:
            elib_grnti_to_fix_with_no = json.load(file)
        with open(self.config["path_grnti_mapping"]) as file:
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

    def _get_balanced_doc_ids(self) -> list:
        """
        Create train data balanced by rubrics.

        Returns balanced_doc_ids - list of document ids, balanced by rubric. Documents of
        all rubrics occurs in balanced_doc_ids the same number of times,
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
        average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
        balanced_doc_ids = []
        for rubric in set(self.train_grnti.values()):
            doc_ids_rubric = np.random.choice(self._docs_of_rubrics[rubric], average_rubric_size)
            balanced_doc_ids.extend(doc_ids_rubric)

            doc_ids_count = Counter(doc_ids_rubric)
            for doc_id, count in doc_ids_count.items():
                if count > 1:
                    new_line_dict = dict()
                    for line_lang in self.train_dict[doc_id].split(' |@')[1:]:
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
                    self.train_dict[doc_id] = new_line
        return balanced_doc_ids

    # def _get_balanced_doc_ids_with_augmentation(self) -> list:
    #     """
    #     Create train data balanced by rubrics with augmentation.
    #
    #     If the rubric size is larger than the average rubric size, a sample is taken
    #     equal to the average rubric size.
    #     If the size of the heading is less than the average rubric size,
    #     all possible documents of rubric are taken; artificial documents are also generated by
    #     combining the two documents in a ratio of 1 to experiment_config.aug_proportion.
    #
    #     Returns
    #     -------
    #     balanced_doc_ids: list
    #         list of document ids, balanced by rubric
    #     train_dict: dict
    #         dict where key - document id, value - document in vowpal wabbit format
    #     """
    #     average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
    #     balanced_doc_ids = []
    #     for rubric in set(self.train_grnti.values()):
    #         if len(self._docs_of_rubrics[rubric]) >= average_rubric_size:
    #             doc_ids_rubric = np.random.choice(self._docs_of_rubrics[rubric],
    #                                               average_rubric_size)
    #             balanced_doc_ids.extend(doc_ids_rubric)
    #         else:
    #             # все возможные уникальные пары айди
    #             doc_id_pair_list = list(itertools.combinations(self._docs_of_rubrics[rubric], 2))
    #             doc_id_pair_list_indexes = list(
    #                 np.random.choice(len(doc_id_pair_list),
    #                                  average_rubric_size - len(self._docs_of_rubrics[rubric]))
    #             )
    #             doc_id_pair_list = [doc_id_pair_list[i] for i in doc_id_pair_list_indexes]
    #             doc_id_unique_list = []
    #
    #             # для каждой пары - новый уникальный айди,
    #             # новая статья как сумма старых и запись в train_dict
    #             for doc_id_pair in doc_id_pair_list:
    #                 doc_id_unique = '_'.join([doc_id_pair[0], doc_id_pair[1]])
    #                 doc_id_unique_list.append(doc_id_unique)
    #                 line_1 = self.train_dict[doc_id_pair[0]]
    #                 line_2 = self.train_dict[doc_id_pair[1]]
    #                 line_unique_dict = dict()
    #                 for line_lang in line_1.split(' |@')[1:-1]:
    #                     lang = line_lang.split()[0]
    #                     line_lang_dict = {
    #                         token_and_count.split(':')[0]: token_and_count.split(':')[1]
    #                         for token_and_count in line_lang.split()[1:]
    #                     }
    #                     new_line = ' '.join([lang] + [':'.join([token, count])
    #                                                   for token, count in line_lang_dict.items()])
    #                     line_unique_dict[lang] = new_line
    #                 for line_lang in line_2.split(' |@')[1:-1]:
    #                     lang = line_lang.split()[0]
    #                     if lang not in line_unique_dict:
    #                         line_lang_dict = {
    #                             token_and_count.split(':')[0]: token_and_count.split(':')[1]
    #                             for token_and_count in line_lang.split()[1:]
    #                         }
    #                         new_line = ' '.join([lang] + [':'.join([token, count])
    #                                                       for token, count in line_lang_dict.items()])
    #                         line_unique_dict[lang] = new_line
    #                     else:
    #                         line_lang_dict = {token_and_count.split(':')[0]: str(
    #                             int(self._config["aug_proportion"] *
    #                                 int(token_and_count.split(':')[1])))
    #                             for token_and_count in line_lang.split()[1:]}
    #                         new_line = ' '.join([':'.join([token, count])
    #                                              for token, count in line_lang_dict.items()])
    #                         line_unique_dict[lang] += ' ' + ' '.join(new_line.split())
    #                 grnti_rubric = line_1.split(' |@')[-1].split()[1].split(':')[0]
    #                 line_unique_dict['GRNTI'] = 'GRNTI ' + f'{grnti_rubric}:10'
    #                 line_unique = ' |@'.join([doc_id_unique] + list(line_unique_dict.values()))
    #                 self.train_dict[doc_id_unique] = line_unique
    #             doc_ids_rubric = self._docs_of_rubrics[rubric] + list(np.random.choice(
    #                 doc_id_unique_list, average_rubric_size - len(self._docs_of_rubrics[rubric])))
    #             balanced_doc_ids.extend(doc_ids_rubric)
    #     return balanced_doc_ids
    #
    def _generate_vw_file_balanced_by_rubric(self):
        """
        Генерирует vw файл, где данные сбалансирваны по рубрикам ГРНТИ.

        :return:
        """
        # генерирую сбалансированные данные
        if self.config.get("need_augmentation", None):
            balanced_doc_ids = self._get_balanced_doc_ids_with_augmentation()
        else:
            balanced_doc_ids = self._get_balanced_doc_ids()
        with open(self._path_balanced_train, 'w') as file:
            file.writelines([self.train_dict[doc_id].strip() + '\n'
                             for doc_id in balanced_doc_ids])

    def generate_batches_balanced_by_rubric(self):
        """
        Возвращает artm.BatchVectorizer, построенный на сбалансированных батчах.

        Генерирует батчи, в которых документы сбалансированны относительно рубрик ГРНТИ.
        Из всего тренировочного датасета сэмплируются документы так, чтобы
        в обучении на эпохе участвовало одинаковое количество документов каждой рубрики ГРНТИ.
        Количество документов каждой рубрики равно average_rubric_size - среднему размеру рубрики ГРНТИ.
        Возвразает artm.BatchVectorizer, построенный на этих батчах.

        Returns
        -------
        batch_vectorizer: artm.BatchVectorizer
            artm.BatchVectorizer, построенный на сбалансированных батчах.
        """
        # TODO: при ошибках добавить
        import artm
        try:
            self._path_to_batches.mkdir(parents=True, exist_ok=True)
            self._generate_vw_file_balanced_by_rubric()

            # строю батчи по сбалансированным данным
            batches_list = list(self._path_to_batches.iterdir())
            if batches_list:
                for batch in batches_list:
                    if batch.is_file():
                        batch.unlink()
                    else:
                        recursively_unlink(batch)

            logging.info('Calling artm')

            _ = artm.BatchVectorizer(
                data_path=str(self._path_balanced_train),
                data_format="vowpal_wabbit",
                target_folder=str(self._path_to_batches),
            )

            logging.info('Calling artm 2nd time')

            if self._path_batches_wiki:
                batch_vectorizer = artm.BatchVectorizer(
                    data_path=[self._path_to_batches, self._path_batches_wiki],
                    data_weight=[1, 1]
                )
                logging.info('Built batches with wiki')
            else:
                batch_vectorizer = artm.BatchVectorizer(
                    data_path=self._path_to_batches
                )
                logging.info('Built batches without wiki')
            return batch_vectorizer
        except Exception as e:
            logging.exception(e)
            raise e
    #
    # # def _merge_batches(self):
    # #     logging.info("Merging batches")
    # #     old_batches = os.listdir(self._batches_dir)
    # #     if len(old_batches) == 0:
    # #         for new_batch in os.listdir(self._new_batches_dir):
    # #             shutil.move(
    # #                 os.path.join(self._new_batches_dir, new_batch),
    # #                 os.path.join(self._batches_dir, new_batch),
    # #             )
    # #
    # #     else:
    # #         new_batches = sorted(os.listdir(self._new_batches_dir))
    # #
    # #         for new_batch, new_batch_name in zip(
    # #                 new_batches,
    # #                 batch_names(os.path.splitext(max(old_batches))[0], len(new_batches)),
    # #         ):
    # #             shutil.move(
    # #                 os.path.join(self._new_batches_dir, new_batch),
    # #                 os.path.join(self._batches_dir, f"{new_batch_name}.batch"),
    # #             )
    #
    # # def get_current_vw(self):
    # #     """
    # #     Возвращает текущий VopakWabbit файл.
    # #
    # #     Returns
    # #     -------
    # #     Путь
    # #     """
    # #     if (
    # #         os.path.exists(self._current_vw_name)
    # #         and os.path.getsize(self._current_vw_name) > self.MAX_FILE_SIZE
    # #     ):
    # #         self._close_current()
    # #
    # #     return self._current_vw_name

    def write_new_docs(self, vw, docs):
        if not all(
                [
                    any([f"{lang}" in self.class_ids for lang in doc])
                    for doc in docs.values()
                ]
        ):
            raise NoTranslationException()

        vw.save_docs(self.train_path, docs)
    #
    # # def _close_current(self):
    # #     shutil.move(
    # #         self._current_vw_name, os.path.join(self._new_vw_dir, f"{uuid.uuid4()}.txt")
    # #     )
    #
    # @property
    # def class_ids(self):
    #     """
    #     Словарь class ids с весами.
    #
    #     Returns
    #     -------
    #     Словарь class ids с весами.
    #     """
    #     return self._class_ids
    #
    # @property
    # def dictionary(self):
    #     """
    #     Возвращает artm.Dictionary.
    #
    #     Returns
    #     -------
    #     artm.Dictionary
    #     """
    #     import artm
    #
    #     dictionary = artm.Dictionary("main_dict")
    #     dictionary.load_text(self._config["dictionary_path"])
    #     return dictionary
############



    def _get_modality_distribution(self) -> typing.Dict[str, int]:
        """
        Возвращает количество документов кажджой модальности из self.class_ids для тренировочных данных.

        :return: modality_distribution_all
            словарь, ключ - модальность, значение - количество документов с такой модальностью
        """
        with open(self.config["train_vw_path"]) as file:
            train_data = file.read()
        modality_distribution = {
            mod: train_data.count(f'|@{mod}')
            for mod in self.class_ids
        }

        # add wiki part of train data
        path_modality_distribution_wiki = self.config.get("path_modality_distribution_wiki", None)
        if self._path_batches_wiki and path_modality_distribution_wiki:
            with open(path_modality_distribution_wiki) as file:
                modality_distribution_wiki = yaml.load(file)
            logging.info("Training data includes Wikipedia articles.")
        else:
            logging.info("Training data DOES NOT includes Wikipedia articles.")

        modality_distribution_all = dict()
        for mod in modality_distribution:
            modality_distribution_all[mod] = modality_distribution[mod]
        for mod in modality_distribution_wiki:
            if mod in modality_distribution_all:
                modality_distribution_all[mod] += modality_distribution_wiki[mod]
            else:
                modality_distribution_all[mod] = modality_distribution_wiki[mod]

        return modality_distribution_all

    def _recursively_unlink(self, path: Path):
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                self._recursively_unlink(child)
        path.rmdir()

    # def _update_classes(self, new_classes):
    #     for cls in new_classes:
    #         self._new_class_ids[f"@{cls}"] = 1
    #
    #     with open(self._new_class_ids_path, "w") as file:
    #         yaml.dump(self._new_class_ids, file)

    # def _update_dictionary(self, new_dictionary):
    #     with tempfile.TemporaryDirectory(dir=self._data_dir) as tmp_dir:
    #         new_dict_dir = ensure_directory(os.path.join(tmp_dir, "new_dicts"))
    #         old_dict_dir = ensure_directory(os.path.join(tmp_dir, "old_dicts"))
    #
    #         self._decompose_dicts(
    #             new_dict_dir,
    #             self._new_class_ids,
    #             new_dictionary,
    #             max_dictionary_size=self._config["max_dictionary_size"],
    #         )
    #         self._decompose_dicts(old_dict_dir, self._class_ids, self.dictionary)
    #
    #         for cls_id in self._class_ids:
    #             shutil.copy(
    #                 os.path.join(old_dict_dir, f"{cls_id[1:]}.txt"),
    #                 os.path.join(new_dict_dir, f"{cls_id[1:]}.txt"),
    #             )
    #
    #         res = []
    #         for cls_id in self._new_class_ids:
    #             with open(os.path.join(new_dict_dir, f"{cls_id[1:]}.txt")) as file:
    #                 res.extend(file.readlines()[2:] if len(res) > 0 else file.readlines())
    #
    #         with open(os.path.join(self._data_dir, "dictionary.txt"), "w") as file:
    #             file.write("".join(res))
    #
    #         self._class_ids = {clsid: val for clsid, val in self._new_class_ids.items()}

    # # TODO: @staticmethod ?
    # def _decompose_dicts(self, directory, cls_ids, dictionary, max_dictionary_size=None):
    #     for cls_id in cls_ids:
    #         filtered = dictionary
    #         inplace = False
    #         for other_id in cls_ids:
    #             if other_id != cls_id:
    #                 filtered = filtered.filter(
    #                     class_id=other_id,
    #                     max_df_rate=0.4,
    #                     min_df_rate=0.5,
    #                     inplace=inplace,
    #                 )
    #                 inplace = True
    #         if max_dictionary_size is not None:
    #             filtered.filter(max_dictionary_size=max_dictionary_size)
    #         filtered.save_text(os.path.join(directory, f"{cls_id[1:]}.txt"))

    def update_config(self, config: str):
        self.config = yaml.safe_load(config)

        with open(self._config_path, "w") as file:
            yaml.safe_dump(self.config)
