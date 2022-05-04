"""
Модуль для поддержания работы с данными
"""

import logging
import os
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

        Args:
            data_dir (str): директория для хранения данных
            experiment_config (dict): конфиг для обучения модели
        """
        self._config_path = experiment_config
        with open(self._config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self._data_dir = data_dir

        with open(self.config['rubrics_train']) as file:
            self.train_grnti: typing.Dict[str, str] = json.load(file)

        self.train_path = self.config["train_vw_path"]

        path_experiment = Path(self.config["path_experiment"])
        path_experiment.mkdir(parents=True, exist_ok=True)
        path_train_data = path_experiment.joinpath('train_data')
        self._path_to_batches = path_train_data.joinpath('batches_balanced')
        self._path_balanced_train = path_train_data.joinpath('train_balanced.txt')
        self._path_batches_wiki = self.config.get("path_wiki_train_batches", None)
        self._balancing_modality = self.config.get("balancing_modality", 'GRNTI')

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
        num_rubric = set(self.train_grnti.values())
        logging.info('Balanced learning is used: at each epoch ' +
                     'rubric-balanced documents are sampled from the training data.')
        logging.info(f'Each epoch uses {self.average_rubric_size} documents ' +
                     f'for each of {num_rubric} rubrics.')

        set_metric('average_rubric_size', self.average_rubric_size)
        set_metric('num_rubric', num_rubric)

        self.update_ds_metrics()
        # self._class_ids_path = os.path.join(data_dir, "classes.yaml")
        # with open(self._class_ids_path, "r") as file:
        #     self._class_ids = yaml.safe_load(file)

        # self._new_class_ids_path = os.path.join(data_dir, "classes_new.yaml")
        # if os.path.exists(self._new_class_ids_path):
        #     with open(self._new_class_ids_path, "r") as file:
        #         self._new_class_ids = yaml.safe_load(file)
        # else:
        #     self._new_class_ids = {class_id: val for class_id, val in self._class_ids.items()}

    def update_ds_metrics(self):
        set_metric('train_size_bytes', os.path.getsize(self.train_path))
        with open(self.train_path, encoding='utf-8') as file:
            train_vw = file.readlines()
            set_metric('train_size_docs', len(train_vw))

    def load_train_data(self):
        with open(self.train_path, encoding='utf-8') as file:
            train_vw = file.readlines()

        self.train_dict = {line.split()[0]: line for line in train_vw}

        docs_of_rubrics = {rubric: [] for rubric in set(self.train_grnti.values())}
        for doc_id, rubric in self.train_grnti.items():
            if doc_id in self.train_dict:
                docs_of_rubrics[rubric].append(doc_id)

        self._docs_of_rubrics: typing.Dict[str, list] = docs_of_rubrics

    def _generate_vw_file_balanced_by_rubric(self):
        """
        Генерирует vw файл, где данные сбалансирваны по рубрикам ГРНТИ.

        Возвращает balance_doc_ids — список идентификаторов документов, сбалансированных по рубрикам.
        Документы всех рубрик встречаются в balance_doc_ids одинаковое количество раз, равное среднему размеру рубрики.

        # TODO: а это точно не вызывает проблем?
        Функция изменяет self.train_dict, умноженая счетчики токенов на
        количество вхождений id документа в balance_doc_ids.
        """
        average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
        # balanced_doc_ids = []
        with open(self._path_balanced_train, 'w') as file:
            for rubric in set(self.train_grnti.values()):
                doc_ids_rubric = np.random.choice(self._docs_of_rubrics[rubric], average_rubric_size)
                # balanced_doc_ids.extend(doc_ids_rubric)

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
                        file.write(new_line + '\n')
                        # self.train_dict[doc_id] = new_line
        # return balanced_doc_ids

    # def _generate_vw_file_balanced_by_rubric(self):
    #     """
    #     Генерирует vw файл, где данные сбалансирваны по рубрикам ГРНТИ.
    #     """
    #     balanced_doc_ids = self._get_balanced_doc_ids()
    #     with open(self._path_balanced_train, 'w') as file:
    #         file.writelines([self.train_dict[doc_id].strip() + '\n'
    #                          for doc_id in balanced_doc_ids])

    def generate_batches_balanced_by_rubric(self):
        """
        Возвращает artm.BatchVectorizer, построенный на сбалансированных батчах.

        Генерирует батчи, в которых документы сбалансированны относительно рубрик ГРНТИ.
        Из всего тренировочного датасета сэмплируются документы так, чтобы
        в обучении на эпохе участвовало одинаковое количество документов каждой рубрики ГРНТИ.
        Количество документов каждой рубрики равно average_rubric_size - среднему размеру рубрики ГРНТИ.

        Если в конфиге для обучения модели self._config присутствует путь
        до батчей, построенных по википедии self._path_batches_wiki, то батчи будут использованы для обучения модели.
        Иначе в обучении будут принимать участие только батчи, сбалансированные относительно рубрик ГРНТИ.

        Возвразает artm.BatchVectorizer, построенный на этих батчах.

        Returns:
            batch_vectorizer (artm.BatchVectorizer): artm.BatchVectorizer, построенный на сбалансированных батчах.
        """
        import artm

        try:
            self._path_to_batches.mkdir(parents=True, exist_ok=True)
            self._generate_vw_file_balanced_by_rubric()

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

    def write_new_docs(self, vw, docs):
        """
        TODO

        Args:
            vw (TODO): TODO
            docs (TODO): TODO
        """
        if not all(
                [
                    any([f"{lang}" in self.class_ids for lang in doc])
                    for doc in docs.values()
                ]
        ):
            raise NoTranslationException()

        vw.save_docs(self.train_path, docs)

    def get_modality_distribution(self) -> typing.Dict[str, int]:
        """
        Возвращает количество документов каждой модальности из self.class_ids для тренировочных данных.

        Если в конфиге для обучения модели self.config передан путь до словаря,
        содержащего количество документов Wikipedia по модальностям, эти данные учитываются для
        оценки всего тренировочного датасета.

        Args:
            modality_distribution_all (dict): словарь, ключ - модальность,
                значение - количество документов с такой модальностью
        """
        with open(self.config["train_vw_path"], encoding='utf-8') as file:
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
            modality_distribution_wiki = dict()
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
        """
        Рекурсивно удаляет все данные, расположенные по пути path.

        Args:
            path (Path): путь до данных, которые нужно удалить
        """
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                self._recursively_unlink(child)
        path.rmdir()

    def update_config(self, config: str):
        """
        TODO

        Args:
            config (TODO): TODO
        """
        self.config = yaml.safe_load(config)
        with open(self._config_path, "w") as file:
            yaml.safe_dump(self.config)
