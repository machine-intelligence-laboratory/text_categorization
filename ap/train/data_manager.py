"""
Модуль для поддержания работы с данными
"""

import json
import logging
import os
import tempfile
import typing
import shutil

from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from ap.train.metrics import set_metric
from ap.utils.general import recursively_unlink, batch_names


class NoTranslationException(Exception):
    pass


class ModelDataManager:
    """
    Класс для поддержания работы с данными
    """

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
        np.random.seed(seed=self.config.get('seed', 42))

        self._data_dir = data_dir

        with open(self.config['balancing_rubrics_train']) as file:
            self.rubrics_train: typing.Dict[str, str] = json.load(file)
        self.average_rubric_size = int(len(self.rubrics_train) / len(set(self.rubrics_train.values())))

        self.train_path = self.config["train_vw_path"]
        with open(self.train_path, 'a') as file:
            pass
        self.new_background_path = self.config.get("new_background_path", None)

        path_experiment = Path(self.config["path_experiment"])
        path_experiment.mkdir(parents=True, exist_ok=True)
        with open(path_experiment.joinpath('experiment_config.yml'), 'w') as file:
            yaml.safe_dump(self.config, file)

        path_train_data = path_experiment.joinpath('train_data')
        self._path_to_batches = path_train_data.joinpath('batches_balanced')
        self._path_balanced_train = path_train_data.joinpath('train_balanced.txt')
        self._balancing_modality = self.config.get("balancing_modality", 'GRNTI')
        self._path_batches_wiki = self.config.get("path_wiki_train_batches", None)
        if self._path_batches_wiki is not None:
            Path(self._path_batches_wiki).mkdir(exist_ok=True)
            self.wiki_batches = list(Path(self._path_batches_wiki).iterdir())
            self.wiki_balancing_type = self.config.get('wiki_balancing_type', False)
            if self.wiki_balancing_type == 'avr_rubric_size':
                # // 1000, т.к. в 1 батче Википедии 1000 документов.
                self.wiki_batches_per_epoch = self.average_rubric_size // 1000 + 1
            elif self.wiki_balancing_type == 'wiki_unisize':
                self.wiki_batches_per_epoch = int(len(self.wiki_batches) /
                                               self.config['artm_model_params']["num_collection_passes"])

        all_modalities_train = {**self.config["MODALITIES_TRAIN"],
                                **self.config["LANGUAGES_TRAIN"]}
        self.class_ids = all_modalities_train

        num_rubric = len(set(self.rubrics_train.values()))
        logging.info('Balanced learning is used: at each epoch ' +
                     'rubric-balanced documents are sampled from the training data.')
        logging.info(f'Each epoch uses {self.average_rubric_size} documents ' +
                     f'for each of {num_rubric} rubrics.')

        set_metric('average_rubric_size', self.average_rubric_size)
        set_metric('num_rubric', num_rubric)

        self.update_ds_metrics()

    def update_ds_metrics(self):
        """
        Обновляет метрики о датасете.
        """
        set_metric('train_size_bytes', os.path.getsize(self.train_path))
        with open(self.train_path, encoding='utf-8') as file:
            train_vw = file.readlines()
            set_metric('train_size_docs', len(train_vw))

    def generate_background_batches(self):
        import artm
        if self.new_background_path is None or not os.path.exists(self.new_background_path):
            return
        with tempfile.TemporaryDirectory(dir=self._data_dir) as temp_dir:
            batch_vectorizer = artm.BatchVectorizer(data_path=self.new_background_path, data_format='vowpal_wabbit',
                                                target_folder=str(temp_dir), batch_size=20)
            old_batches = os.listdir(self._path_batches_wiki)
            if len(old_batches) == 0:
                for new_batch in os.listdir(temp_dir):
                    shutil.move(
                        os.path.join(temp_dir, new_batch),
                        os.path.join(self._path_batches_wiki, new_batch),
                    )

            else:
                new_batches = sorted(os.listdir(temp_dir))

                for new_batch, new_batch_name in zip(
                        new_batches,
                        batch_names(os.path.splitext(max(old_batches))[0], len(new_batches)),
                ):
                    shutil.move(
                        os.path.join(temp_dir, new_batch),
                        os.path.join(self._path_batches_wiki, f"{new_batch_name}.batch"),
                    )

    def load_train_data(self):
        """
        Загружает тренировочные данные.

        Создает два атрибута:
            - self.train_docs - словарь, где по doc_id содержиться документ в Vowpal Wabbit формате
            - self._docs_of_rubrics - словарь, где по рубрике хранится
                список всех doc_id с такой рубрикой из self.rubrics_train.
        """
        with open(self.train_path, encoding='utf-8') as file:
            train_vw = file.readlines()

        self.train_docs = {line.split()[0]: line for line in train_vw}

        docs_of_rubrics = {rubric: [] for rubric in set(self.rubrics_train.values())}
        for doc_id, rubric in self.rubrics_train.items():
            if doc_id in self.train_docs:
                docs_of_rubrics[rubric].append(doc_id)

        self._docs_of_rubrics: typing.Dict[str, list] = docs_of_rubrics

    def _generate_vw_file_balanced_by_rubric(self):
        """
        Генерирует vw файл, где данные сбалансирваны по рубрикам из self.rubrics_train.

        Возвращает balance_doc_ids — список идентификаторов документов, сбалансированных по рубрикам.
        Документы всех рубрик встречаются в balance_doc_ids одинаковое количество раз, равное среднему размеру рубрики.

        Функция изменяет vw-документы, умноженая счетчики токенов на
        количество вхождений id документа в doc_ids_rubric.
        """
        with open(self._path_balanced_train, 'w') as file:
            for rubric in set(self.rubrics_train.values()):
                doc_ids_rubric = np.random.choice(self._docs_of_rubrics[rubric], self.average_rubric_size)

                doc_ids_count = Counter(doc_ids_rubric)
                for doc_id, count in doc_ids_count.items():
                    if count > 1:
                        new_line_dict = {}
                        for line_lang in self.train_docs[doc_id].split(' |@')[1:]:
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

        Возвращает artm.BatchVectorizer, построенный на этих батчах.

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

            if self._path_batches_wiki is not None:
                if self.wiki_balancing_type in ('avr_rubric_size', 'wiki_unisize'):
                    wiki_batch_subsample = np.random.choice(
                        list(Path(self._path_batches_wiki).iterdir()), self.wiki_batches_per_epoch)
                    for batch in wiki_batch_subsample:
                        shutil.copy(batch, self._path_to_batches)
                    batch_vectorizer = artm.BatchVectorizer(
                        data_path=str(self._path_to_batches)
                    )
                else:
                    batch_vectorizer = artm.BatchVectorizer(
                        data_path=[str(self._path_to_batches), self._path_batches_wiki],
                        data_weight=[1, 1]
                    )
                logging.info('Built batches with wiki')
            else:
                batch_vectorizer = artm.BatchVectorizer(
                    data_path=str(self._path_to_batches)
                )
                logging.info('Built batches without wiki')
            return batch_vectorizer
        except Exception as e:
            logging.exception(e)
            raise e

    def write_new_docs(self, vw, docs: typing.Dict[str, typing.Dict[str, str]]):
        """
        Сохраняет документы.

        Args:
            vw (VowpalWabbitBPE): объект класса VowpalWabbitBPE для сохранения VW файлов.
            docs (dict): документы
        """
        if not all(
                [
                    any([f"{lang}" in self.class_ids for lang in doc])
                    for doc in docs.values()
                ]
        ):
            raise NoTranslationException()

        background, rubrics = {}, {}
        for idx, doc in docs.items():
            (background, rubrics)['UDK' in doc and 'GRNTI' in doc][idx] = doc

        if len(rubrics) > 0:
            vw.save_docs(self.train_path, rubrics)

        if len(background) > 0:
            vw.save_docs(self.new_background_path, background)

    def get_modality_distribution(self) -> typing.Dict[str, int]:
        """
        Возвращает количество документов каждой модальности из self.class_ids для тренировочных данных.

        Если в конфиге для обучения модели self.config передан путь до словаря,
        содержащего количество документов Wikipedia по модальностям, эти данные учитываются для
        оценки всего тренировочного датасета.

        Returns:
            modality_distribution_all (dict): словарь, ключ это модальность,
            значение это количество документов с такой модальностью
        """
        with open(self.config["train_vw_path"], encoding='utf-8') as file:
            train_data = file.read()
        modality_distribution = {
            mod: train_data.count(f'|@{mod}')
            for mod in self.class_ids
        }

        # add wiki part of train data
        path_modality_distribution_wiki = self.config.get("path_modality_distribution_wiki", None)
        if self._path_batches_wiki is not None and path_modality_distribution_wiki is not None:
            with open(path_modality_distribution_wiki) as file:
                modality_distribution_wiki = yaml.load(file)
            logging.info("Training data includes Wikipedia articles.")
        else:
            modality_distribution_wiki = {}
            logging.info("Training data DOES NOT includes Wikipedia articles.")

        modality_distribution_all = {}
        for mod in modality_distribution:
            modality_distribution_all[mod] = modality_distribution[mod]
        for mod in modality_distribution_wiki:
            if mod in modality_distribution_all:
                modality_distribution_all[mod] += modality_distribution_wiki[mod]
            else:
                modality_distribution_all[mod] = modality_distribution_wiki[mod]

        return modality_distribution_all

    def update_config(self, config: str):
        """
        Обновляет конфиг, хранящийся по пути self._config_path.

        Args:
            config: конфиг обучаемой модели
        """
        self.config = yaml.safe_load(config)
        with open(self._config_path, "w"):
            yaml.safe_dump(self.config)
