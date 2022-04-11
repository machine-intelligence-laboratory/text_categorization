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

        Returns:
        """
        self._config_path = experiment_config
        with open(self._config_path, "r") as file:
            self._config = yaml.safe_load(file)

        self._data_dir = data_dir

        self.train_grnti: typing.Dict[str, str] = self._get_rubric_of_train_docs()
        self.train_path = self._config["train_vw_path"]

        path_experiment = Path(self._config["path_experiment"])
        path_experiment.mkdir(parents=True, exist_ok=True)
        path_train_data = path_experiment.joinpath('train_data')
        self._path_to_batches = path_train_data.joinpath('batches_balanced')
        self._path_to_batches.mkdir(parents=True, exist_ok=True)
        self._path_balanced_train = path_train_data.joinpath('train_balanced.txt')
        self._path_batches_wiki = self._config["path_wiki_train_batches"]

        # self._batches_dir = ensure_directory(os.path.join(data_dir, "batches"))
        # self._new_batches_dir = ensure_directory(os.path.join(data_dir, "batches_balanced"))

        # self._current_vw_name = os.path.join(data_dir, "train_balanced.txt")

        # TODO: в добучении
        # старые модальности - вытащить из модели
        # новые - из конфига

        all_modalities_train = {**self._config["MODALITIES_TRAIN"],
                                **self._config["LANGUAGES_TRAIN"]}
        self._class_ids = all_modalities_train

        self.average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
        logging.info('Balanced learning is used: at each epoch' +
                     'rubric-balanced documents are sampled from the training data.')
        logging.info(f'Each epoch uses {self.average_rubric_size} documents ' +
                     f'for each of {self._config["num_rubric"]} rubrics.')
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
        Возвращает словарь, где ключ - id документа, значение - номер рубрики ГРНТИ этого документа.
        Рубрики не включают рубрику "нет".

        Returns:
            train_grnti (dict): словарь, где ключ - id документа, значение - номер рубрики ГРНТИ этого документа.
        """
        with open(self._config["path_articles_rubrics_train_grnti"]) as file:
            articles_grnti_with_no = json.load(file)
        with open(self._config["path_elib_train_rubrics_grnti"]) as file:
            elib_grnti_to_fix_with_no = json.load(file)
        with open(self._config["path_grnti_mapping"]) as file:
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
        Создаёт тренировочные данные, сбалансированные относительно рубрик ГРНТИ.

        Возвращает balance_doc_ids — список идентификаторов документов, сбалансированных по рубрикам.
        Документы всех рубрик встречаются в balance_doc_ids одинаковое количество раз, равное среднему размеру рубрики.
        # TODO: а это точно не вызывает проблем?
        Функция изменяет self.train_dict, умноженая счетчики токенов на
        количество вхождений id документа в balance_doc_ids.

        Returns:
            balanced_doc_ids (list): список id документов, сбанасированный относительно рубрик ГРНТИ
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

    def _get_balanced_doc_ids_with_augmentation(self) -> list:
        """
        Создаёт тренировочные данные с применение аугментаци, сбалансированные относительно рубрик ГРНТИ.

        If the rubric size is larger than the average rubric size, a sample is taken
        equal to the average rubric size.
        If the size of the heading is less than the average rubric size,
        all possible documents of rubric are taken; artificial documents are also generated by
        combining the two documents in a ratio of 1 to experiment_config.aug_proportion.

        Returns:
            balanced_doc_ids (list): список id документов, сбанасированный относительно рубрик ГРНТИ
        """
        average_rubric_size = int(len(self.train_grnti) / len(set(self.train_grnti.values())))
        balanced_doc_ids = []
        for rubric in set(self.train_grnti.values()):
            if len(self._docs_of_rubrics[rubric]) >= average_rubric_size:
                doc_ids_rubric = np.random.choice(self._docs_of_rubrics[rubric],
                                                  average_rubric_size)
                balanced_doc_ids.extend(doc_ids_rubric)
            else:
                # все возможные уникальные пары айди
                doc_id_pair_list = list(itertools.combinations(self._docs_of_rubrics[rubric], 2))
                doc_id_pair_list_indexes = list(
                    np.random.choice(len(doc_id_pair_list),
                                     average_rubric_size - len(self._docs_of_rubrics[rubric]))
                )
                doc_id_pair_list = [doc_id_pair_list[i] for i in doc_id_pair_list_indexes]
                doc_id_unique_list = []

                # для каждой пары - новый уникальный айди,
                # новая статья как сумма старых и запись в train_dict
                for doc_id_pair in doc_id_pair_list:
                    doc_id_unique = '_'.join([doc_id_pair[0], doc_id_pair[1]])
                    doc_id_unique_list.append(doc_id_unique)
                    line_1 = self.train_dict[doc_id_pair[0]]
                    line_2 = self.train_dict[doc_id_pair[1]]
                    line_unique_dict = dict()
                    for line_lang in line_1.split(' |@')[1:-1]:
                        lang = line_lang.split()[0]
                        line_lang_dict = {
                            token_and_count.split(':')[0]: token_and_count.split(':')[1]
                            for token_and_count in line_lang.split()[1:]
                        }
                        new_line = ' '.join([lang] + [':'.join([token, count])
                                                      for token, count in line_lang_dict.items()])
                        line_unique_dict[lang] = new_line
                    for line_lang in line_2.split(' |@')[1:-1]:
                        lang = line_lang.split()[0]
                        if lang not in line_unique_dict:
                            line_lang_dict = {
                                token_and_count.split(':')[0]: token_and_count.split(':')[1]
                                for token_and_count in line_lang.split()[1:]
                            }
                            new_line = ' '.join([lang] + [':'.join([token, count])
                                                          for token, count in line_lang_dict.items()])
                            line_unique_dict[lang] = new_line
                        else:
                            line_lang_dict = {token_and_count.split(':')[0]: str(
                                int(self._config["aug_proportion"] *
                                    int(token_and_count.split(':')[1])))
                                for token_and_count in line_lang.split()[1:]}
                            new_line = ' '.join([':'.join([token, count])
                                                 for token, count in line_lang_dict.items()])
                            line_unique_dict[lang] += ' ' + ' '.join(new_line.split())
                    grnti_rubric = line_1.split(' |@')[-1].split()[1].split(':')[0]
                    line_unique_dict['GRNTI'] = 'GRNTI ' + f'{grnti_rubric}:10'
                    line_unique = ' |@'.join([doc_id_unique] + list(line_unique_dict.values()))
                    self.train_dict[doc_id_unique] = line_unique
                doc_ids_rubric = self._docs_of_rubrics[rubric] + list(np.random.choice(
                    doc_id_unique_list, average_rubric_size - len(self._docs_of_rubrics[rubric])))
                balanced_doc_ids.extend(doc_ids_rubric)
        return balanced_doc_ids

    def _generate_vw_file_balanced_by_rubric(self):
        """
        Генерирует vw файл, где данные сбалансирваны по рубрикам ГРНТИ.
        """
        if self._config.get("need_augmentation", None):
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

        Если в конфиге для обучения модели self._config присутствует путь
        до батчей, построенных по википедии self._path_batches_wiki, то батчи будут использованы для обучения модели.
        Иначе в обучении будут принимать участие только батчи, сбалансированные относительно рубрик ГРНТИ.

        Возвразает artm.BatchVectorizer, построенный на этих батчах.

        Returns:
            batch_vectorizer (artm.BatchVectorizer): artm.BatchVectorizer, построенный на сбалансированных батчах.
        """
        # TODO: при ошибках добавить
        import artm

        try:
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
    #     return self._current_vw_name
    #
    # # def get_current_vw(self):
    # #     """
    # #     Возвращает текущий Vowpal Wabbit файл.
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
    #     Returns:
    #         (dict) : Словарь class ids с весами.
    #     """
    #     return self._class_ids
    #
    # @property
    # def dictionary(self):
    #     """
    #     Возвращает словарь, которым инициализировалась модель.
    #
    #     Returns:
    #         dictionary (artm.Dictionary) : словарь, которым инициализировалась модель
    #     """
    #     import artm
    #
    #     dictionary = artm.Dictionary("main_dict")
    #     dictionary.load_text(self._config["dictionary_path"])
    #     return dictionary

    def _get_modality_distribution(self) -> typing.Dict[str, int]:
        """
        Возвращает количество документов каждой модальности из self.class_ids для тренировочных данных.

        Если в конфиге для обучения модели self._config передан путь до словаря,
        содержащего количество документов Wikipedia по модальностям, эти данные учитываются для
        оценки всего тренировочного датасета.

        Returns:
            modality_distribution_all (dict): словарь, ключ - модальность,
                значение - количество документов с такой модальностью
        """
        with open(self._config["train_vw_path"]) as file:
            train_data = file.read()
        modality_distribution = {
            mod: train_data.count(f'|@{mod}')
            for mod in self._class_ids
        }

        # add wiki part of train data
        path_modality_distribution_wiki = self._config.get("path_modality_distribution_wiki", None)
        if path_modality_distribution_wiki:
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
        self._config = yaml.safe_load(config)

        with open(self._config_path, "w") as file:
            yaml.safe_dump(self._config)

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
    #
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
