import datetime
import logging
import os
import typing

from pathlib import Path
from time import sleep

from prometheus_client import Gauge, start_http_server
from tqdm import tqdm
from ap.topic_model.v1.TopicModelTrain_pb2 import StartTrainTopicModelRequest
from ap.train.data_manager import ModelDataManager
from ap.utils.general import ensure_directory, recursively_unlink

class ModelTrainer:
    def __init__(
            self,
            data_manager: ModelDataManager,
            models_dir: str,
    ):
        from prometheus_client import Gauge
        #
        # """
        # Initialize a model trainer.
        #
        # Parameters
        # ----------
        # data_manager - data manager
        # conf - training configuration dict
        # models_dir - a path to store new models
        # """
        # self._conf = conf
        self._data_manager = data_manager
        #
        #

        self._models_dir = ensure_directory(models_dir)
        model_name = self.generate_model_name()
        self._path_to_dump_model = Path(self._data_manager._config["path_experiment"]).joinpath(model_name)

    def _init_metrics(self):
        # start_http_server(8001, addr='0.0.0.0')
        self._iteration = Gauge('training_iteration', 'Current training iteration')
        self._average_rubric_size = Gauge('average_rubric_size', 'Average rubric size')
        self._num_rubric = Gauge('num_rubric', 'Number of rubrics')

        self._average_rubric_size.set(self._data_manager.average_rubric_size)
        self._num_rubric.set(self._data_manager._config["num_rubric"])

    def _load_model(self, train_type):
        import artm
        current_models = os.listdir(self._models_dir)
        # TODO: для дообучения
        # добавить условие: есть язык не из 100 языков
        # new_modalities = list(config["LANGUAGES_ALL"]).extend(list(config["MODALITIES_TRAIN"]))
        # if not set(new_modalities).issubset(set(self._class_ids))
        if (
                train_type == StartTrainTopicModelRequest.TrainType.FULL
                or len(current_models) == 0
        ):
            logging.info("Start full training")
            self.model = self._create_initial_model()
        else:
            # TODO: загрузить модель для дообучения
            last_model = max(current_models)
            logging.info("Start training based on %s model", last_model)

            self.model = artm.load_artm_model(os.path.join(self._models_dir, last_model))

    def _create_initial_model(self):
        """
        Creating an initial topic model.

        Returns
        -------
        model: artm.ARTM
            initial artm topic model with parameters from experiment_config
        """
        import artm
        from topicnet.cooking_machine import rel_toolbox_lite

        artm_model_params = self._data_manager._config["artm_model_params"]

        dictionary = artm.Dictionary()
        dictionary.load_text(self._data_manager._config["dictionary_path"])

        background_topic_list = [f'topic_{i}' for i in range(artm_model_params["num_bcg_topic"])]
        subject_topic_list = [
            f'topic_{i}' for i in range(
                artm_model_params["num_bcg_topic"],
                artm_model_params["NUM_TOPICS"] - artm_model_params["num_bcg_topic"])
        ]

        modalities_with_weight = {f'@{lang}': weight for lang, weight in self._data_manager.class_ids.items()}
        languages_with_weight = {f'@{lang}': weight for lang, weight in self._data_manager._config["LANGUAGES_TRAIN"].items()}
        model = artm.ARTM(num_topics=artm_model_params["NUM_TOPICS"],
                          theta_columns_naming='title',
                          class_ids=modalities_with_weight,
                          show_progress_bars=True,
                          dictionary=dictionary)

        model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore',
                                                 topic_names=subject_topic_list))
        for lang in model._class_ids:
            model.scores.add(artm.SparsityPhiScore(name=f'SparsityPhiScore_{lang}',
                                                   class_id=lang,
                                                   topic_names=subject_topic_list))
            model.scores.add(artm.PerplexityScore(name=f'PerlexityScore_{lang}',
                                                  class_ids=lang,
                                                  dictionary=dictionary))

        # SmoothTheta
        model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(
                name='SmoothThetaRegularizer',
                tau=artm_model_params["tau_SmoothTheta"],
                topic_names=background_topic_list)
        )
        rel_toolbox_lite.handle_regularizer(
            use_relative_coefficients=True,
            model=model,
            regularizer=model.regularizers['SmoothThetaRegularizer'],
            data_stats=rel_toolbox_lite.count_vocab_size(
                dictionary=dictionary,
                modalities=languages_with_weight)
        )

        # SparseTheta
        model.regularizers.add(
            artm.SmoothSparseThetaRegularizer(
                name='SparseThetaRegularizer',
                tau=artm_model_params["tau_SparseTheta"],
                topic_names=subject_topic_list)
        )
        rel_toolbox_lite.handle_regularizer(
            use_relative_coefficients=True,
            model=model,
            regularizer=model.regularizers['SparseThetaRegularizer'],
            data_stats=rel_toolbox_lite.count_vocab_size(
                dictionary=dictionary,
                modalities=languages_with_weight)
        )

        # DecorrelatorPhi
        model.regularizers.add(
            artm.DecorrelatorPhiRegularizer(
                name='DecorrelatorPhiRegularizer',
                tau=artm_model_params["tau_DecorrelatorPhi"],
                gamma=0, topic_names=subject_topic_list)
        )
        rel_toolbox_lite.handle_regularizer(
            use_relative_coefficients=True,
            model=model, regularizer=model.regularizers['DecorrelatorPhiRegularizer'],
            data_stats=rel_toolbox_lite.count_vocab_size(
                dictionary=dictionary,
                modalities=languages_with_weight)
        )
        return model

    @property
    def model_scores(self):
        """
        Возвращает все скоры тематической модели

        :return:
        artm.scores.Scores
            список скоров тематической модели
        """
        return self.model.scores

    @property
    def model_scores_value(self) -> dict:
        """
        Возвращает значения всех скоров тематической модели на текущей эпохе

        :return:
        scores_value
            значения всех скоров тематической модели на текущей эпохе
        """

        scores_value = {score: self.model.score_tracker[score].value[-1]
                        for score in self.model.scores}
        return scores_value

    @property
    def model_info(self):
        """
        Возвращает характеристики модели

        Очень подробна информация о характеристиках модели:
        - названия тем
        - названия модальностей
        - веса модальностей
        - метрики
        - информация о регуляризаторах
        - другое
        :return:
        """

        return self.model.info

    def model_main_info(self):
        """
        Возвращает основную информацию о модели
        :return:
        """
        info = self._data_manager._config["artm_model_params"]
        info["Модальности"] = self._data_manager._class_ids
        info["need_augmentation"] = self._data_manager._config.get("need_augmentation", False)
        if info["need_augmentation"]:
            info["aug_proportion"] = self._data_manager._config.get("aug_proportion")
        info["metrics_to_calculate"] = self._data_manager._config["metrics_to_calculate"]
        info["num_modalities"] = len(info["Модальности"])
        info["dictionary_path"] = self._data_manager._config["dictionary_path"]

        return info # параметры,

    def _train_epoch(self):
        batch_vectorizer = self._data_manager.generate_batches_balanced_by_rubric()
        self.model.fit_offline(batch_vectorizer, num_collection_passes=1)

    def train_model(self, train_type: StartTrainTopicModelRequest.TrainType):
        """
        # TODO: обновить док-стринг
        Train model synchronously. Save trained model to a new subfolder of self._models_dir.

        Parameters
        ----------
        train_type - full for full train from scratch, update to get the latest model and train it.
        """
        main_info = self.model_main_info()
        # Здесь можно визуализировать основную информацию о модели main_info
        logging.info("Start model training")
        self._init_metrics()
        self._load_model(train_type)
        self._data_manager.load_train_data()
        for epoch in range(self._data_manager._config['artm_model_params']["num_collection_passes"]):
            logging.info('Training epoch %i', epoch)
            self._iteration.set(epoch+1)
            self._train_epoch()

            # тут нужно визуализировать epoch
            scores_value = self.model_scores_value
            # тут можно визуализировать скоры модели scores_value
            if "PerlexityScore_@ru" in scores_value:
                logging.info(f"PerlexityScore_@ru: {scores_value['PerlexityScore_@ru']}")
            if self._path_to_dump_model.exists():
                recursively_unlink(self._path_to_dump_model)
            self.model.dump_artm_model(str(self._path_to_dump_model))

    @staticmethod
    def generate_model_name() -> str:
        """
        Генерирует новое имя для модели.

        Returns
        -------
        Новое имя для модели
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
