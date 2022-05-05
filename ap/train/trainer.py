import datetime
import logging
import math
import os
import typing

from pathlib import Path
from time import sleep

from prometheus_client import Gauge, start_http_server
from tqdm import tqdm
from ap.topic_model.v1.TopicModelTrain_pb2 import StartTrainTopicModelRequest
from ap.train.data_manager import ModelDataManager
from ap.train.metrics import set_metric
from ap.utils.general import ensure_directory, recursively_unlink


class ModelTrainer:
    def __init__(
            self,
            data_manager: ModelDataManager,
            models_dir: str,
    ):
        """
        Initialize a model trainer.

        Args:
            data_manager (ModelDataManager): data manager
            models_dir (str): a path to store new models
        """

        from prometheus_client import Gauge

        self._data_manager = data_manager
        self._models_dir = ensure_directory(models_dir)
        model_name = self.generate_model_name()
        self._path_to_dump_model = Path(self._data_manager.config["path_experiment"]).joinpath(model_name)

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

        set_metric('num_topics', self._data_manager.config["artm_model_params"]["NUM_TOPICS"])
        set_metric('num_bcg_topics', self._data_manager.config["artm_model_params"]["num_bcg_topic"])

    def _create_initial_model(self):
        """
        Creating an initial topic model.
        """
        import artm
        from topicnet.cooking_machine import rel_toolbox_lite

        artm_model_params = self._data_manager.config["artm_model_params"]

        dictionary = artm.Dictionary()
        dictionary.load_text(self._data_manager.config["dictionary_path"])

        background_topic_list = [f'topic_{i}' for i in range(artm_model_params["num_bcg_topic"])]
        subject_topic_list = [
            f'topic_{i}' for i in range(
                artm_model_params["num_bcg_topic"],
                artm_model_params["NUM_TOPICS"] - artm_model_params["num_bcg_topic"])
        ]

        modalities_with_weight = {f'@{lang}': weight
                                  for lang, weight in self._data_manager.class_ids.items()}
        languages_with_weight = {f'@{lang}': weight
                                 for lang, weight in self._data_manager.config["LANGUAGES_TRAIN"].items()}
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

        Returns:
            (artm.scores.Scores): список скоров тематической модели
        """
        return list(self.model.score_tracker.keys())

    @property
    def model_scores_value(self) -> dict:
        """
        Возвращает значения всех скоров тематической модели на текущей эпохе

        Returns:
            scores_value (dict): значения всех скоров тематической модели на текущей эпохе
        """

        scores_value = {score: self.model.score_tracker[score].value[-1]
                        for score in self.model_scores}
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

        Returns:
            характеристики модели
        """

        return self.model.info

    def set_metrics(self):
        """
        Возвращает основную информацию о модели

        Returns:
            info: основная информация о модели
        """
        set_metric('num_modalities', len(self._data_manager.class_ids))

        set_metric('tau_DecorrelatorPhi', self._data_manager.config["artm_model_params"]['tau_DecorrelatorPhi'])
        set_metric('tau_SmoothTheta', self._data_manager.config["artm_model_params"]['tau_SmoothTheta'])
        set_metric('tau_SparseTheta', self._data_manager.config["artm_model_params"]['tau_SparseTheta'])

        for mod, val in self._data_manager.get_modality_distribution().items():
            set_metric(f'modality_distribution_{mod}', val)

    def _train_epoch(self):
        batch_vectorizer = self._data_manager.generate_batches_balanced_by_rubric()
        self.model.fit_offline(batch_vectorizer, num_collection_passes=1)

    def train_model(self, train_type: StartTrainTopicModelRequest.TrainType):
        """
        Функция обучения тематической модели.

        Parameters:
            train_type (StartTrainTopicModelRequest.TrainType):
                full for full train from scratch, update to get the latest model and train it.
        """
        logging.info("Start model training")
        self.set_metrics()
        self._load_model(train_type)
        self._data_manager.load_train_data()
        for epoch in range(self._data_manager.config['artm_model_params']["num_collection_passes"]):
            logging.info(f'Training epoch {epoch}')
            set_metric('training_iteration', epoch + 1)
            self._train_epoch()

            scores_value = self.model_scores_value
            if "PerlexityScore_@ru" in scores_value:
                logging.info(f"PerlexityScore_@ru: {scores_value['PerlexityScore_@ru']}")
                set_metric("perlexity_score_ru",
                           -1 if math.isnan(scores_value['PerlexityScore_@ru']) else scores_value['PerlexityScore_@ru'])
            if "PerlexityScore_@en" in scores_value:
                logging.info(f"PerlexityScore_@en: {scores_value['PerlexityScore_@en']}")
                set_metric("perlexity_score_en",
                           -1 if math.isnan(scores_value['PerlexityScore_@en']) else scores_value['PerlexityScore_@en'])
            if self._path_to_dump_model.exists():
                recursively_unlink(self._path_to_dump_model)
            self.model.dump_artm_model(str(self._path_to_dump_model))
    logging.info("EndTrainTopicModel")

    @staticmethod
    def generate_model_name() -> str:
        """
        Генерирует новое имя для модели.

        Returns:
            Новое имя для модели
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
