import datetime
import logging
import os
import typing

from ap.topic_model.v1.TopicModelTrain_pb2 import StartTrainTopicModelRequest
from ap.train.data_manager import ModelDataManager
from ap.utils.general import ensure_directory


class ModelTrainer:
    def __init__(
        self,
        data_manager: ModelDataManager,
        conf: typing.Dict[str, typing.Any],
        models_dir: str,
    ):
        """
        Initialize a model trainer.

        Parameters
        ----------
        data_manager - data manager
        conf - training configuration dict
        models_dir - a path to store new models
        """
        self._conf = conf
        self._models_dir = ensure_directory(models_dir)
        self._data_manager = data_manager

    def train_model(self, train_type: StartTrainTopicModelRequest.TrainType):
        """
        Train model synchronously. Save trained model to a new subfolder of self._models_dir.

        Parameters
        ----------
        train_type - full for full train from scratch, update to get the latest model and train it.
        """
        import artm

        model_name = self.generate_model_name()
        batch_vectorizer = self._data_manager.prepare_batches()

        current_models = os.listdir(self._models_dir)
        if (
            train_type == StartTrainTopicModelRequest.TrainType.FULL
            or len(current_models) == 0
        ):
            logging.info("Start full training")

            model = artm.ARTM(
                num_topics=self._conf["num_topics"],
                theta_columns_naming="title",
                class_ids=self._data_manager.class_ids,
                cache_theta=True,
                show_progress_bars=True,
                num_processors=8,
                regularizers=[
                    artm.DecorrelatorPhiRegularizer(
                        gamma=self._conf["gamma"], tau=self._conf["tau"]
                    )
                ],
                dictionary=self._data_manager.dictionary,
            )
            model.scores.add(artm.PerplexityScore(name="PerplexityScore"))
            num_epochs = self._conf["num_epochs_full"]
        else:
            last_model = max(current_models)
            logging.info("Start training based on %s model", last_model)

            model = artm.load_artm_model(os.path.join(self._models_dir, last_model))
            num_epochs = self._conf["num_epochs_update"]

        model.fit_offline(
            batch_vectorizer=batch_vectorizer, num_collection_passes=num_epochs
        )
        model.dump_artm_model(os.path.join(self._models_dir, model_name))

    def generate_model_name(self) -> str:
        """
        Генерирует новое имя для модели.

        Returns
        -------
        Новое имя для модели
        """
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
