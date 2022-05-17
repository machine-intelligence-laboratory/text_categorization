import concurrent
import logging
import typing

from concurrent import futures
from time import sleep

import click
import grpc
import yaml

from ap.topic_model.v1.TopicModelTrain_pb2 import (
    AddDocumentsToModelRequest,
    AddDocumentsToModelResponse,
    StartTrainTopicModelRequest,
    StartTrainTopicModelResponse,
    TrainTopicModelStatusRequest,
    TrainTopicModelStatusResponse, UpdateModelConfigurationRequest, UpdateModelConfigurationResponse,
)
from ap.topic_model.v1.TopicModelTrain_pb2_grpc import (
    TopicModelTrainServiceServicer,
    add_TopicModelTrainServiceServicer_to_server,
)
from ap.train.data_manager import ModelDataManager, NoTranslationException
from ap.train.metrics import send_metric, run_metrics_server, inc_metric

from ap.train.trainer import ModelTrainer
from ap.utils.bpe import load_bpe_models
from ap.utils.general import docs_from_pack, id_to_str
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class TopicModelTrainServiceImpl(TopicModelTrainServiceServicer):
    def __init__(
            self,
            train_conf: str,
            data_dir: str
    ):
        """
        Инициализирует сервер.

        Args:
            train_conf (typing.Dict[str, typing.Any]): словарь с конфигурацией обучения
            data_dir (str): путь к директории с данными
        """
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
        self._training_future = None

        with open(train_conf, "r") as file:
            self._config = yaml.safe_load(file)

        self._executor.submit(run_metrics_server, self._config)
        sleep(10)

        bpe_models = load_bpe_models(self._config["BPE_models"])
        self._vw = VowpalWabbitBPE(bpe_models)

        self._data_manager = ModelDataManager(data_dir, train_conf)
        self._trainer = ModelTrainer(self._data_manager)

    def AddDocumentsToModel(
            self, request: AddDocumentsToModelRequest, context
    ) -> AddDocumentsToModelResponse:
        """
        Добавляет документы в модель.

        Args:
            request (AddDocumentsToModelRequest): запрос с документами
            context: не используется

        Returns:
            (AddDocumentsToModelResponse): Ответ
        """
        try:
            logging.info("AddDocumentsToModel")

            docs = docs_from_pack(request.Collection)
            grouped_docs = {}
            for parallel_docs in request.ParallelDocuments:
                base_id = id_to_str(parallel_docs.Ids[0])
                grouped_docs[base_id] = docs[base_id]
                for i in range(1, len(parallel_docs.Ids)):
                    grouped_docs[base_id].update(docs[id_to_str(parallel_docs.Ids[i])])

            self._data_manager.write_new_docs(self._vw, grouped_docs)

            inc_metric('added_docs', len(grouped_docs))
            self._data_manager.update_ds_metrics()
        except NoTranslationException:
            return AddDocumentsToModelResponse(
                Status=AddDocumentsToModelResponse.AddDocumentsStatus.NO_TRANSLATION
            )
        except Exception as exception:
            logging.error(exception)
            return AddDocumentsToModelResponse(
                Status=AddDocumentsToModelResponse.AddDocumentsStatus.EXCEPTION
            )
        return AddDocumentsToModelResponse(
            Status=AddDocumentsToModelResponse.AddDocumentsStatus.OK
        )

    def StartTrainTopicModel(
            self, request: StartTrainTopicModelRequest, context
    ) -> StartTrainTopicModelResponse:
        """
        Запускает обучение.

        Args:
            request (StartTrainTopicModelRequest): запрос с типом обучения.
            context: не используется.

        Returns:
            (StartTrainTopicModelResponse): Статус запуска.
        """
        logging.info("StartTrainTopicModel")

        if self._training_future is not None and self._training_future.running():
            return StartTrainTopicModelResponse(
                Status=StartTrainTopicModelResponse.StartTrainTopicModelStatus.ALREADY_STARTED
            )

        self._training_future = self._executor.submit(
            self._trainer.train_model, request.Type
        )

        return StartTrainTopicModelResponse(
            Status=StartTrainTopicModelResponse.StartTrainTopicModelStatus.OK
        )

    def TrainTopicModelStatus(
            self, request: TrainTopicModelStatusRequest, context
    ) -> TrainTopicModelStatusResponse:
        """
        Возвращает статус текущей сессии обучения.

        Args:
            request (TrainTopicModelStatusRequest): пустой запрос
            context: контекст, не используется

        Returns
            (TrainTopicModelStatusResponse): Статус
        """
        if self._training_future is None or (
                self._training_future.done() and self._training_future.exception() is None
        ):
            return TrainTopicModelStatusResponse(
                Status=TrainTopicModelStatusResponse.TrainTopicModelStatus.COMPLETE
            )
        elif not self._training_future.done():
            return TrainTopicModelStatusResponse(
                Status=TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
            )
        elif (
                self._training_future.cancelled()
                or self._training_future.exception() is not None
        ):
            logging.error(str(self._training_future.exception()))
            return TrainTopicModelStatusResponse(
                Status=TrainTopicModelStatusResponse.TrainTopicModelStatus.ABORTED
            )

    def UpdateModelConfiguration(self, request: UpdateModelConfigurationRequest,
                                 context) -> UpdateModelConfigurationResponse:
        """обновление конфигурации обучения
        """
        self._data_manager.update_config(request.Configuration)
        return UpdateModelConfigurationResponse(
            Status=UpdateModelConfigurationResponse.UpdateModelConfigurationStatus.OK)


@click.command()
@click.option(
    "--config", help="A path to experiment yaml config",
)
@click.option(
    "--data", help="A path to data directories",
)
def serve(config, data):
    """
    Запускает сервер.

    Args:
        config (TODO): TODO
        data (TODO): TODO
    """
    from prometheus_client import start_http_server

    # TODO: дообучение:
    # если пути config["BPE_models"] нет - не надо загружать модели
    logging.basicConfig(level=logging.DEBUG)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TopicModelTrainServiceServicer_to_server(
        TopicModelTrainServiceImpl(config, data),
        server,
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
