import concurrent
import logging
import typing

from concurrent import futures

import click
import grpc
import yaml

from ap.topic_model.v1.TopicModelTrain_pb2 import (
    AddDocumentsToModelRequest,
    AddDocumentsToModelResponse,
    StartTrainTopicModelRequest,
    StartTrainTopicModelResponse,
    TrainTopicModelStatusRequest,
    TrainTopicModelStatusResponse,
)
from ap.topic_model.v1.TopicModelTrain_pb2_grpc import (
    TopicModelTrainServiceServicer,
    add_TopicModelTrainServiceServicer_to_server,
)
from ap.train.data_manager import ModelDataManager, NoTranslationException
from ap.train.trainer import ModelTrainer
from ap.utils.bpe import load_bpe_models
from ap.utils.general import docs_from_pack, id_to_str
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class TopicModelTrainServiceImpl(TopicModelTrainServiceServicer):
    def __init__(
            self,
            bpe_models: typing.Dict[str, typing.Any],
            train_conf: typing.Dict[str, typing.Any],
            models_dir: str,
            data_dir: str
    ):
        """
        Инициализирует сервер.

        Parameters
        ----------
        train_conf - словарь с конфигурацией обучения
        models_dir - путь к директория сохранения файлов
        data_dir - путь к директории с данными
        """
        self._vw = VowpalWabbitBPE(bpe_models)
        self._data_manager = ModelDataManager(data_dir, train_conf)
        self._trainer = ModelTrainer(StartTrainTopicModelRequest.TrainType, self._data_manager, train_conf, models_dir)

        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)
        self._training_future = None

    def AddDocumentsToModel(
            self, request: AddDocumentsToModelRequest, context
    ) -> AddDocumentsToModelResponse:
        """
        Добавляет документы в модель.

        Parameters
        ----------
        request - запрос с документами
        context - не используется

        Returns
        -------
        Ответ
        """
        try:
            logging.info("AddDocumentsToModel")

            docs = docs_from_pack(request.Collection)

            for parallel_docs in request.ParallelDocuments:
                base_id = id_to_str(parallel_docs.Ids[0])
                for i in range(1, len(parallel_docs.Ids)):
                    docs[base_id].update(docs[id_to_str(parallel_docs.Ids[i])])

            self._data_manager.write_new_docs(self._vw, docs)
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

        Parameters
        ----------
        request - запрос с типом обучения.
        context - не используется.

        Returns
        -------
        Статус запуска.
        """
        logging.info("StartTrainTopicModel")

        if self._training_future is not None and self._training_future.running():
            return StartTrainTopicModelResponse(
                Status=StartTrainTopicModelResponse.StartTrainTopicModelStatus.ALREADY_STARTED
            )

        self._training_future = self._executor.submit(
            self._trainer.train_model, [request.Type]
        )
        return StartTrainTopicModelResponse(
            Status=StartTrainTopicModelResponse.StartTrainTopicModelStatus.OK
        )

    def TrainTopicModelStatus(
            self, request: TrainTopicModelStatusRequest, context
    ) -> TrainTopicModelStatusResponse:
        """
        Возвращает статус текущей сессии обучения.

        Parameters
        ----------
        request - пустой запрос
        context - контекст, не используется

        Returns
        -------
        Статус
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
                or logging.error(self._training_future.exception()) is not None
        ):
            logging.error(str(self._training_future.exception()))
            return TrainTopicModelStatusResponse(
                Status=TrainTopicModelStatusResponse.TrainTopicModelStatus.ABORTED
            )


@click.command()
@click.option(
    "--config", help="A path to experiment yaml config",
)
@click.option(
    "--models", help="A path to store trained bigARTM models",
)
@click.option(
    "--bpe", help="A path to a directory with BPE models",
)
@click.option(
    "--data", help="A path to data directories",
)
def serve(models, config, bpe, data, rubric):
    """
    Запускает сервер.

    Parameters
    ----------
    models - Путь к моделям
    data - Путь к данным
    """
    with open(config, "r") as file:
        train_conf = yaml.safe_load(file)

    logging.basicConfig(level=logging.DEBUG)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TopicModelTrainServiceServicer_to_server(
        TopicModelTrainServiceImpl(load_bpe_models(bpe), train_conf, models, data),
        server,
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
