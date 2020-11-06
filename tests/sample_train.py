from time import sleep

import grpc

from ap.topic_model.v1.TopicModelBase_pb2 import (DocId, Document,
                                                  DocumentPack, ParallelDocIds)
from ap.topic_model.v1.TopicModelTrain_pb2 import (
    AddDocumentsToModelRequest, StartTrainTopicModelRequest,
    TrainTopicModelStatusRequest, TrainTopicModelStatusResponse)
from ap.topic_model.v1.TopicModelTrain_pb2_grpc import \
    TopicModelTrainServiceStub

if __name__ == "__main__":
    # ALL_LANGUAGES = ["ru", "en", 'cs', 'de', 'es', 'fr', 'it', 'ja',
    #                  'kk', 'ky', 'nl', 'pl', 'pt', 'tr', 'zh']
    # num_languages = len(ALL_LANGUAGES)
    # class_ids = {'@' + language: 1 for language in ALL_LANGUAGES[:num_languages]}
    # with open("./train_conf.yaml", "w") as f:
    #     yaml.dump({"num_epochs": 20, "tau": 0.25, "gamma": 0}, f, indent=3)

    channel = grpc.insecure_channel("localhost:50051")
    grpc_stub = TopicModelTrainServiceStub(channel)

    docs = [
        Document(
            Id=DocId(Lo=0, Hi=0),
            Tokens=[
                "introductorio",
                "proporciona",
                "rasfondo",
                "histórico",
                "sobr",
                "seguida",
            ],
            Language="es",
        ),
        Document(
            Id=DocId(Lo=0, Hi=1),
            Tokens=[
                "bevat",
                "meer",
                "dan",
                "500",
                "analoog",
                "gestructureerde",
                "coherente",
                "ook",
            ],
            Language="nl",
        ),
    ]
    parallel_docs = [
        ParallelDocIds(Ids=[DocId(Lo=0, Hi=1)]),
        ParallelDocIds(Ids=[DocId(Lo=0, Hi=0)]),
    ]
    resp = grpc_stub.AddDocumentsToModel(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=parallel_docs
        )
    )

    print(resp)

    resp = grpc_stub.StartTrainTopicModel(
        StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.FULL)
    )

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)

    print(resp)

    resp = grpc_stub.StartTrainTopicModel(
        StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.UPDATE)
    )

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)
