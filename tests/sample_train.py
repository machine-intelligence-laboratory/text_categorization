from time import sleep

import grpc

from ap.topic_model.v1.TopicModelBase_pb2 import (
    DocId,
    Document,
    DocumentPack,
    ParallelDocIds,
)
from ap.topic_model.v1.TopicModelTrain_pb2 import (
    AddDocumentsToModelRequest,
    StartTrainTopicModelRequest,
    TrainTopicModelStatusRequest,
    TrainTopicModelStatusResponse,
)
from ap.topic_model.v1.TopicModelTrain_pb2_grpc import TopicModelTrainServiceStub

if __name__ == "__main__":

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
    # resp = grpc_stub.AddDocumentsToModel(
    #     AddDocumentsToModelRequest(
    #         Collection=DocumentPack(Documents=docs), ParallelDocuments=parallel_docs
    #     )
    # )

    # print(resp)

    print('before')
    resp = grpc_stub.StartTrainTopicModel(
        StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.FULL)
    )
    print('after')

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)

    print(resp)

    # resp = grpc_stub.StartTrainTopicModel(
    #     StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.UPDATE)
    # )

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)
