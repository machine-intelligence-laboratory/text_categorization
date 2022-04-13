from time import sleep

import grpc

from ap.topic_model.v1.TopicModelBase_pb2 import (
    DocId,
    Document,
    DocumentPack,
    ParallelDocIds, Modality,
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
                "документ",
                "слово",
                "научный",
                "ответ",
                "еще",
                "что-то",
            ],
            Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='11806946'),],
        ),
        Document(
            Id=DocId(Lo=0, Hi=1),
            Tokens=[
                "другой",
                "документ",
                "русский",
                "500",
                "язык",
                "наука",
                "ок"
            ],
            Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='11806946'),],
        ),
    ]
    parallel_docs = [
        ParallelDocIds(Ids=[DocId(Lo=0, Hi=1), DocId(Lo=0, Hi=0)])
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

    # resp = grpc_stub.StartTrainTopicModel(
    #     StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.UPDATE)
    # )

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)
