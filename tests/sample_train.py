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
            Tokens=["документ", "слово", "научный", "ответ", "еще", "что-то"],
            Modalities=[Modality(Key="lang", Value='ru'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64')],
        ),
        Document(
            Id=DocId(Lo=0, Hi=1),
            Tokens=["document", "word", "scientific", "answer", "more", "something"],
            Modalities=[Modality(Key="lang", Value='en'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64')],
        ),
        Document(
            Id=DocId(Lo=0, Hi=2),
            Tokens=["documento", "parola", "scientifica", "Rispondere", "Di più", "qualche cosa"],
            Modalities=[Modality(Key="lang", Value='it'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64')],
        ),

        Document(
            Id=DocId(Lo=1, Hi=0),
            Tokens=["другой", "документ", "русский", "500", "язык", "наука", "ок"],
            Modalities=[Modality(Key="lang", Value='ru'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64'), ],
        ),
        Document(
            Id=DocId(Lo=1, Hi=1),
            Tokens=["another", "document", "Russian", "500", "language", "science", "ok"],
            Modalities=[Modality(Key="lang", Value='en'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64'), ],
        ),
        Document(
            Id=DocId(Lo=1, Hi=2),
            Tokens=["un altro", "documento", "russo", "500", "lingua", "scienza", "ok"],
            Modalities=[Modality(Key="lang", Value='it'),
                        Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='64'), ],
        ),

        Document(
            Id=DocId(Lo=2, Hi=0),
            Tokens=["сегодня", "хорошая", "погода"],
            Modalities=[Modality(Key="lang", Value='ru'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='64')],
        ),
        Document(
            Id=DocId(Lo=2, Hi=1),
            Tokens=["today", "good", "weather"],
            Modalities=[Modality(Key="lang", Value='en'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='64')],
        ),
        Document(
            Id=DocId(Lo=2, Hi=2),
            Tokens=["oggi", "buono", "tempo"],
            Modalities=[Modality(Key="lang", Value='it'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='64')],
        ),

        Document(
            Id=DocId(Lo=3, Hi=0),
            Tokens=["раз", "два", "три"],
            Modalities=[Modality(Key="lang", Value='ru'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='61')],
        ),
        Document(
            Id=DocId(Lo=3, Hi=1),
            Tokens=["one", "two", "three"],
            Modalities=[Modality(Key="lang", Value='en'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='61')],
        ),

        Document(
            Id=DocId(Lo=4, Hi=0),
            Tokens=["кружка", "ложка", "миска", "нож"],
            Modalities=[Modality(Key="lang", Value='ru'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='61')],
        ),
        Document(
            Id=DocId(Lo=4, Hi=1),
            Tokens=["mug", "spoon", "bowl", "knife"],
            Modalities=[Modality(Key="lang", Value='en'),
                        Modality(Key="UDK", Value='6'), Modality(Key="GRNTI", Value='61')],
        ),
    ]
    parallel_docs = [
        ParallelDocIds(Ids=[DocId(Lo=0, Hi=0), DocId(Lo=0, Hi=1), DocId(Lo=0, Hi=2)]),
        ParallelDocIds(Ids=[DocId(Lo=1, Hi=0), DocId(Lo=1, Hi=1), DocId(Lo=1, Hi=2)]),
        ParallelDocIds(Ids=[DocId(Lo=2, Hi=0), DocId(Lo=2, Hi=1), DocId(Lo=2, Hi=2)]),
        ParallelDocIds(Ids=[DocId(Lo=3, Hi=0), DocId(Lo=3, Hi=1)]),
        ParallelDocIds(Ids=[DocId(Lo=4, Hi=0), DocId(Lo=4, Hi=1)]),
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
