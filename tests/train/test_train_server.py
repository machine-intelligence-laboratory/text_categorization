import os
import shutil
from time import sleep

import pytest
import yaml

from ap.topic_model.v1.TopicModelBase_pb2 import (
    DocId,
    Document,
    DocumentPack,
    ParallelDocIds,
)
from ap.topic_model.v1.TopicModelTrain_pb2 import (
    AddDocumentsToModelRequest,
    AddDocumentsToModelResponse,
    StartTrainTopicModelRequest,
    StartTrainTopicModelResponse,
    TrainTopicModelStatusRequest,
    TrainTopicModelStatusResponse,
)


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from ap.topic_model.v1.TopicModelTrain_pb2_grpc import (
        add_TopicModelTrainServiceServicer_to_server,
    )

    return add_TopicModelTrainServiceServicer_to_server


@pytest.fixture(scope="module")
def models_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("models")


@pytest.fixture(scope="module")
def data_dir(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")
    langs = [
        "ru",
        "en",
        "cs",
        "de",
        "es",
        "fr",
        "it",
        "ja",
        "kk",
        "ky",
        "nl",
        "pl",
        "pt",
        "tr",
        "zh",
    ]
    class_ids = {"@" + language: 1 for language in langs}
    with open(os.path.join(data_dir, "classes.yaml"), "w") as file:
        yaml.safe_dump(class_ids, file)

    shutil.copy("./tests/data/dictionary.txt", os.path.join(data_dir, "dictionary.txt"))

    return data_dir


@pytest.fixture(scope="module")
def test_conf():
    return {
        "num_epochs_full": 3,
        "num_epochs_update": 2,
        "num_topics": 100,
        "num_bg_topics": 100,
        "tau": 0.2,
        "gamma": 0,
        "max_dictionary_size": 10,
    }


@pytest.fixture(scope="module")
def grpc_servicer(test_conf, models_dir, data_dir, bpe_models):
    from ap.train.server import TopicModelTrainServiceImpl

    return TopicModelTrainServiceImpl(bpe_models, test_conf, models_dir, data_dir)


@pytest.fixture(scope="module")
def grpc_stub_cls():
    from ap.topic_model.v1.TopicModelTrain_pb2_grpc import TopicModelTrainServiceStub

    return TopicModelTrainServiceStub


@pytest.fixture(scope="function")
def clean_data(data_dir):
    vw_new = os.path.join(data_dir, "vw_new")
    for file in os.listdir(vw_new):
        os.remove(os.path.join(vw_new, file))


@pytest.mark.usefixtures("clean_data")
def test_add_documents(models_dir, data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"], Language="en"),
        Document(Id=DocId(Lo=0, Hi=1), Tokens=["c", "D"], Language="en"),
    ]
    parallel_docs = ParallelDocIds(Ids=[DocId(Lo=0, Hi=0)])
    resp = grpc_stub.add_documents_to_model(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=[parallel_docs]
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.OK

    with open(os.path.join(data_dir, "vw_new", "actual.txt"), "r") as file:
        res = file.readlines()
        assert len(res) == 2


@pytest.mark.usefixtures("clean_data")
def test_add_documents_new_lang(models_dir, data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"], Language="gf"),
        Document(Id=DocId(Lo=0, Hi=1), Tokens=["c", "D"], Language="en"),
    ]
    parallel_docs = ParallelDocIds(Ids=[DocId(Lo=0, Hi=0), DocId(Lo=0, Hi=1)])
    resp = grpc_stub.add_documents_to_model(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=[parallel_docs]
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.OK

    with open(os.path.join(data_dir, "vw_new", "actual.txt"), "r") as file:
        res = file.readlines()
        assert len(res) == 2


@pytest.mark.usefixtures("clean_data")
def test_add_documents_new_lang_no_translation(models_dir, data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"], Language="rq"),
        Document(Id=DocId(Lo=0, Hi=1), Tokens=["c", "D"], Language="rq"),
        Document(Id=DocId(Lo=0, Hi=2), Tokens=["c", "D"], Language="fr"),
    ]
    parallel_docs = ParallelDocIds(Ids=[DocId(Lo=0, Hi=0)])
    resp = grpc_stub.add_documents_to_model(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=[parallel_docs]
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.NO_TRANSLATION
    assert not os.path.exists(os.path.join(data_dir, "vw_new", "actual.txt"))


def test_start_train(data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"], Language="gf"),
        Document(Id=DocId(Lo=0, Hi=1), Tokens=["c", "D"], Language="en"),
        Document(Id=DocId(Lo=1, Hi=0), Tokens=["e", "f"], Language="gf"),
        Document(Id=DocId(Lo=1, Hi=1), Tokens=["c", "b"], Language="en"),
        Document(Id=DocId(Lo=2, Hi=0), Tokens=["a", "f"], Language="gf"),
        Document(Id=DocId(Lo=2, Hi=1), Tokens=["a", "b"], Language="en"),
    ]
    parallel_docs = [
        ParallelDocIds(Ids=[DocId(Lo=0, Hi=0), DocId(Lo=0, Hi=1)]),
        ParallelDocIds(Ids=[DocId(Lo=1, Hi=0), DocId(Lo=1, Hi=1)]),
        ParallelDocIds(Ids=[DocId(Lo=2, Hi=0), DocId(Lo=2, Hi=1)]),
    ]

    resp = grpc_stub.add_documents_to_model(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=parallel_docs
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.OK

    resp = grpc_stub.start_train_topic_model(
        StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.FULL)
    )
    assert resp.Status == StartTrainTopicModelResponse.StartTrainTopicModelStatus.OK

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)

    assert (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.COMPLETE
    )
    assert len(os.listdir(os.path.join(data_dir, "vw_new"))) == 0
    assert len(os.listdir(os.path.join(data_dir, "batches_new"))) == 0
    assert len(os.listdir(os.path.join(data_dir, "vw"))) > 0
    assert len(os.listdir(os.path.join(data_dir, "batches"))) > 0

    with open(os.path.join(data_dir, "dictionary.txt")) as file:
        assert len(file.readlines()) == 10

    resp = grpc_stub.start_train_topic_model(
        StartTrainTopicModelRequest(Type=StartTrainTopicModelRequest.TrainType.UPDATE)
    )
    assert resp.Status == StartTrainTopicModelResponse.StartTrainTopicModelStatus.OK

    while (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.RUNNING
    ):
        sleep(1)

    assert (
        grpc_stub.TrainTopicModelStatus(TrainTopicModelStatusRequest()).Status
        == TrainTopicModelStatusResponse.TrainTopicModelStatus.COMPLETE
    )
    assert len(os.listdir(os.path.join(data_dir, "vw_new"))) == 0
    assert len(os.listdir(os.path.join(data_dir, "batches_new"))) == 0
    assert len(os.listdir(os.path.join(data_dir, "vw"))) > 0
    assert len(os.listdir(os.path.join(data_dir, "batches"))) > 0
