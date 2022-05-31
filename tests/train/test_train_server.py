import os
import shutil
from time import sleep

import pytest
import yaml

from ap.topic_model.v1.TopicModelBase_pb2 import (
    DocId,
    Document,
    DocumentPack,
    ParallelDocIds, Modality,
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
    shutil.copy("tests/data/train.txt", os.path.join(data_dir, "train.txt"))
    shutil.copy("tests/data/test_config.yml", os.path.join(data_dir, "test_config.yml"))
    with open(os.path.join(data_dir, "test_config.yml")) as f:
        c = yaml.safe_load(f)

    c['train_vw_path'] = os.path.join(data_dir, "train.txt")
    c['path_experiment'] = os.path.join(data_dir, "best_model")
    c['path_wiki_train_batches'] = os.path.join(data_dir, "batches_train")

    with open(os.path.join(data_dir, "test_config.yml"), 'w') as f:
        yaml.safe_dump(c, f)

    return data_dir


@pytest.fixture(scope="module")
def test_conf(data_dir):
    return os.path.join(data_dir, "test_config.yml")


@pytest.fixture(scope="module")
def grpc_servicer(test_conf, data_dir):
    from ap.train.server import TopicModelTrainServiceImpl

    return TopicModelTrainServiceImpl(test_conf, data_dir)


@pytest.fixture(scope="module")
def grpc_stub_cls():
    from ap.topic_model.v1.TopicModelTrain_pb2_grpc import TopicModelTrainServiceStub

    return TopicModelTrainServiceStub


@pytest.fixture(scope="function")
def clean_data(data_dir):
    shutil.copy("tests/data/train.txt", os.path.join(data_dir, "train.txt"))
    shutil.copy("tests/data/test_config.yml", os.path.join(data_dir, "test_config.yml"))
    with open(os.path.join(data_dir, "test_config.yml")) as f:
        c = yaml.safe_load(f)

    c['train_vw_path'] = os.path.join(data_dir, "train.txt")
    c['path_experiment'] = os.path.join(data_dir, "best_model")
    c['path_wiki_train_batches'] = os.path.join(data_dir, "batches_train")

    with open(os.path.join(data_dir, "test_config.yml"), 'w') as f:
        yaml.safe_dump(c, f)


@pytest.mark.usefixtures("clean_data")
def test_add_documents(models_dir, data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"],
                 Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value='6'),
                             Modality(Key="GRNTI", Value='1'), ]),
        Document(Id=DocId(Lo=0, Hi=1), Tokens=["c", "D"],
                 Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value='6'),
                             Modality(Key="GRNTI", Value='1'), ]),
    ]
    parallel_docs = ParallelDocIds(Ids=[DocId(Lo=0, Hi=0)])
    resp = grpc_stub.AddDocumentsToModel(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=[parallel_docs]
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.OK

    with open(os.path.join(data_dir, "train.txt"), "r") as file:
        res = file.readlines()
        assert len(res) == 42


@pytest.mark.usefixtures("clean_data")
def test_add_documents_new_lang_no_bpe(models_dir, data_dir, grpc_stub):
    docs = [
        Document(Id=DocId(Lo=0, Hi=0), Tokens=["a", "b"],
                 Modalities=[Modality(Key="lang", Value='de'), Modality(Key="UDK", Value='6'),
                             Modality(Key="GRNTI", Value='1'), ]),
    ]
    parallel_docs = ParallelDocIds(Ids=[DocId(Lo=0, Hi=0)])
    resp = grpc_stub.AddDocumentsToModel(
        AddDocumentsToModelRequest(
            Collection=DocumentPack(Documents=docs), ParallelDocuments=[parallel_docs]
        )
    )

    assert resp.Status == AddDocumentsToModelResponse.AddDocumentsStatus.EXCEPTION
    with open(os.path.join(data_dir, "train.txt"), "r") as file:
        res = file.readlines()
        assert len(res) == 40


def test_start_train(data_dir, grpc_stub):
    resp = grpc_stub.StartTrainTopicModel(
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

    resp = grpc_stub.StartTrainTopicModel(
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