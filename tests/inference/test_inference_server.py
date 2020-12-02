import os
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from ap.topic_model.v1.TopicModelBase_pb2 import DocId, Document, DocumentPack
from ap.topic_model.v1.TopicModelInference_pb2 import GetDocumentsEmbeddingRequest


@pytest.fixture(scope="module")
def artm_model():
    mocked_model = MagicMock()
    mocked_model.transform = Mock(
        return_value=pd.DataFrame.from_dict({"0_0": [3, 2, 1, 0], "1_0": [3, 2, 1, 0]})
    )
    return mocked_model
    # return  artm.load_artm_model("../antiplagiat_models/PLSA_L14_V20000_TOP100_E1_G0_T10")


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from ap.topic_model.v1.TopicModelInference_pb2_grpc import (
        add_TopicModelInferenceServiceServicer_to_server,
    )

    return add_TopicModelInferenceServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(artm_model, bpe_models):
    from ap.inference.server import TopicModelInferenceServiceImpl

    return TopicModelInferenceServiceImpl(artm_model, bpe_models, os.getcwd())


@pytest.fixture(scope="module")
def grpc_stub_cls(grpc_channel):
    from ap.topic_model.v1.TopicModelInference_pb2_grpc import (
        TopicModelInferenceServiceStub,
    )

    return TopicModelInferenceServiceStub


def test_embeddings(artm_model, grpc_stub):
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
    resp = grpc_stub.GetDocumentsEmbedding(
        GetDocumentsEmbeddingRequest(Pack=DocumentPack(Documents=docs))
    )
    assert len(resp.Embeddings) == len(docs)
    artm_model.transform.assert_called_once()
