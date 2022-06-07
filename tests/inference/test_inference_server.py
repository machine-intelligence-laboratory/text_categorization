import os

from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ap.topic_model.v1.TopicModelBase_pb2 import DocId, Document, DocumentPack, Modality
from ap.topic_model.v1.TopicModelInference_pb2 import GetDocumentsEmbeddingRequest, GetTopicExplanationRequest


@pytest.fixture(scope="module")
def artm_model():
    num_topic = 4
    topics = [f'topic_{i}' for i in range(num_topic)]
    tokens = {
            "минимальный": np.array([0.8, 0.2, 0, 0]),
            "остаточный": np.array([0.2, 0.8, 0, 0]),
            "заболевание": np.array([0.0, 0.2, 0.8, 0]),
            "в": np.array([0.25, 0.25, 0.25, 0.25]),
            "острый": np.array([0.0, 0.0, 0.2, 0.8]),
            "миелоидный": np.array([0.25, 0.25, 0.25, 0.25]),
            "раз": np.array([0.25, 0.25, 0.25, 0.25]),
            "два": np.array([0.25, 0.25, 0.25, 0.25]),
            "три": np.array([0.25, 0.25, 0.25, 0.25]),
            "выходи": np.array([0.25, 0.25, 0.25, 0.25])
        }
    phi = np.stack([t for t in tokens.values()])
    doc = ["минимальный",
            "в",
            "два",
            "три",
            "выходи"]
    doc = np.linalg.norm(np.sum([tokens[x] for x in doc]))
    thetas = [[5.72787840e-02, 5.13726954e-02, 8.18006932e-04, 8.90530514e-01],
                [4.87438921e-04, 6.83337098e-01, 2.91425844e-01, 2.47496193e-02]]
    mocked_model = MagicMock()
    mocked_model.transform = Mock(
        side_effect=[pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                  columns=[
                                      "0_0",
                                  ],
                                  data=thetas[0]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1]),
                      pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                   columns=[
                                       "0_0",
                                   ],
                                   data=thetas[1])]
    )
    mocked_model.get_phi = Mock(
        return_value=pd.DataFrame(index=list(tokens.keys()),
            columns=[f'topic_{i}' for i in range(num_topic)],
            data=np.random.rand(10, num_topic))
    )
    mocked_model.class_ids = ['@ru']
    return mocked_model


@pytest.fixture(scope="module")
def grpc_add_to_server():
    from ap.topic_model.v1.TopicModelInference_pb2_grpc import (
        add_TopicModelInferenceServiceServicer_to_server,
    )

    return add_TopicModelInferenceServiceServicer_to_server


@pytest.fixture(scope="module")
def grpc_servicer(artm_model, bpe_models):
    from ap.inference.server import TopicModelInferenceServiceImpl

    return TopicModelInferenceServiceImpl(artm_model, bpe_models, os.getcwd(), 'tests/data')


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
            Modalities=[Modality(Key="lang", Value='es'), Modality(Key="UDK", Value='6'),
                        Modality(Key="GRNTI", Value='11806946'), ],
        ),
    ]
    resp = grpc_stub.GetDocumentsEmbedding(
        GetDocumentsEmbeddingRequest(Pack=DocumentPack(Documents=docs))
    )
    assert len(resp.Embeddings) == len(docs)
    artm_model.transform.assert_called_once()
