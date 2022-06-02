import os

from unittest.mock import MagicMock, Mock

import artm
import numpy as np
import pandas as pd
import pytest

from ap.topic_model.v1.TopicModelBase_pb2 import DocId, Document, DocumentPack, Modality
from ap.topic_model.v1.TopicModelInference_pb2 import GetDocumentsEmbeddingRequest, GetTopicExplanationRequest


@pytest.fixture(scope="module")
def artm_model():
    num_topic = 4
    mocked_model = MagicMock()
    mocked_model.transform = Mock(
        return_value=pd.DataFrame(index=[f'topic_{i}' for i in range(num_topic)],
                                  columns=[
                                      "0_0",
                                  ],
                                  data=np.random.rand(num_topic, 1))
    )
    mocked_model.get_phi = Mock(
        return_value=pd.DataFrame(index=[
            "минимальный",
            "остаточный",
            "заболевание",
            "в",
            "острый",
            "миелоидный",
            "раз",
            "два",
            "три",
            "выходи"
        ],
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


# def test_explain(artm_model, grpc_stub):
    # doc = Document(
    #     Id=DocId(Lo=0, Hi=0),
    #     Tokens=[
    #         "минимальный",
    #         "требование",
    #         "разработка",
    #         "такой",
    #         "остаточный",
    #         "заболевание",
    #         "в",
    #         "острый",
    #         "миелоидный",
    #         "являться",
    #         "дать"
    #     ],
    #     Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value='6'),
    #                 Modality(Key="GRNTI", Value='11806946'), ],
    # )
def test_explain(artm_model, grpc_stub):
    with open('tests/data/test_ru.txt') as file:
        data = file.read()
    tokens_with_counter = data.split('|@')[1].split()[1:]
    tokens = []
    for token_with_counter in tokens_with_counter:
        token, counter = token_with_counter.split(':')
        tokens.extend([token] * int(counter))
    udk = data.split('|@')[2].split()[1].split(':')[0]
    grnti = data.split('|@')[3].split()[1].split(':')[0]
    doc = Document(
        Id=DocId(Lo=0, Hi=0),
        Tokens=tokens,
        Modalities=[Modality(Key="lang", Value='ru'), Modality(Key="UDK", Value=udk),
                    Modality(Key="GRNTI", Value=grnti), ],
    )

    resp = grpc_stub.GetTopicExplanation(GetTopicExplanationRequest(Doc=doc))
    print(resp.Explanation.Topic)
    print(resp.Explanation.NewTopic)
    print(resp.Explanation.RemovedTokens)
    print(resp.Explanation.AddedTokens)
    #artm_model.transform.assert_called_once()
