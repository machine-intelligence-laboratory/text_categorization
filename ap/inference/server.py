import datetime
import logging
import os
import tempfile
import uuid
from concurrent import futures

import artm
import click
import grpc

from ap.topic_model.v1.TopicModelBase_pb2 import Embedding
from ap.topic_model.v1.TopicModelInference_pb2 import (
    GetDocumentsEmbeddingRequest,
    GetDocumentsEmbeddingResponse,
)
from ap.topic_model.v1.TopicModelInference_pb2_grpc import (
    TopicModelInferenceServiceServicer,
    add_TopicModelInferenceServiceServicer_to_server,
)
from ap.utils.bpe import load_bpe_models
from ap.utils.general import id_to_str
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class TopicModelInferenceServiceImpl(TopicModelInferenceServiceServicer):
    def __init__(self, artm_model, bpe_models, work_dir):
        """
        Создает инференс сервер.

        Parameters
        ----------
        artm_model - BigARTM модель
        work_dir - рабочая директория для сохранения временных файлов
        """
        self._artm_model = artm_model
        self._vw = VowpalWabbitBPE(bpe_models)
        self._work_dir = work_dir

    def _create_batches(self, dock_pack, batches_dir):
        documents = []
        vocab = set()

        for doc in dock_pack.Documents:
            modality = "@" + doc.Language
            vw_doc = self._vw.convert_doc({modality: " ".join(doc.Tokens)})
            documents.append((id_to_str(doc.Id), modality, vw_doc[modality]))
            vocab.update(((modality, token) for token in vw_doc[modality].keys()))

        batch = artm.messages.Batch()
        batch.id = str(uuid.uuid4())
        dictionary = {}
        use_bag_of_words = True

        for i, (modality, token) in enumerate(vocab):
            batch.token.append(token)
            batch.class_id.append(modality)
            dictionary[(modality, token)] = i

        for idx, modality, doc in documents:
            item = batch.item.add()
            item.title = idx

            if use_bag_of_words:
                local_dict = {}
                for token in doc:
                    if token not in local_dict:
                        local_dict[token] = 0
                    local_dict[token] += 1

                for k, v in local_dict.items():
                    item.token_id.append(dictionary[(modality, k)])
                    item.token_weight.append(v)
            else:
                for token in doc:
                    item.token_id.append(dictionary[(modality, token)])
                    item.token_weight.append(1.0)

        with open(os.path.join(batches_dir, "aaaaaa.batch"), "wb") as fout:
            fout.write(batch.SerializeToString())

    def _transform(self, docs):
        with tempfile.TemporaryDirectory(dir=self._work_dir) as temp_dir:
            batches_dir = os.path.join(temp_dir, "batches")
            os.makedirs(batches_dir)
            self._create_batches(docs, batches_dir)
            batch_vectorizer = artm.BatchVectorizer(
                data_format="batches", data_path=batches_dir,
            )

            return self._artm_model.transform(batch_vectorizer)

    def GetDocumentsEmbedding(self, request: GetDocumentsEmbeddingRequest, context):
        """
        Возвращает ембеддинги документов.

        Parameters
        ----------
        request - реквест
        context - контекст, не используется

        Returns
        -------
        Ответ
        """
        logging.info(
            "Got request to calculate embeddings for %d documents",
            len(request.Pack.Documents),
        )
        processing_start = datetime.datetime.now()
        embeddings = self._transform(request.Pack)  # docs_from_pack(request.Pack))

        logging.info(
            "Embeddings calculated, spent %.2f seconds",
            (datetime.datetime.now() - processing_start).total_seconds(),
        )
        return GetDocumentsEmbeddingResponse(
            Embeddings=[
                Embedding(Vector=embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"].tolist())
                for doc in request.Pack.Documents
            ]
        )


@click.command()
@click.option(
    "--model", help="A path to a bigARTM model",
)
@click.option(
    "--bpe", help="A path to a directory with BPE models",
)
def serve(model, bpe):
    """
    Запуск инференс сервера.

    Parameters
    ----------
    model - путь к модели
    """
    logging.basicConfig(level=logging.DEBUG)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_TopicModelInferenceServiceServicer_to_server(
        TopicModelInferenceServiceImpl(
            artm.load_artm_model(model), load_bpe_models(bpe), os.getcwd()
        ),
        server,
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
