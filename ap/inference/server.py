import datetime
import logging
import os
import tempfile
import uuid
import json

from concurrent import futures

import artm
import click
import grpc
import pandas as pd

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
    def __init__(self, artm_model, bpe_models, work_dir, rubric_dir):
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
        self._rubric_dir = rubric_dir

    # TODO: может стоит испортировать? дублируется с ap/train/data_manager.py
    def get_rubric_of_train_docs(self):
        """
        Get dict where keys - document ids, value - number of GRNTI rubric of document.

        Do not contains rubric 'нет'.

        Returns
        -------
        train_grnti: dict
            dict where keys - document ids, value - numer of GRNTI rubric of document.
        """
        with open(os.path.join(self._rubric_dir, 'grnti_codes.json')) as file:
            articles_grnti_with_no = json.load(file)
        with open(os.path.join(self._rubric_dir, "elib_train_grnti_codes.json")) as file:
            elib_grnti_to_fix_with_no = json.load(file)
        with open(os.path.join(self._rubric_dir, "grnti_to_number.json")) as file:
            grnti_to_number = json.load(file)

        articles_grnti = {doc_id: rubric
                          for doc_id, rubric in articles_grnti_with_no.items()
                          if rubric != 'нет'}

        elib_grnti = {doc_id[:-len('.txt')]: rubric
                      for doc_id, rubric in elib_grnti_to_fix_with_no.items()
                      if rubric != 'нет'}

        train_grnti = dict()
        for doc_id in articles_grnti:
            rubric = str(grnti_to_number[articles_grnti[doc_id]])
            train_grnti[doc_id] = rubric
        for doc_id in elib_grnti:
            rubric = str(grnti_to_number[elib_grnti[doc_id]])
            train_grnti[doc_id] = rubric
        return train_grnti

    def _create_batches(self, dock_pack, batches_dir):
        with open(os.path.join(self._rubric_dir, 'udk_codes.json'), "r") as file:
            udk_codes = json.loads(file.read())

        grnti_codes = self.get_rubric_of_train_docs()

        documents = []
        vocab = set()

        for doc in dock_pack.Documents:
            modality = ["@" + doc.Language]
            doc_id = id_to_str(doc.Id)

            doc_vw_dict = {doc.Language: " ".join(doc.Tokens)}
            if doc_id in udk_codes:
                modality += ["@UDK"]
                doc_vw_dict.update({"@UDK": udk_codes[doc_id]})
            if doc_id in grnti_codes:
                modality += ["@GRNTI"]
                if grnti_codes[doc_id] != "нет":
                    doc_vw_dict.update({"@GRNTI": grnti_codes[doc_id]})

            vw_doc = self._vw.convert_doc(doc_vw_dict)

            for modl in modality:
                key = modl if modl in ["@UDK", "@GRNTI"] else modl[1:]
                documents.append((id_to_str(doc.Id), key, vw_doc[key]))
                vocab.update(((key, token) for token in vw_doc[key].keys()))

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

                for key, value in local_dict.items():
                    item.token_id.append(dictionary[(modality, key)])
                    item.token_weight.append(value)
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

        emb_doc = list()
        for doc in request.Pack.Documents:
            emb = embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"]
            if isinstance(emb, pd.DataFrame):
                emb_doc += [Embedding(Vector=embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"].iloc[:, 0].tolist())]
            else:
                emb_doc += [Embedding(Vector=embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"].tolist())]

        return GetDocumentsEmbeddingResponse(
            Embeddings=emb_doc
        )


@click.command()
@click.option(
    "--model", help="A path to a bigARTM model",
)
@click.option(
    "--bpe", help="A path to a directory with BPE models",
)
@click.option(
    "--rubric", help="A path to a directory with Rubric jsons",
)
def serve(model, bpe, rubric):
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
            artm.load_artm_model(model), load_bpe_models(bpe), os.getcwd(), rubric
        ),
        server,
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logging.info("Server started")
    server.wait_for_termination()


if __name__ == "__main__":
    # TODO: сюда надо передавать model, bpe, rubric
    serve()
