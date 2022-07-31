import datetime
import logging
import os
import tempfile
import json

from concurrent import futures
from collections import Counter

import artm
import click
import grpc
import pandas as pd

from ap.topic_model.v1.TopicModelBase_pb2 import Embedding, TopicExplanation
from ap.topic_model.v1.TopicModelInference_pb2 import (
    GetDocumentsEmbeddingRequest,
    GetDocumentsEmbeddingResponse, GetTopicExplanationRequest, GetTopicExplanationResponse,
)
from ap.topic_model.v1.TopicModelInference_pb2_grpc import (
    TopicModelInferenceServiceServicer,
    add_TopicModelInferenceServiceServicer_to_server,
)
from ap.utils.bpe import load_bpe_models
from ap.utils.general import id_to_str, get_modalities
from ap.utils.prediction_visualization import augment_text
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


class TopicModelInferenceServiceImpl(TopicModelInferenceServiceServicer):
    def __init__(self, artm_model, bpe_models, work_dir, rubric_dir):
        """
        Создает инференс сервер.

        Args:
            artm_model (artm.ARTM): тематическая модель
            bpe_models: загруженные BPE модели
            work_dir: рабочая директория для сохранения временных файлов
            rubric_dir: директория, где хранятся json-файлы с рубриками
        """
        # self._artm_model = artm_model
        self._artm_model =  artm_model

        self._vw = VowpalWabbitBPE(bpe_models)
        self._work_dir = work_dir
        self._rubric_dir = rubric_dir

    def _get_lang(self, doc):
        for modality in doc.Modalities:
            if modality.Key == 'lang':
                return modality.Value

        raise Exception("No language")

    def _transform(self, docs):
        with open(os.path.join(self._rubric_dir, 'udk_codes.json'), "r") as file:
            udk_codes = json.loads(file.read())

        with open(os.path.join(self._rubric_dir, 'rubrics_train_grnti.json'), "r") as file:
            grnti_codes = json.load(file)
        to_write = ''
        for doc in docs.Documents:
            lang = self._get_lang(doc)
            modality = ["@" + lang]
            doc_id = id_to_str(doc.Id)

            doc_vw_dict = {lang: " ".join(doc.Tokens)}
            if doc_id in udk_codes:
                modality += ["@UDK"]
                doc_vw_dict.update({"UDK": udk_codes[doc_id]})
            if doc_id in grnti_codes:
                modality += ["@GRNTI"]
                if grnti_codes[doc_id] != "нет":
                    doc_vw_dict.update({"GRNTI": grnti_codes[doc_id]})

            vw_doc = self._vw.convert_doc(doc_vw_dict)
            to_write += f'{doc_id} '
            to_write += ' '.join([f'|@{lang} ' + ' '.join([':'.join([str(token), str(counter)])
                                for token, counter in token_with_counter.items()])
                                for lang, token_with_counter in vw_doc.items()]) + '\n'
        with open('./inference_data.txt', 'w') as file:
            file.write(to_write)
        batch_vectorizer = artm.BatchVectorizer(data_path='./inference_data.txt',
                                                data_format='vowpal_wabbit',
                                                target_folder='tmp_batches')
        return self._artm_model.transform(batch_vectorizer)

    def GetDocumentsEmbedding(self, request: GetDocumentsEmbeddingRequest, context):
        """
        Возвращает ембеддинги документов.

        Args:
            request (GetDocumentsEmbeddingRequest): реквест
            context: контекст, не используется

        Returns:
            (GetDocumentsEmbeddingResponse): Ответ
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

        emb_doc = []
        for doc in request.Pack.Documents:
            emb = embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"]
            if isinstance(emb, pd.DataFrame):
                emb_doc += [Embedding(Vector=embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"].iloc[:, 0].tolist())]
            else:
                emb_doc += [Embedding(Vector=embeddings[f"{doc.Id.Hi}_{doc.Id.Lo}"].tolist())]

        return GetDocumentsEmbeddingResponse(
            Embeddings=emb_doc
        )

    def GetTopicExplanation(self, request: GetTopicExplanationRequest, context) -> GetTopicExplanationResponse:
        """
        Объяснение тематической модели
        Args:
            request: grpc запрос, содержащий документ
            context: не используется

        Returns:
            Объяснение тематической модели
        """
        with tempfile.TemporaryDirectory(dir=self._work_dir) as temp_dir:
            doc_vw = {id_to_str(request.Doc.Id): get_modalities(request.Doc)}
            vw_file = os.path.join(temp_dir, 'vw.txt')
            print('doc_vw', doc_vw)
            with open(vw_file, 'w') as file:
                to_write = []
                for doc_id, mod_dict in doc_vw.items():
                    line = f'{doc_id}'
                    for mod, content in mod_dict.items():
                        line += f' |@{mod} ' + \
                                ' '.join([f'{token}:{count}' for token, count in Counter(content.split()).items()])
                    line += '\n'
                    to_write.append(line)
                file.writelines(to_write)
            interpretation = augment_text(self._artm_model, vw_file, os.path.join(temp_dir, 'target'))
            return GetTopicExplanationResponse(Explanation=TopicExplanation(
                                               Topic=interpretation[id_to_str(request.Doc.Id)]['topic_from'],
                                               NewTopic=interpretation[id_to_str(request.Doc.Id)]['topic_to'],
                                               RemovedTokens=interpretation[id_to_str(request.Doc.Id)]['Removed'],
                                               AddedTokens=interpretation[id_to_str(request.Doc.Id)]['Added']))


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

    Args:
        model (str): путь к модели
        bpe (str): путь к обученным BPE моделям
        rubric (str): путь к директории с json-файлами рубрик
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
    serve()
