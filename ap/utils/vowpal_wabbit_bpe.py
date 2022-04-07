import re
import typing

from collections import Counter
from string import punctuation

import youtokentome as yttm

from zhon import hanzi


class VowpalWabbitBPE:
    def __init__(self, bpe_models, use_counters=True):
        """
        Создает класс сохранения VW файлов с BPE преобразованием

        Parameters:
            bpe_models: TODO
            use_counters: признак использования каунтеров
        """
        self._bpe_models = bpe_models
        self._use_counters = use_counters
        self.punctuation = punctuation + hanzi.punctuation

    def save_docs(
        self, target_file: typing.Dict[str, str], doc: typing.Dict[str, typing.Dict[str, str]]
    ):
        """
        Конвертирует документы в BOW и сохраняет их.

        Parameters:
            target_file: путь к файлу
            doc: сырые документы
        """
        self.save_bow(target_file, self.convert_to_bow(doc))

    def save_bow(
        self,
        target_file: typing.Dict[str, str],
        sessions_bow_messages: typing.Dict[
            str, typing.Dict[str, typing.Union[str, typing.Counter]]
        ],
    ):
        """
        Сохраняет BOW представление документов.

        Parameters:
            target_file: словарь с vw
            sessions_bow_messages: документы в формате BOW
        """
        for key, modality_bows in sessions_bow_messages.items():
            new_message_str_format = str(key).replace(" ", "_")
            for modality, _ in modality_bows.items():
                if modality == "plain_text":
                    continue
                if self._use_counters:
                    modality_content = " ".join(
                        [
                            token + ":" + str(count)
                            for token, count in sessions_bow_messages[key][
                                modality
                            ].items()
                        ]
                    )
                else:
                    modality_content = sessions_bow_messages[key][modality]
                new_message_str_format += " |@{} {}".format(modality, modality_content)
                target_file[str(key).replace(" ", "_")] = new_message_str_format

    def convert_to_bow(
        self, data: typing.Dict[str, typing.Dict[str, str]]
    ) -> typing.Dict[str, typing.Dict[str, typing.Union[str, typing.Counter]]]:
        """
        Конвертирует набор документов в BOW представление (см. VowpalWabbit.convet_doct).

        Parameters:
            data: словарь айди документа->документ

        Returns:
            словарь айди документа->документ в виде BOW
        """
        sessions_bow_messages = dict()
        for elem_id, elem in data.items():
            sessions_bow_messages[elem_id] = self.convert_doc(elem)
        return sessions_bow_messages

    def convert_doc(
        self, doc: typing.Dict[str, str]
    ) -> typing.Dict[str, typing.Union[str, typing.Counter]]:
        """
        Конвертирует исходный документ в формат BOW.

        Parameters:
            doc словарь язык->текст документа

        Returns:
            словарь язык->BOW документа. Если use_counters==True, словарь в виде Counter
        """

        res = {}

        for modality, mod_elem in doc.items():
            print(modality)
            tokens = " ".join(self._token_filtration(mod_elem))
            if modality not in ['@UDK', '@GRNTI']:
                tokens = self._bpe_models[modality].encode(
                    tokens, output_type=yttm.OutputType.SUBWORD
                )
            res[modality] = Counter(tokens)

        return res

    def _token_filtration(self, text):
        tokens = []
        if isinstance(text, str):
            tokens = text.split()
            tokens = [
                re.sub(r"[\:\|\@]", "", token)
                for token in tokens
                if token not in self.punctuation
            ]
        elif isinstance(text, list):
            tokens = [
                re.sub(r"[\:\|\@ ]", "_", token.strip())
                for token in text
                if token not in self.punctuation
            ]
        tokens = [token for token in tokens if len(token) > 0]
        return tokens
