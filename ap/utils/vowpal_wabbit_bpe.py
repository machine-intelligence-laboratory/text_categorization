"""
Модуль для работы с данными в формате Vowpal Wabbit.
"""


import re
import typing

from collections import Counter
from string import punctuation

import youtokentome as yttm

from zhon import hanzi


class VowpalWabbitBPE:
    """Класс для сохранения VW файлов с BPE преобразованием."""
    def __init__(self, bpe_models: dict, use_counters: bool = True):
        """
        Создает класс сохранения VW файлов с BPE преобразованием.

        Args:
            bpe_models (dict): словарь с обученными BPE моделями
            use_counters (bool): признак использования каунтеров.
        """
        self._bpe_models = bpe_models
        self._use_counters = use_counters
        self.punctuation = punctuation + hanzi.punctuation

    def save_docs(
            self, target_file: str, doc: typing.Dict[str, typing.Dict[str, str]]
    ):
        """
        Конвертирует документы в BOW и сохраняет их.

        Args:
            target_file: путь к файлу.
            doc: сырые документы.
        """
        self.save_bow(target_file, self.convert_to_bow(doc))

    def save_bow(
            self, target_file: str,
            sessions_bow_messages: typing.Dict[str, typing.Dict[str, typing.Union[str, typing.Counter]]],
    ):
        """
        Сохраняет BOW представление документов.

        Args:
            target_file (str): путь к файлу.
            sessions_bow_messages (dict): doc_id -> dict: modality -> dict: token -> count.
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
                            for token, count in sessions_bow_messages[key][modality].items()
                        ]
                    )
                else:
                    modality_content = sessions_bow_messages[key][modality]
                new_message_str_format += f" |@{modality} {modality_content}"
            with open(target_file, 'a', encoding='utf8') as file:
                file.write(new_message_str_format)
                file.write('\n')

    def convert_to_bow(
            self, data: typing.Dict[str, typing.Dict[str, str]]
    ) -> typing.Dict[str, typing.Dict[str, typing.Union[str, typing.Counter]]]:
        """
        Конвертирует набор документов в BOW представление (см. VowpalWabbit.convet_doct).

        Args:
            data (dict): словарь айди документа -> документ.

        Returns:
            sessions_bow_messages (dict): словарь айди документа -> документ в виде BOW.
        """
        sessions_bow_messages = {}
        for elem_id, elem in data.items():
            sessions_bow_messages[elem_id] = self.convert_doc(elem)
        return sessions_bow_messages

    def convert_doc(
            self, doc: typing.Dict[str, str]
    ) -> typing.Dict[str, typing.Union[str, typing.Counter]]:
        """
        Конвертирует исходный документ в формат BOW.

        Args:
            doc (dict): словарь язык -> текст документа.

        Returns:
            res (dict): словарь язык -> BOW документа. Если use_counters==True, словарь в виде Counter.
        """

        res = {}

        for modality, mod_elem in doc.items():
            print(modality)
            tokens = " ".join(self._token_filtration(mod_elem))
            if modality not in ['UDK', 'GRNTI']:
                tokens = self._bpe_models[modality].encode(
                    tokens, output_type=yttm.OutputType.SUBWORD
                )
            else:
                tokens = [tokens]
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
