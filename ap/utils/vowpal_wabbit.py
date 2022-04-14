"""Класс для работы с данными в формате Vowpal Wabbit."""

import re
import typing

from collections import Counter
from string import punctuation

from zhon import hanzi


class VowpalWabbit:
    """
    Класс сохранения VW файлов.
    """
    def __init__(self, use_counters):
        """
        Создает класс сохранения VW файлов.

        Args:
            use_counters: признак использования каунтеров
        """
        self._use_counters = use_counters
        self.punctuation = punctuation + hanzi.punctuation

    def save_docs(
        self, target_file: typing.TextIO, doc: typing.Dict[str, typing.Dict[str, str]]
    ):
        """
        Конвертирует документы в BOW и сохраняет их.

        Args:
            target_file: путь к файлу
            doc: сырые документы
        """
        self._save_bow(target_file, self._convert_to_bow(doc))

    def _save_bow(
        self,
        target_file: typing.TextIO,
        sessions_bow_messages: typing.Dict[
            str, typing.Dict[str, typing.Union[str, typing.Counter]]
        ],
    ):
        """
        Сохраняет BOW представление документов.

        Args:
            target_file: путь к файлу
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
            target_file.write(new_message_str_format)
            target_file.write("\n")

    def _convert_to_bow(
        self, data: typing.Dict[str, typing.Dict[str, str]]
    ) -> typing.Dict[str, typing.Dict[str, typing.Union[str, typing.Counter]]]:
        """
        Конвертирует набор документов в BOW представление (см. VowpalWabbit.convet_doct).

        Args:
            data: словарь айди документа->документ

        Returns:
            sessions_bow_messages: словарь айди документа->документ в виде BOW
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

        Args:
            doc: словарь язык->текст документа

        Returns:
            res: словарь язык->BOW документа. Если use_counters==True, словарь в виде Counter
        """
        res = {}
        if self._use_counters:
            res = {m: Counter() for m in doc.keys()}
        else:
            res = {m: "" for m in doc.keys()}
        res["plain_text"] = ""
        if isinstance(doc.get("text"), str):
            res["plain_text"] += " ".join(doc["text"].split()) + " | "
        for modality, mod_elem in doc.items():
            tokens = self._token_filtration(mod_elem)
            if self._use_counters:
                res[modality] += Counter(tokens)
            else:
                res[modality] += " ".join(tokens)

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
