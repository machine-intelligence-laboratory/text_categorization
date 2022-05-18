import os
import typing

from pathlib import Path


def id_to_str(id) -> str:
    """
    Конвертирует DocId в строку.

    Args:
        id (DocId): id документа

    Returns:
        Строка из DocId
    """
    return f"{id.Hi}_{id.Lo}"


def get_modalities(doc):
    """
    Возвращает словарь, где по языку хранятся токены на этом языке.

    Args:
        doc: документ

    Returns:
        res (dict): словарь, ключ -- язык, значение -- строка токенов на этом языке
            через пробел
    """
    res = {}

    for modality in doc.Modalities:
        if modality.Key == 'lang':
            res[modality.Value] = " ".join(doc.Tokens)
        else:
            res[modality.Key] = modality.Value

    return res


def docs_from_pack(pack) -> typing.Dict[str, typing.Dict[str, str]]:
    """
    Создает dict документов из pack.

    Args:
        pack (ap.topic_model.v1.TopicModelBase_pb2.DocumentPack): TODO

    Returns:
        dict из документов
    """


    return {
        id_to_str(doc.Id): get_modalities(doc)
        for doc in pack.Documents
    }


def ensure_directory(path: str) -> str:
    """
    Создает директорию, если ее нет, и возвращает path.

    Args:
        path (str): путь к директории

    Returns:
        path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def batch_names(starts_from, count) -> typing.Generator[str, None, None]:
    """
    Генерирует названия батчей в соответствие с форматом BatchVectorizer.

    Args:
        starts_from: название файла последнего батча в директории
        count: количество батчей

    Returns:
        (typing.Generator[str, None, None]): Генератор имен батчей
    """
    orda = ord("a")
    letters = 26
    starts_from_int = sum(
        [letters ** i * (ord(x) - orda) for i, x in enumerate(reversed(starts_from))]
    )

    for x in range(starts_from_int + 1, starts_from_int + 1 + count):
        str_name = []
        for _ in range(len(starts_from)):
            str_name.append(chr(x % letters + orda))
            x = int(x / letters)

        yield "".join(reversed(str_name))


def recursively_unlink(path: Path):
    """
    Рекурсивно удаляет файлы и директории.

    Args:
        path (Path): путь, по которому необходимо удалить все файлы.
    """
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            recursively_unlink(child)
    path.rmdir()
