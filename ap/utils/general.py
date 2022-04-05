import os
import typing

from pathlib import Path

from ap.topic_model.v1.TopicModelBase_pb2 import DocId, DocumentPack


def id_to_str(id: DocId) -> str:
    """
    Конвертирует DocId в строку.

    Args:
        id (DocId): id документа

    Returns:
        Строка из DocId
    """
    return f"{id.Hi}_{id.Lo}"


def docs_from_pack(pack: DocumentPack) -> typing.Dict[str, typing.Dict[str, str]]:
    """
    Создает dict документов из DocumentPack.

    Args:
        pack (DocumentPack): TODO

    Returns:
        dict из документов
    """
    return {
        id_to_str(doc.Id): {doc.Language: " ".join(doc.Tokens)}
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
    Рекурсивное удаление файлов и директорий

    Args:
        path (Path): путь, по которому необходимо удалить все файлы.
    """
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            recursively_unlink(child)
    path.rmdir()
