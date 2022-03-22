import os
import typing

import artm


def get_num_entries(dictionary: artm.Dictionary) -> int:
    """
    Возвращает размер словаря.

    Parameters
    ----------
    dictionary словарь

    Returns
    -------
    количество токенов в словаре
    """
    return next(
        x for x in dictionary._master.get_info().dictionary if x.name == dictionary.name
    ).num_entries


def limit_classwise(
    dictionary: artm.Dictionary,
    cls_ids: typing.Iterable[str],
    max_dictionary_size: int,
    tmp_dir: str,
    out_file: str,
):
    """
    Ограничивает словарь.

    Таким образом, что в разрезе каждоого class id будет не более max_dictionary_size токенов.
    Сохраняет словарь в текстовым форматом в файле out_file.

    Parameters
    ----------
    dictionary исходный словарь
    cls_ids модальности
    max_dictionary_size максимальный размер словаря в разрезе модальности
    tmp_dir директория для хранения промежуточных результатов
    out_file файл, в котором сохраняется результат
    """
    for cls_id in cls_ids:
        filtered = dictionary
        inplace = False
        for other_id in cls_ids:
            if other_id != cls_id:
                filtered = filtered.filter(
                    class_id=other_id, max_df_rate=0.4, min_df_rate=0.5, inplace=inplace
                )
                inplace = True
        filtered.filter(max_dictionary_size=max_dictionary_size)
        filtered.save_text(os.path.join(tmp_dir, f"{cls_id[1:]}.txt"))

    res = []
    for cls_id in cls_ids:
        with open(os.path.join(tmp_dir, f"{cls_id[1:]}.txt")) as file:
            res.extend(file.readlines()[2:] if len(res) > 0 else file.readlines())

    with open(out_file, "w") as file:
        file.write("".join(res))
