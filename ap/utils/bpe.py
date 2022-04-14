import os

import youtokentome as yttm


def load_bpe_models(bpe_path: str) -> dict:
    """
    Загружает обученные BPE модели.

    Args:
        bpe_path (str): путь до папки с обученными BPE моделями.
    """
    res = {}
    for model in os.listdir(bpe_path):
        res[model.split("_")[2]] = yttm.BPE(
            model=os.path.join(bpe_path, model), n_threads=-1
        )

    return res
