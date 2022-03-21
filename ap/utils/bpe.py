import os

import youtokentome as yttm


def load_bpe_models(bpe_path):
    res = {}
    for model in os.listdir(bpe_path):
        res[model.split("_")[2]] = yttm.BPE(
            model=os.path.join(bpe_path, model), n_threads=-1
        )

    return res
