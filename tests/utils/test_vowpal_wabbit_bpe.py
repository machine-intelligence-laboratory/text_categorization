import os

from collections import Counter

from ap.utils.bpe import load_bpe_models
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


def test_convert_doc_plain():
    doc = {"ru": "привет как твои дела & *", "en": "g h j k f"}
    bpe_models = load_bpe_models('tests/data/bpe')

    vw = VowpalWabbitBPE(bpe_models=bpe_models, use_counters=False)
    res = vw.convert_doc(doc)

    assert res["en"] == Counter({'▁g': 1, '▁h': 1, '▁j': 1, '▁k': 1, '▁f': 1})
    assert res["ru"] == Counter({'▁привет': 1, '▁как': 1, '▁тво': 1, 'и': 1, '▁дела': 1})


def test_convert_doc_counter():
    doc = {"ru": "привет привет как твои дела & *", "en": "g h j k f"}
    bpe_models = load_bpe_models('tests/data/bpe')

    vw = VowpalWabbitBPE(bpe_models=bpe_models, use_counters=True)
    res = vw.convert_doc(doc)

    assert res["en"]["▁g"] == 1
    assert res["en"]["▁k"] == 1

    assert res["ru"]["▁привет"] == 2
    assert res["ru"]["▁тво"] == 1
    assert res["ru"]["и"] == 1


def test_save_docs(tmp_path):
    bpe_models = load_bpe_models('tests/data/bpe')
    vw = VowpalWabbitBPE(bpe_models=bpe_models, use_counters=True)
    docs = {
        "1": {"ru": "привет как твои дела & *"},
        "2": {"en": "hi how are you"},
    }
    with open(tmp_path.joinpath("temp_vw.txt"), "w") as _:
        vw.save_docs(os.path.join(tmp_path, "temp_vw.txt"), docs)

    with open(tmp_path.joinpath("temp_vw.txt")) as file:
        res = file.readlines()
        assert len(res) == 2
