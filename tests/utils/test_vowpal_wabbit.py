import os

from ap.utils.vowpal_wabbit import VowpalWabbit


def test_convert_doc_plain():
    doc = {"text": "a a b c & *", "en": "g h j k f"}

    vw = VowpalWabbit(False)
    res = vw.convert_doc(doc)

    assert res["en"] == "g h j k f"
    assert res["text"] == "a a b c"
    assert "plain_text" in res


def test_convert_doc_counter():
    doc = {"text": "a a b c & *", "en": "g h j k f"}

    vw = VowpalWabbit(True)
    res = vw.convert_doc(doc)

    assert res["en"]["g"] == 1
    assert res["en"]["k"] == 1

    assert res["text"]["a"] == 2
    assert res["text"]["b"] == 1

    assert "plain_text" in res


def test_save_docs(tmp_path):
    vw = VowpalWabbit(True)
    docs = {
        "1": {"text": "a a b c & *", "en": "g h j k f"},
        "2": {"text": "a a b ff c & *", "fr": "g h j k f"},
    }
    with open(os.path.join(tmp_path, "temp_vw"), "w") as f:
        vw.save_docs(f, docs)

    with open(os.path.join(tmp_path, "temp_vw")) as f:
        res = f.readlines()
        assert len(res) == 2
