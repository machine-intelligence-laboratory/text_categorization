from pathlib import Path

import artm

from ap.utils.dictionary import get_num_entries, limit_classwise


def test_get_num_entries():
    "Тестирование функции get_num_entries"
    dictionary = artm.Dictionary()
    dictionary.load_text('tests/data/dictionary.txt')

    dictionary_size = get_num_entries(dictionary)

    assert dictionary_size == 332453


def test_limit_classwise_limit_size():
    "Проверка того что limit_classwise урезает словарь."

    dictionary: artm.Dictionary = artm.Dictionary()
    dictionary.load_text('tests/data/dictionary.txt')

    out_file = 'tests/data/dictionary_limited.txt'
    tmp_dir = 'tests/data/'
    max_dictionary_size = 20
    cls_ids = ['ru', 'en', 'de']
    limit_classwise(dictionary, cls_ids=cls_ids, max_dictionary_size=max_dictionary_size,
                    tmp_dir=tmp_dir, out_file=out_file)

    with open(out_file) as file:
        dictionary_limited = file.readlines()

    for cls_id in cls_ids:
        assert dictionary_limited.count(cls_id) <= max_dictionary_size
        assert dictionary_limited.count('cs') == 0
        assert dictionary_limited.count('it') == 0
        assert dictionary_limited.count('fr') == 0
        assert dictionary_limited.count('pl') == 0
        assert dictionary_limited.count('zh') == 0
        Path(tmp_dir).joinpath(cls_id + '.txt').unlink()

    Path(out_file).unlink()
