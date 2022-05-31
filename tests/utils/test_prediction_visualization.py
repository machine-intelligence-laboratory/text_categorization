"""Модуль для тестирования работы ap.utils.prediction_visualization"""

import artm

from ap.utils.prediction_visualization import augment_text


def test_augment_text():
    model = artm.load_artm_model('tests/data/model')
    input_text = 'tests/data/test_ru.txt'
    with open(input_text) as file:
        data = file.readline()
    doc_id = data.split()[0]
    tmp_dir = 'tests/data/tmp_dir'

    interpretation_info = augment_text(model, input_text, tmp_dir)
    print(interpretation_info[doc_id]['topic_from'])
    print(interpretation_info[doc_id]['topic_to'])
    print(interpretation_info[doc_id]['Added'])
    print(interpretation_info[doc_id]['Removed'])

    assert interpretation_info != {}
