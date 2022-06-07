"""Модуль для тестирования работы ap.utils.prediction_visualization"""

import re

import artm

from pathlib import Path

from ap.utils.prediction_visualization import augment_text


def test_augment_text():
    model_path = [path
                  for path in Path('tests/data/best_model').iterdir()
                  if re.match(r'\d+_\d+', path.name)][0]
    model = artm.load_artm_model(model_path)
    input_text = 'tests/data/test_ru_bpe.txt'
    with open(input_text) as file:
        data = file.readline()
    doc_id = data.split()[0]
    tmp_dir = 'tests/data/tmp_dir'

    interpretation_info = augment_text(model, input_text, tmp_dir)

    assert interpretation_info != {}
