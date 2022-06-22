import re

from pathlib import Path

import yaml


from ap.utils.search_quality import calculate_search_quality, dump_train_centroids


def test_calculate_search_quality():
    with open('tests/data/test_config.yml') as file:
        config = yaml.safe_load(file)

    DATA_PATH = 'tests/data/'
    model_path = [path
                  for path in Path('tests/data/best_model').iterdir()
                  if re.match(r'\d+_\d+', path.name)][0]

    config['path_model'] = model_path
    config['path_test'] = DATA_PATH + '/test_BPE/'
    config['path_subsamples_list'] = [DATA_PATH + 'subsamples_15_udk/']
    config['path_rubrics_list'] = [DATA_PATH + 'udk_codes.json']
    config['languages_for_metric_calculation'] = [
        'ru',
        'it',
    ]

    search_quality = calculate_search_quality(config)

    assert isinstance(search_quality['udk_codes']['average_frequency'], float)
    assert isinstance(search_quality['udk_codes']['average_percent'], float)


def test_dump_train_centroids():
    with open('tests/data/test_config.yml') as file:
        config = yaml.safe_load(file)

    path_model = str([path
                  for path in Path('tests/data/best_model').iterdir()
                  if re.match(r'\d+_\d+', path.name)][0])
    bcg_topic_list = [f'topic_{i}' for i in range(config['artm_model_params']['num_bcg_topic'])]
    current_languages = ['ru', 'it']
    path_train_centroids = 'tests/data/train_by_lang'
    dump_train_centroids(path_model, bcg_topic_list, current_languages, path_train_centroids)
