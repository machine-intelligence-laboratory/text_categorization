import json
import re

import pandas as pd

from pathlib import Path

import artm
import yaml

from ap.utils import rank_metric


def test_quality_of_models():
    with open('tests/data/test_config.yml') as file:
        config = yaml.safe_load(file)

    path_model = str([path
                      for path in Path('tests/data/best_model').iterdir()
                      if re.match(r'\d+_\d+', path.name)][0])
    bcg_topic_list = [f'topic_{i}' for i in range(config['artm_model_params']['num_bcg_topic'])]
    path_train_centroids = 'tests/data/train_by_lang'
    metrics_to_calculate = ['analogy']
    DATA_PATH = 'tests/data/'
    path_subsamples = DATA_PATH + 'subsamples_15_udk/'
    path_rubrics = DATA_PATH + 'udk_codes.json'
    path_test = DATA_PATH + '/test_BPE/'
    current_languages = ['ru', 'it']
    recalculate_test_thetas = True
    path_experiment = Path(config['path_experiment'])
    path_experiment_result = str(config.get('path_results', path_experiment.joinpath('results')))

    quality_model = rank_metric.quality_of_models(
        path_train_centroids, bcg_topic_list,
        metrics_to_calculate,
        path_model, path_experiment_result, path_subsamples, path_rubrics,
        path_test, current_languages, recalculate_test_thetas
    )
    assert 'average_frequency_analogy' in quality_model
    assert 'average_percent_analogy' in quality_model


def test_rank():
    with open('tests/data/test_config.yml') as file:
        config = yaml.safe_load(file)
    lang_one = 'ru'
    lang_two = 'it'
    current_languages = [lang_one, lang_two]
    model_path = [path
                  for path in Path('tests/data/best_model').iterdir()
                  if re.match(r'\d+_\d+', path.name)][0]
    model = artm.load_artm_model(str(model_path))
    bcg_topic_list = ['topic_0']
    sbj_topic_list = [topic for topic in model.topic_names
                      if topic not in bcg_topic_list]
    metrics_to_calculate = ['analogy']
    DATA_PATH = 'tests/data/'
    path_subsamples = DATA_PATH + 'subsamples_15_udk/'
    path_rubrics = DATA_PATH + 'udk_codes.json'
    path_test = DATA_PATH + '/test_BPE/'

    recalculate_test_thetas = False
    path_experiment = Path(config['path_experiment'])
    path_experiment_result = config.get('path_results', path_experiment.joinpath('results'))
    path_thetas = str(path_experiment_result.joinpath('theta_lang.joblib'))
    rbm = rank_metric.RankingByModel(
        bcg_topic_list, metrics_to_calculate,
        str(model_path), path_subsamples, path_rubrics
    )

    theta_lang = rbm.get_thetas(path_test, path_thetas,
                                current_languages, recalculate_test_thetas)
    data_lang_one = theta_lang[lang_one]
    data_lang_two = theta_lang[lang_two]
    idx_one, theta_one = data_lang_one.columns, data_lang_one
    idx_two, theta_two = data_lang_two.columns, data_lang_two

    theta_one = theta_one.loc[sbj_topic_list]
    theta_two = theta_two.loc[sbj_topic_list]

    with open(path_rubrics) as file:
        rubrics = json.load(file)
    search_indices = idx_one.intersection(idx_two)
    search_indices = [doc_id for doc_id in search_indices
                      if doc_id in rubrics]
    search_indices = pd.Index(search_indices)

    path_train_centroids = 'tests/data/train_by_lang'

    path_train_lang = Path(path_train_centroids)
    metrics_first_lang = rbm._rank(
        path_train_lang, search_indices, lang_one, lang_two,
        theta_one[search_indices], theta_two[search_indices]
    )

    assert 'analogy' in metrics_first_lang
    assert 'percent_same_rubric' in metrics_first_lang['analogy']
    assert 'average_position' in metrics_first_lang['analogy']
