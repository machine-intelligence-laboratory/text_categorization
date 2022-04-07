from pathlib import Path

import json
import warnings

import artm
import joblib
import numpy as np

from tqdm import tqdm

import ap.utils.rank_metric as rank_metric
import ap.utils.config as config

warnings.filterwarnings('ignore')


def dump_train_centroids(model_path, bcg_topic_list, path_train_centroids):
    path_train_centroids = Path(path_train_centroids)
    model = artm.load_artm_model(model_path)
    sbj_topic_list = [topic for topic in model.topic_names
                      if topic not in bcg_topic_list]

    print('Calculation of train centroids were started.')
    for lang in tqdm(config.LANGUAGES_MAIN):
        target_folder = path_train_centroids.joinpath('tmp_batches')
        target_folder.mkdir(exist_ok=True)
        batches_list = list(target_folder.iterdir())
        if batches_list:
            for batch in batches_list:
                batch.unlink()
        batch_vectorizer = artm.BatchVectorizer(
            data_path=str(path_train_centroids.joinpath(f'{lang}', f'train_{lang}.txt')),
            data_format='vowpal_wabbit',
            target_folder=str(target_folder)
        )
        theta = model.transform(batch_vectorizer=batch_vectorizer)
        a_train = theta.loc[sbj_topic_list].values.T.mean(axis=0)
        joblib.dump(a_train, path_train_centroids.joinpath(f'{lang}', 'centroid.joblib'))

        if list(target_folder.iterdir()):
            for content in list(target_folder.iterdir()):
                content.unlink()
        target_folder.rmdir()
    print('Train centroids were calculated.')


def calculate_search_quality(config_experiment):
    path_model = Path(config_experiment['path_model'])
    model_name = str(path_model.name)
    path_experiment_result = Path(config_experiment['path_results'])

    bcg_topic_list = config_experiment.get('bcg_topic_list', ['topic_0'])
    metrics_to_calculate = config_experiment.get('metrics_to_calculate', 'analogy')
    path_train_centroids = Path(config_experiment['path_train_thetas'])
    recalculate_train_centroids = config_experiment.get('recalculate_train_centroids', False)
    recalculate_test_thetas = config_experiment.get('recalculate_test_thetas', True)

    matrix_norm_metric = np.linalg.norm
    axis = 1
    current_languages = config.LANGUAGES_MAIN

    if recalculate_train_centroids:
        dump_train_centroids(path_model, bcg_topic_list, path_train_centroids)

    path_test = Path(config.path_articles_test_bpe)
    mode = 'test'
    path_subsamples = config.path_articles_subsamples_udk
    path_rubrics = config.path_articles_rubrics_train_udk
    quality_udk = rank_metric.quality_of_models(
        path_train_centroids, bcg_topic_list,
        metrics_to_calculate, mode,
        path_model, path_experiment_result,
        matrix_norm_metric, path_subsamples, path_rubrics,
        path_test, current_languages, recalculate_test_thetas, axis=axis
    )

    path_subsamples = config.path_articles_subsamples_grnti
    path_rubrics = config.path_articles_rubrics_train_grnti
    quality_grnti = rank_metric.quality_of_models(
        path_train_centroids, bcg_topic_list,
        metrics_to_calculate, mode,
        path_model, path_experiment_result,
        matrix_norm_metric, path_subsamples, path_rubrics,
        path_test, current_languages, recalculate_test_thetas, axis=axis
    )

    current_languages = ['ru']
    path_test = Path(config.path_vak_val_raw)
    mode = 'val'
    path_subsamples = config.path_vak_subsamples
    path_rubrics = config.path_vak_rubrics
    quality_vak = rank_metric.quality_of_models(
        path_train_centroids, bcg_topic_list,
        metrics_to_calculate, mode,
        path_model, path_experiment_result,
        matrix_norm_metric, path_subsamples, path_rubrics,
        path_test, current_languages, recalculate_test_thetas=True, axis=axis
    )

    frequency = 'average_frequency_analogy'
    percent = 'average_percent_analogy'

    quality = {
        "Средняя частота УДК": quality_udk[model_name][frequency],
        "Средний процент УДК": quality_udk[model_name][percent],
        "Средняя частота ГРНТИ": quality_grnti[model_name][frequency],
        "Средний процент ГРНТИ": quality_grnti[model_name][percent],
        "Средняя частота ВАК": quality_vak[model_name][frequency],
        "Средний процент ВАК": quality_vak[model_name][percent],
    }

    with open(path_experiment_result.joinpath(model_name + '.json'), 'w') as file:
        json.dump(quality, file)

    return quality
