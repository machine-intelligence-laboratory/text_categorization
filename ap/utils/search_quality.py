"""
Модуль для вычисления качества тематической модели.
"""

import json
import typing
import warnings

from pathlib import Path

import artm
import joblib

from tqdm import tqdm

from ap.utils import rank_metric

warnings.filterwarnings('ignore')


def dump_train_centroids(model_path: str, bcg_topic_list: typing.List[str],
                         current_languages: typing.List[str], path_train_centroids: str):
    """
    Вычисляет центроиды, необходимые для преобразования текста из одного языка в другой.

    Args:
        model_path (str): путь до тематичекой модели
        bcg_topic_list (list): список тем, которые не будут использоваться для построения
        current_languages (list): названия языков, используемых для подсчёта метрик
        path_train_centroids (str): путь для выгрузки центроид
    """
    path_train_centroids = Path(path_train_centroids)
    model = artm.load_artm_model(model_path)
    sbj_topic_list = [topic for topic in model.topic_names
                      if topic not in bcg_topic_list]

    print('Calculation of train centroids were started.')
    for lang in tqdm(current_languages):
        target_folder = path_train_centroids.joinpath('tmp_batches')
        target_folder.mkdir(exist_ok=True, parents=True)
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


def calculate_search_quality(config_experiment) -> \
        typing.Dict[str, typing.Dict[str, typing.Dict[str, float]]]:
    """
    Вычисление качества модели по 6 метрикам:
        - Средняя частота УДК,
        - Средний процент УДК,
        - Средняя частота ГРНТИ,
        - Средний процент ГРНТИ,
        - Средняя частота ВАК,
        - Средний процент ВАК.

    Args:
        config_experiment (dict): конфиг эксперимента, содержащий:

            - config_experiment["path_experiment"] (str): путь до папки с экспериментом
            - config_experiment['path_model'] (str): путь до тестируемой модели,
            - config_experiment['path_results'] (str): путь для выгрузки результата эксперимента,
            - config_experiment["artm_model_params"] (dict): параметры тематической модели ARTM,
                - config_experiment["artm_model_params"]["num_bcg_topic"] (int): количество фоновых тем,
            - config_experiment["bcg_topic_list"] (typing.Optional[typing.List[str]]): список фоновых тем,
            - config_experiment['metrics_to_calculate'] (str): название меры близости ('analogy' или 'eucl'),
            - config_experiment['path_train_thetas'] (str): путь до центроид
            - config_experiment['recalculate_train_centroids'] (bool): признак необходимости вычислять центроиды
            - config_experiment['recalculate_test_thetas'] (bool): признак необходимости вычислять матрицы Тэта для \
                тестовых данных
            - config_experiment["languages_for_metric_calculation"] (list): названия языков, \
                используемых для подсчёта метрик
            - config_experiment['path_test'] (str): путь до тестовых данных
            - config_experiment['path_subsamples_list'] (list): список путей к json-файлам, содержащим \
                подвыборки индексов документов, по которым будет производиться поиск
            - config_experiment['path_rubrics_list'] (list):  список путей к json-файлам с рубриками, \
                где по doc_id содержится рубрика документа

    Returns:
        quality (dict): подсчитанные метрики качества
    """
    path_experiment = Path(config_experiment["path_experiment"])
    path_model = str(config_experiment.get('path_model', path_experiment.joinpath('topic_model')))
    path_experiment_result = str(config_experiment.get('path_results', path_experiment.joinpath('results')))
    num_bcg_topic = config_experiment["artm_model_params"]["num_bcg_topic"]
    bcg_topic_list = config_experiment.get('bcg_topic_list', [f'topic_{i}' for i in range(num_bcg_topic)])
    metrics_to_calculate = config_experiment.get('metrics_to_calculate', ['analogy'])
    path_train_centroids = config_experiment['path_train_thetas']
    recalculate_train_centroids = config_experiment.get('recalculate_train_centroids', False)
    recalculate_test_thetas = config_experiment.get('recalculate_test_thetas', True)
    path_test = config_experiment['path_test']
    path_subsamples_list = config_experiment['path_subsamples_list']
    path_rubrics_list = config_experiment['path_rubrics_list']
    current_languages = config_experiment["languages_for_metric_calculation"]

    if recalculate_train_centroids:
        dump_train_centroids(path_model, bcg_topic_list, current_languages, path_train_centroids)

    frequency = 'average_frequency_analogy'
    percent = 'average_percent_analogy'
    quality = {}

    for path_rubrics, path_subsamples in zip(path_rubrics_list, path_subsamples_list):
        quality_model = rank_metric.quality_of_models(
            path_train_centroids, bcg_topic_list,
            metrics_to_calculate,
            path_model, path_experiment_result, path_subsamples, path_rubrics,
            path_test, current_languages, recalculate_test_thetas
        )
        quality[Path(path_rubrics).stem] = {
            "average_frequency": quality_model[frequency],
            "average_percent": quality_model[percent],
        }

    with open(Path(path_experiment_result).joinpath('metrics.json'), 'w') as file:
        json.dump(quality, file)

    return quality
