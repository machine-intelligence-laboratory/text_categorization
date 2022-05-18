import json
import os

from itertools import combinations_with_replacement
from pathlib import Path

import artm
import joblib
import numpy as np
import pandas as pd
import typing

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class RankingByModel:
    def __init__(
            self, bcg_topic_list: list, metrics_to_calculate: typing.List[str],
            model_path, matrix_norm_metric, path_subsamples, path_rubrics,
            mode, **kwargs
    ):
        """
        Class for ranking document search between language pairs.

        Args:
            bcg_topic_list (list): список фоновых тем тематической модели
            metrics_to_calculate (list of str ('analogy', 'eucl')): список названий метрик близости
                для использования в ранжировании
            model_path (str): путь тематической модели
            matrix_norm_metric (callable): способ измерения нормы матрицы векторов
                например: matrix_norm_metric = np.linalg.norm
            path_subsamples (str): путь к файлу в формате json, содержащему подвыборки индексов документов,
                по которым будет производиться поиск
            path_rubrics (str): путь к json-файлу, где по doc_id содержится его рубрика
            mode (str): тип данных, например 'test'
        """
        self._model = artm.load_artm_model(model_path)
        self._metrics_to_calculate = metrics_to_calculate
        self._sbj_topic_list = [topic for topic in self._model.topic_names
                                if topic not in bcg_topic_list]
        self._matrix_norm_metric = matrix_norm_metric
        self._path_subsamples = path_subsamples
        with open(path_rubrics) as file:
            self._rubrics = json.load(file)
        self._rubric_docs = {rubric: [] for rubric in set(self._rubrics.values())}
        for doc_id, rubric in self._rubrics.items():
            self._rubric_docs[rubric].append(doc_id)
        self._axis = kwargs.get('axis', 1)
        self._mode = mode

    def get_thetas(self, path_test, path_thetas, current_languages, recalculate_test_thetas=True):
        if not recalculate_test_thetas and path_thetas.exists():
            theta_lang = joblib.load(path_thetas)
            print('Existing thetas of test data were loaded.')
        else:
            theta_lang = dict()
            for lang in current_languages:
                data_lang_path = str(path_test.joinpath(
                    f'{self._mode}_{lang}_120k.txt'))
                _, theta = self._get_document_embeddings(data_lang_path)
                theta_lang[lang] = theta
                joblib.dump(theta_lang, path_thetas)
            print('Thetas of test data were calculated.')
        return theta_lang

    def _get_document_embeddings(self, path_to_data):
        if os.path.isfile(path_to_data):
            path_to_batches = os.path.join(
                os.path.dirname(path_to_data), Path(path_to_data).stem + "_rank_batches"
            )
            batch_vectorizer = artm.BatchVectorizer(
                data_path=path_to_data,
                data_format="vowpal_wabbit",
                target_folder=path_to_batches,
            )

        elif len(os.listdir(path_to_data)) > 0:
            batch_vectorizer = artm.BatchVectorizer(data_path=path_to_data, data_format="batches")
        else:
            raise ValueError("Unknown data format")

        theta = self._model.transform(batch_vectorizer=batch_vectorizer)
        theta = theta.loc[self._sbj_topic_list]
        return theta.columns, theta

    def _metrics_on_analogy_similarity(
            self, theta_original,
            doc_id, top_10_percent,
            a_train, a_z_train,
            subsample_for_doc_id, vectors_source
    ):

        b = theta_original[doc_id].values
        similarity = cosine_similarity(vectors_source, [a_z_train - a_train + b]).T[0]

        rating = subsample_for_doc_id[np.argsort(similarity)][::-1]
        top_rating = rating[:len(top_10_percent)]
        num_same_rubric = sum([self._rubrics[current_id] == self._rubrics[doc_id]
                               for current_id in top_rating])

        percent_same_rubric = num_same_rubric / len(top_10_percent)
        position = np.argwhere(rating == doc_id)[0][0]
        count = 1 if position < len(top_10_percent) else 0

        return percent_same_rubric, count

    def _metrics_on_eucl_similarity(
            self,
            search_num, search_indices, doc_id, top_10_percent,
            subsample_for_doc_id, vectors_original, vectors_source
    ):
        difference = vectors_source - vectors_original[search_num]
        vectors_norm = self._matrix_norm_metric(difference, self._axis)

        rating = subsample_for_doc_id[np.argsort(vectors_norm)]
        top_rating = rating[:len(top_10_percent)]
        num_same_rubric = sum([self._rubrics[current_id] == self._rubrics[doc_id]
                               for current_id in top_rating])
        percent_same_rubric = num_same_rubric / len(top_10_percent)
        position = np.argwhere(rating == search_indices[search_num])[0][0]
        count = 1 if position < len(top_10_percent) else 0

        return percent_same_rubric, count

    def _rank(self, path_train_lang, search_indices, lang_original, lang_source,
              theta_original, theta_source):
        average_position = dict()
        percent_same_rubric = dict()
        for metric in self._metrics_to_calculate:
            average_position[metric] = list()
            percent_same_rubric[metric] = list()
        with open(Path(self._path_subsamples).joinpath(
                lang_original, f'{lang_source}.json')) as file:
            subsamples = json.load(file)
        search_indices = list(set(search_indices).intersection(set(subsamples)))
        a_train = joblib.load(path_train_lang.joinpath(f'{lang_original}', 'centroid.joblib'))
        a_z_train = joblib.load(path_train_lang.joinpath(f'{lang_source}', 'centroid.joblib'))

        for search_num, doc_id in enumerate(search_indices):
            doc_id = search_indices[search_num]
            top_10_percent, bottom_90_percent = subsamples[doc_id]
            subsample_for_doc_id = pd.Index(top_10_percent + bottom_90_percent)

            vectors_original = theta_original.values.T
            vectors_source = theta_source[subsample_for_doc_id].values.T

            # косинусная близость
            if 'analogy' in self._metrics_to_calculate:
                metric = 'analogy'
                percent, count = self._metrics_on_analogy_similarity(
                    theta_original, doc_id, top_10_percent,
                    a_train, a_z_train,
                    subsample_for_doc_id, vectors_source
                )
                percent_same_rubric[metric].append(percent)
                average_position[metric].append(count)

            # евклидово расстроение
            if 'eucl' in self._metrics_to_calculate:
                metric = 'eucl'
                percent, count = self._metrics_on_eucl_similarity(
                    search_num, search_indices, doc_id, top_10_percent,
                    subsample_for_doc_id, vectors_original, vectors_source
                )
                percent_same_rubric[metric].append(percent)
                average_position[metric].append(count)

        metrics = dict()
        for metric in self._metrics_to_calculate:
            metrics[metric] = {
                'percent_same_rubric': np.mean(percent_same_rubric[metric]),
                'average_position': np.mean(average_position[metric])
            }
        return metrics

    def get_ranking(self, path_train_lang: str, lang_one: str, lang_two: str,
                    data_lang_one: str, data_lang_two: str):
        """
        Function returning average position of search documents in the other language.

        Parameters:
            path_train_lang: str
                path to train thetas by languages
            lang_one: str
                name of the first language
            lang_two: str
                name of the second language
            data_lang_one: str
                path to folder with batches or path to vw file
            data_lang_two: str
                path to folder with batches or path to vw file

        Returns:
            metrics
            len(search_indices)
        """
        if isinstance(data_lang_one, pd.DataFrame):
            idx_one, theta_one = data_lang_one.columns, data_lang_one
        else:
            idx_one, theta_one = self._get_document_embeddings(data_lang_one)
        if isinstance(data_lang_two, pd.DataFrame):
            idx_two, theta_two = data_lang_two.columns, data_lang_two
        else:
            idx_two, theta_two = self._get_document_embeddings(data_lang_two)

        assert len(idx_one) == len(set(idx_one))
        assert len(idx_two) == len(set(idx_two))

        search_indices = idx_one.intersection(idx_two)
        search_indices = [doc_id for doc_id in search_indices
                          if doc_id in self._rubrics]
        search_indices = pd.Index(search_indices)

        theta_one = theta_one.loc[self._sbj_topic_list]
        theta_two = theta_two.loc[self._sbj_topic_list]

        metrics_first_lang = self._rank(
            path_train_lang, search_indices, lang_one, lang_two,
            theta_one[search_indices], theta_two[search_indices]
        )
        metrics_second_lang = self._rank(
            path_train_lang, search_indices, lang_two, lang_one,
            theta_two[search_indices], theta_one[search_indices]
        )

        metrics = dict()
        for metric in self._metrics_to_calculate:
            metrics[metric] = {
                'percent_same_rubric': {
                    'first_lang': metrics_first_lang[metric]['percent_same_rubric'],
                    'second_lang': metrics_second_lang[metric]['percent_same_rubric']
                },
                'average_position': {
                    'first_lang': metrics_first_lang[metric]['average_position'],
                    'second_lang': metrics_second_lang[metric]['average_position']
                }
            }

        return metrics, len(search_indices)

    def metrics_to_df(self, path_train_lang, path_experiment_result,
                      current_languages, theta_lang, rbm):
        percent = dict()
        frequency = dict()
        for metric in self._metrics_to_calculate:
            percent[metric] = pd.DataFrame(index=current_languages, columns=current_languages)
            frequency[metric] = pd.DataFrame(index=current_languages, columns=current_languages)
        intersections = pd.DataFrame(index=current_languages, columns=current_languages)

        lang_pairs = list(combinations_with_replacement(current_languages, r=2))
        for lang_0, lang_1 in tqdm(lang_pairs, total=len(lang_pairs)):
            metrics, intersection = rbm.get_ranking(
                path_train_lang,
                lang_0, lang_1, theta_lang[lang_0], theta_lang[lang_1]
            )
            for metric in self._metrics_to_calculate:
                percent[metric].loc[lang_0, lang_1] = round(
                    metrics[metric]['percent_same_rubric']['first_lang'], 3
                )
                percent[metric].loc[lang_1, lang_0] = round(
                    metrics[metric]['percent_same_rubric']['second_lang'], 3
                )
                frequency[metric].loc[lang_0, lang_1] = round(
                    metrics[metric]['average_position']['first_lang'], 3
                )
                frequency[metric].loc[lang_1, lang_0] = round(
                    metrics[metric]['average_position']['second_lang'], 3
                )

                intersections.loc[lang_0, lang_1] = intersection
                intersections.loc[lang_1, lang_0] = intersection

        for metric in self._metrics_to_calculate:
            frequency[metric].to_csv(path_experiment_result.joinpath(f'frequency_{metric}.csv'))
            percent[metric].to_csv(path_experiment_result.joinpath(f'percent_{metric}.csv'))
        intersections.to_csv(path_experiment_result.joinpath('intersections.csv'))

        return percent, frequency, intersections


def quality_of_models(path_train_lang, bcg_topic_list,
                      metrics_to_calculate, mode,
                      path_model, path_experiment_result,
                      matrix_norm_metric, path_subsamples, path_rubrics,
                      path_test, current_languages, recalculate_test_thetas, **kwargs):
    """
    Function to calculate quality of models.

    Args:
        path_train_lang (str):
            path to train thetas by languages
        bcg_topic_list (list of str):
            backgroung topics of a topic model
            for example init as:
            bcg_topic_list = ['topic_0']
        metrics_to_calculate (list of str ('analogy', 'eucl')):
            list of names of proximity measures to use in ranking
        path_model (str):
            path to models
        path_experiment_result (pathlib.Path):
            path to the folder to save results of models
        matrix_norm_metric (callable):
            a way to measure norm of a matrix of vectors
            for example init as:
            matrix_norm_metric = np.linalg.norm
        path_subsamples (str):
            path to folder with subsemples of different languages in json format
        path_rubrics (str):
            path to file with rubrics of documents in json format
        path_test (str):
            path to folder with txt files for calculate metrics
        current_languages (list of str):
            names of languages to use to calculate metrics
        recalculate_test_thetas (bool):
            True means rebuild thetas
            False means load existing thetas

    Returns:
        quality_experiment (dict): словарь словарей - по имени модели хранятся названия метрик,
                                    по названиям метрик хранятся значения метрик этих моделей
    """
    quality_experiment = dict()
    quality_experiment[path_model.name] = dict()
    path_model_result = path_experiment_result.joinpath(path_model.name)
    path_model_result.mkdir(parents=True, exist_ok=True)
    path_thetas = path_model_result.joinpath('theta_lang.joblib')

    rbm = RankingByModel(
        bcg_topic_list, metrics_to_calculate,
        path_model, matrix_norm_metric, path_subsamples, path_rubrics, mode, kwargs=kwargs
    )
    theta_lang = rbm.get_thetas(path_test, path_thetas, current_languages, recalculate_test_thetas)

    percent, frequency, _ = rbm.metrics_to_df(
        path_train_lang, path_model_result, current_languages, theta_lang, rbm
    )

    # Вычисляю значения метрик
    for metric in metrics_to_calculate:
        average_frequency = frequency[metric].sum().sum() / frequency[metric].count().sum()
        average_percent = percent[metric].sum().sum() / percent[metric].count().sum()
        quality_experiment[path_model.name][
            f'average_frequency_{metric}'] = average_frequency
        quality_experiment[path_model.name][
            f'average_percent_{metric}'] = average_percent
    print(quality_experiment[path_model.name])

    with open(path_model_result.joinpath('metrics.json'), 'w') as file:
        json.dump(quality_experiment[path_model.name], file)

    with open(path_experiment_result.joinpath('metrics.json'), 'w') as file:
        json.dump(quality_experiment, file)

    return quality_experiment
