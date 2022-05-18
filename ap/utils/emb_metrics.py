import json
import os

from itertools import product
from pathlib import Path

import artm
import joblib
import numpy as np
import pandas as pd
import typing

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


import ap.utils.config as config


def generate_theta(path_models: str, save_path: str):
    """
    Создает и сохраняет матрицы Тэта.

    Args:
        path_models (str): путь к папке с тематическими моделями
        save_path (str): путь для сохранения матриц Тэта
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    for model in tqdm(os.listdir(path_models)):
        save_path_theta = save_path.joinpath(model)
        save_path_theta.mkdir(parents=True, exist_ok=True)
        model_artm = artm.load_artm_model(os.path.join(path_models, model))

        for lang in config.LANGUAGES_MAIN:
            test_path = '/data/datasets/Antiplagiat/texts_vw/test_BPE/' \
                        f'test_test/test_{lang}_120k_rank_batches'
            batch_vectorizer = artm.BatchVectorizer(data_path=test_path, data_format="batches",)
            vec = model_artm.transform(batch_vectorizer=batch_vectorizer).T
            joblib.dump(vec, save_path_theta.joinpath(f'theta.{lang}'))


def _check_rank_quality(vec_first, vec_second):
    max_val = sum([1/i for i in range(1, vec_first.shape[1]+1)])

    docs = list(set(vec_first.index).intersection(set(vec_second.index)))
    all_cos = list()
    if docs:
        for doc in docs:
            a = vec_first.loc[doc].values.argsort()
            b = vec_second.loc[doc].values.argsort()

            ranks_a = np.empty_like(a)
            ranks_a[a] = np.arange(len(a))
            ranks_a += 1

            ranks_b = np.empty_like(b)
            ranks_b[b] = np.arange(len(b))
            ranks_b += 1

            all_cos += [np.sum(1 / np.c_[ranks_a, ranks_b].mean(axis=1)) / max_val]

        mean_cos = np.mean(all_cos)

        return mean_cos
    return 0


def _check_analogy_quality(vec_first, vec_second, relevant=True):
    a = vec_first.values.mean(axis=0)
    a_z = vec_second.values.mean(axis=0)

    docs = list(set(vec_first.index.values).intersection(set(vec_second.index.values)))

    if docs:
        all_cos = list()
        for doc in docs:
            b = vec_first.loc[doc].values
            if relevant:
                b_z = vec_second.loc[doc].values
                all_cos += [cosine_similarity([b_z], [a_z - a + b])[0][0]]
            else:
                b_z = vec_second.loc[~vec_second.index.isin([doc])].values
                all_cos += [cosine_similarity(b_z, [a_z - a + b]).mean()]
        return all_cos
    return list()


def _check_cluster_similarity_quality(class_json_path, vec, in_class=True):
    with open(class_json_path, 'r') as file:
        classes = json.loads(file.read())
    doc_classes = dict()

    class_cos = dict()
    docs = set(list(classes)).intersection(vec.index.values)

    for doc in docs:
        if classes[doc] in doc_classes:
            doc_classes[classes[doc]] += [doc]
        else:
            doc_classes[classes[doc]] = [doc]

    for rubric, doc_list in doc_classes.items():
        center = vec.loc[doc_list].values.mean(axis=0)
        if in_class:
            class_cos[rubric] = cosine_similarity(
                [center], vec.loc[doc_list].values
            )[0]
        else:
            class_cos[rubric] = cosine_similarity(
                [center], vec.loc[~vec.index.isin(doc_list)].values
            )[0]

    return class_cos


def _check_cluster_intersection_quality(class_json_path, vec):
    with open(class_json_path, 'r') as file:
        classes = json.loads(file.read())
    doc_classes = dict()

    mean_intersection = dict()

    docs = set(list(classes)).intersection(vec.index.values)

    for doc in docs:
        if classes[doc] in doc_classes:
            doc_classes[classes[doc]] += [doc]
        else:
            doc_classes[classes[doc]] = [doc]

    for rubric, doc_list in doc_classes.items():
        center = vec.loc[doc_list].values.mean(axis=0)
        mean_cos = cosine_similarity([center], vec.loc[doc_list].values)[0].mean()

        cos_values = cosine_similarity(
            [center], vec.loc[~vec.index.isin(doc_list)].values
        )[0]
        elements = cos_values[np.where(cos_values >= mean_cos)].shape[0]
        mean_intersection[rubric] = [elements/len(doc_list)]

    return mean_intersection


def get_topic_profile(path_models, save_path):
    """
    Function to get topic profiles for models.

    Args:
        path_models (str): путь к папке с тематическими моделями
        save_path (str): путь для сохранения матриц Тэта

    Returns:
        (pd.DataFrame): DataFrame with topic profiles for models
    """
    mean_rank = list()

    for model in tqdm(os.listdir(path_models)):
        qual_df_rank = pd.DataFrame(columns=config.LANGUAGES_MAIN, index=config.LANGUAGES_MAIN)

        for l_first, l_second in product(config.LANGUAGES_MAIN, config.LANGUAGES_MAIN):
            if l_first == l_second:
                pass
            else:
                vec_first = joblib.load(os.path.join(save_path, f'{model}/theta.{l_first}'))
                vec_second = joblib.load(os.path.join(save_path, f'{model}/theta.{l_second}'))
                qual_df_rank.loc[l_first, l_second] = _check_rank_quality(vec_first, vec_second)

        mean_rank += [qual_df_rank.mean()]

    return pd.concat(mean_rank, axis=1).rename(
        columns=dict(enumerate(os.listdir(path_models)))).T


def get_mean_classes_intersection(path_models, save_path, path_categories):
    """
    Function to evaluate classes intersection.

    Args:
        path_models (str): путь к папке с тематическими моделями
        save_path (str): путь для сохранения матриц Тэта
        path_categories (str): json-файл с рубриками документов

    Returns:
        (pd.DataFrame): DataFrame with evaluation of classes intersection
    """
    mean_intersect = list()

    for model in tqdm(os.listdir(path_models)):
        for lang in config.LANGUAGES_MAIN:
            vec = joblib.load(os.path.join(save_path, f'{model}/theta.{lang}'))
            mean_intersect += [
                pd.DataFrame(
                    _check_cluster_intersection_quality(path_categories, vec),
                    index=['_'.join([model, lang])]
                )
            ]
    return pd.concat(mean_intersect, axis=0).rename(
        columns=dict(enumerate(os.listdir(path_models))))


def get_analogy_distribution(path_models: str, save_path: str) -> typing.Dict[str, dict]:
    """
    Function to get analogy distribution for models.

    Args:
        path_models  (str): путь к папке с тематическими моделями
        save_path (str): путь для сохранения матриц Тэта

    Returns:
        pair_analogy (dict): dict with distribution of analogy measure
    """
    pair_analogy = dict()

    for model in tqdm(os.listdir(path_models)):
        for l_first, l_second in product(config.LANGUAGES_MAIN, config.LANGUAGES_MAIN):
            if l_first == l_second:
                pass
            else:
                vec_first = joblib.load(os.path.join(save_path, f'{model}/theta.{l_first}'))
                vec_second = joblib.load(os.path.join(save_path, f'{model}/theta.{l_second}'))
                pair_analogy['_'.join([model, l_first, l_second])] = {
                    'relevant': _check_analogy_quality(vec_first, vec_second),
                    'not_relevant': _check_analogy_quality(vec_first, vec_second, relevant=False),
                }

    return pair_analogy


def get_cos_distribution(path_models: str, save_path: str, path_categories: str) -> typing.Dict[str, dict]:
    """
    Function to get cosine distribution in classes for models.

    Args:
        path_models (str): путь к папке с тематическими моделями
        save_path (str): путь для сохранения матриц Тэта
        path_categories (str): json-файл с рубриками документов

    Returns:
        pair_cos (dict): dict with distribution of cosine measure for classes
    """
    pair_cos = dict()

    for model in tqdm(os.listdir(path_models)):
        for lang in config.LANGUAGES_MAIN:
            vec = joblib.load(os.path.join(save_path, f'{model}/theta.{lang}'))
            pair_cos['_'.join([model, lang])] = {
                'in_class': _check_cluster_similarity_quality(path_categories, vec),
                'not_in_class': _check_cluster_similarity_quality(
                    path_categories, vec, in_class=False
                ),
            }

    return pair_cos
