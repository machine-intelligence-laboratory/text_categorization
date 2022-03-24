import json
import itertools
import pprint
import typing

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import artm
import joblib
import numpy as np

from nip import load
from topicnet.cooking_machine import rel_toolbox_lite
from tqdm import tqdm
from search_quality import calculate_search_quality


def _create_init_model(experiment_config) -> artm.artm_model.ARTM:
    """
    Creating an initial topic model.

    Returns
    -------
    model: artm.ARTM
        initial artm topic model with parameters from experiment_config
    """
    artm_model_params = experiment_config["artm_model_params"]

    dictionary = artm.Dictionary()
    dictionary.load_text(experiment_config["dictionary_path"])

    background_topic_list = [f'topic_{i}' for i in range(artm_model_params["num_bcg_topic"])]
    subject_topic_list = [
        f'topic_{i}' for i in range(
            artm_model_params["num_bcg_topic"],
            artm_model_params["NUM_TOPICS"]-artm_model_params["num_bcg_topic"])
    ]

    model = artm.ARTM(num_topics=artm_model_params["NUM_TOPICS"],
                      theta_columns_naming='title',
                      class_ids={f'@{lang}': 1 for lang in experiment_config["LANGUAGES_ALL"]},
                      show_progress_bars=True,
                      dictionary=dictionary)

    model.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore',
                                             topic_names=subject_topic_list))
    for lang in model.class_ids:
        model.scores.add(artm.SparsityPhiScore(name=f'SparsityPhiScore_{lang}',
                                               class_id=lang,
                                               topic_names=subject_topic_list))
        model.scores.add(artm.PerplexityScore(name=f'PerlexityScore_{lang}',
                                              class_ids=lang,
                                              dictionary=dictionary))

    # SmoothTheta
    model.regularizers.add(
        artm.SmoothSparseThetaRegularizer(
            name='SmoothThetaRegularizer',
            tau=artm_model_params["tau_SmoothTheta"],
            topic_names=background_topic_list)
    )
    rel_toolbox_lite.handle_regularizer(
        use_relative_coefficients=True,
        model=model,
        regularizer=model.regularizers['SmoothThetaRegularizer'],
        data_stats=rel_toolbox_lite.count_vocab_size(
            dictionary=dictionary,
            modalities={f'@{lang}': 1 for lang in experiment_config["LANGUAGES_ALL"]})
    )

    # SparseTheta
    model.regularizers.add(
        artm.SmoothSparseThetaRegularizer(
            name='SparseThetaRegularizer',
            tau=artm_model_params["tau_SparseTheta"],
            topic_names=subject_topic_list)
    )
    rel_toolbox_lite.handle_regularizer(
        use_relative_coefficients=True,
        model=model,
        regularizer=model.regularizers['SparseThetaRegularizer'],
        data_stats=rel_toolbox_lite.count_vocab_size(
            dictionary=dictionary,
            modalities={f'@{lang}': 1 for lang in experiment_config["LANGUAGES_ALL"]})
    )

    # DecorrelatorPhi
    model.regularizers.add(
        artm.DecorrelatorPhiRegularizer(
            name='DecorrelatorPhiRegularizer',
            tau=artm_model_params["tau_DecorrelatorPhi"],
            gamma=0, topic_names=subject_topic_list)
    )
    rel_toolbox_lite.handle_regularizer(
        use_relative_coefficients=True,
        model=model, regularizer=model.regularizers['DecorrelatorPhiRegularizer'],
        data_stats=rel_toolbox_lite.count_vocab_size(
            dictionary=dictionary,
            modalities={f'@{lang}': 1 for lang in experiment_config["LANGUAGES_ALL"]})
    )
    return model


def _get_balanced_doc_ids(
        train_dict: typing.Dict[str, str],
        train_grnti: typing.Dict[str, str],
        docs_of_rubrics: typing.Dict[str, list]
) -> typing.Tuple[list, typing.Dict[str, str]]:
    """
    Create train data balanced by rubrics.

    Returns balanced_doc_ids - list of document ids, balanced by rubric. Documents of
    all rubrics occurs in balanced_doc_ids the same number of times,
    equal to average_rubric_size.

    Returns train_dict - dict where key - document id, value - document in
    Vowpal Wabbit format. Function change train_dict, multiplying token counters
    by number of occurrences of document id in balanced_doc_ids.

    Returns
    -------
    balanced_doc_ids: list
        list of document ids, balanced by rubric
    train_dict: dict
        dict where key - document id, value - document in vowpal wabbit format
    """
    average_rubric_size = int(len(train_grnti) / len(set(train_grnti.values())))
    balanced_doc_ids = []
    for rubric in set(train_grnti.values()):
        doc_ids_rubric = np.random.choice(docs_of_rubrics[rubric], average_rubric_size)
        balanced_doc_ids.extend(doc_ids_rubric)

        doc_ids_count = Counter(doc_ids_rubric)
        for doc_id, count in doc_ids_count.items():
            if count > 1:
                new_line_dict = dict()
                for line_lang in train_dict[doc_id].split(' |@')[1:]:
                    lang = line_lang.split()[0]
                    line_lang_dict = {
                        token_with_count.split(':')[0]: count *
                        int(token_with_count.split(':')[1])
                        for token_with_count in line_lang.split()[1:]
                    }
                    new_line_lang = ' '.join([lang] +
                                             [':'.join([token, str(count)])
                                              for token, count in line_lang_dict.items()])
                    new_line_dict[lang] = new_line_lang
                new_line = ' |@'.join([doc_id] + list(new_line_dict.values()))
                train_dict[doc_id] = new_line
    return balanced_doc_ids, train_dict


def _get_balanced_doc_ids_with_augmentation(
        train_dict: typing.Dict[str, str],
        train_grnti: typing.Dict[str, str],
        docs_of_rubrics: typing.Dict[str, list],
        experiment_config
) -> typing.Tuple[list, typing.Dict[str, str]]:
    """
    Create train data balanced by rubrics with augmentation.

    If the rubric size is larger than the average rubric size, a sample is taken
    equal to the average rubric size.
    If the size of the heading is less than the average rubric size,
    all possible documents of rubric are taken; artificial documents are also generated by
    combining the two documents in a ratio of 1 to experiment_config.aug_proportion.

    Returns
    -------
    balanced_doc_ids: list
        list of document ids, balanced by rubric
    train_dict: dict
        dict where key - document id, value - document in vowpal wabbit format
    """
    average_rubric_size = int(len(train_grnti) / len(set(train_grnti.values())))
    balanced_doc_ids = []
    for rubric in set(train_grnti.values()):
        if len(docs_of_rubrics[rubric]) >= average_rubric_size:
            doc_ids_rubric = np.random.choice(docs_of_rubrics[rubric], average_rubric_size)
            balanced_doc_ids.extend(doc_ids_rubric)
        else:
            # все возможные уникальные пары айди
            doc_id_pair_list = list(itertools.combinations(docs_of_rubrics[rubric], 2))
            doc_id_pair_list_indexes = list(
                np.random.choice(len(doc_id_pair_list),
                                 average_rubric_size - len(docs_of_rubrics[rubric]))
            )
            doc_id_pair_list = [doc_id_pair_list[i] for i in doc_id_pair_list_indexes]
            doc_id_unique_list = []

            # для каждой пары - новый уникальный айди,
            # новая статья как сумма старых и запись в train_dict
            for doc_id_pair in doc_id_pair_list:
                doc_id_unique = '_'.join([doc_id_pair[0], doc_id_pair[1]])
                doc_id_unique_list.append(doc_id_unique)
                line_1 = train_dict[doc_id_pair[0]]
                line_2 = train_dict[doc_id_pair[1]]
                line_unique_dict = dict()
                for line_lang in line_1.split(' |@')[1:-1]:
                    lang = line_lang.split()[0]
                    line_lang_dict = {
                        token_and_count.split(':')[0]: token_and_count.split(':')[1]
                        for token_and_count in line_lang.split()[1:]
                    }
                    new_line = ' '.join([lang] + [':'.join([token, count])
                                                  for token, count in line_lang_dict.items()])
                    line_unique_dict[lang] = new_line
                for line_lang in line_2.split(' |@')[1:-1]:
                    lang = line_lang.split()[0]
                    if lang not in line_unique_dict:
                        line_lang_dict = {
                            token_and_count.split(':')[0]: token_and_count.split(':')[1]
                            for token_and_count in line_lang.split()[1:]
                        }
                        new_line = ' '.join([lang] + [':'.join([token, count])
                                                      for token, count in line_lang_dict.items()])
                        line_unique_dict[lang] = new_line
                    else:
                        line_lang_dict = {token_and_count.split(':')[0]: str(
                            int(experiment_config["aug_proportion"] *
                                int(token_and_count.split(':')[1])))
                                          for token_and_count in line_lang.split()[1:]}
                        new_line = ' '.join([':'.join([token, count])
                                             for token, count in line_lang_dict.items()])
                        line_unique_dict[lang] += ' ' + ' '.join(new_line.split())
                grnti_rubric = line_1.split(' |@')[-1].split()[1].split(':')[0]
                line_unique_dict['GRNTI'] = 'GRNTI ' + f'{grnti_rubric}:10'
                line_unique = ' |@'.join([doc_id_unique] + list(line_unique_dict.values()))
                train_dict[doc_id_unique] = line_unique
            doc_ids_rubric = docs_of_rubrics[rubric] + list(np.random.choice(
                doc_id_unique_list, average_rubric_size - len(docs_of_rubrics[rubric])))
            balanced_doc_ids.extend(doc_ids_rubric)
    return balanced_doc_ids, train_dict


def _get_rubric_of_train_docs(experiment_config) -> dict:
    """
    Get dict where keys - document ids, value - number of rubric of document.

    Do not contents rubric 'нет'.

    Returns
    -------
    train_grnti: dict
        dict where keys - document ids, value - number of GRNTI rubric of document.
    """
    # TODO: надо ли открывать файлы на каждой итерации или можно подавать открытые?
    with open(experiment_config["path_articles_rubrics_train_grnti"]) as file:
        articles_grnti_with_no = json.load(file)
    with open(experiment_config["path_elib_train_rubrics_grnti"]) as file:
        elib_grnti_to_fix_with_no = json.load(file)
    with open(experiment_config["path_grnti_mapping"]) as file:
        grnti_to_number = json.load(file)

    articles_grnti = {doc_id: rubric
                      for doc_id, rubric in articles_grnti_with_no.items()
                      if rubric != 'нет'}

    elib_grnti = {doc_id[:-len('.txt')]: rubric
                  for doc_id, rubric in elib_grnti_to_fix_with_no.items()
                  if rubric != 'нет'}

    train_grnti = dict()
    for doc_id in articles_grnti:
        rubric = str(grnti_to_number[articles_grnti[doc_id]])
        train_grnti[doc_id] = rubric
    for doc_id in elib_grnti:
        rubric = str(grnti_to_number[elib_grnti[doc_id]])
        train_grnti[doc_id] = rubric
    return train_grnti


def _get_modality_distribution(modality_list, path_train):
    """
    Get document number foe each of modality.


    :modality_list path_train: typing.List[str]
        list with title of modalities to calculate distribution
    :param path_train: pathlib.Path
        path to txt file with non-wiki part of train data.
    :return: typing.Dict[str, int]
        dict, key - modality, value - amount of documents of this modality
    """
    # TODO: add wiki part of train data
    with open(path_train) as file:
        train_data = file.read()

    modality_distribution = {
        mod: train_data.count('|@{mod}')
        for mod in modality_list
    }

    return modality_distribution



def _recursively_unlink(path: Path):
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _recursively_unlink(child)
    path.rmdir()


def _train_iteration(
        model, experiment_config, train_grnti, docs_of_rubrics,
        path_balanced_train, path_to_batches, path_batches_wiki=None
) -> typing.Dict[str, float]:
    train_dict = joblib.load(experiment_config["train_dict_path"])

    # генерирую сбалансированные данные
    if experiment_config["need_augmentation"]:
        balanced_doc_ids, train_dict = _get_balanced_doc_ids_with_augmentation(
            train_dict, train_grnti, docs_of_rubrics, experiment_config
        )
    else:
        balanced_doc_ids, train_dict = _get_balanced_doc_ids(
            train_dict, train_grnti, docs_of_rubrics
        )

    with open(path_balanced_train, 'w') as file:
        file.writelines([train_dict[doc_id].strip() + '\n'
                         for doc_id in balanced_doc_ids])
    del train_dict

    # строю батчи по сбалансированным данным
    batches_list = list(path_to_batches.iterdir())
    if batches_list:
        for batch in batches_list:
            if batch.is_file():
                batch.unlink()
            else:
                _recursively_unlink(batch)
    _ = artm.BatchVectorizer(
        data_path=str(path_balanced_train),
        data_format="vowpal_wabbit",
        target_folder=str(path_to_batches),
    )
    if path_batches_wiki:
        batch_vectorizer = artm.BatchVectorizer(
            data_path=[path_to_batches, path_batches_wiki],
            data_weight=[1, 1]
        )
    else:
        batch_vectorizer = artm.BatchVectorizer(
            data_path=path_to_batches
        )
    model.fit_offline(batch_vectorizer, num_collection_passes=1)


def fit_topic_model(experiment_config):
    """
    The function fits topic model according to the experiment_config file.

    Parameters
    ----------
    experiment_config - yaml config with parameters of a model

    Returns
    -------
    """
    path_experiment = Path(experiment_config["path_experiment"])
    path_experiment.mkdir(parents=True, exist_ok=True)
    path_to_dump_model = path_experiment.joinpath('topic_model')
    path_train_data = path_experiment.joinpath('train_data')
    path_to_batches = path_train_data.joinpath('batches_balanced')
    path_to_batches.mkdir(parents=True, exist_ok=True)
    path_balanced_train = path_train_data.joinpath('train_balanced.txt')

    train_grnti = _get_rubric_of_train_docs(experiment_config)
    train_dict = joblib.load(experiment_config["train_dict_path"])

    docs_of_rubrics = {rubric: [] for rubric in set(train_grnti.values())}
    for doc_id, rubric in train_grnti.items():
        if doc_id in train_dict:
            docs_of_rubrics[rubric].append(doc_id)
    del train_dict

    model = _create_init_model(experiment_config)
    path_batches_wiki = experiment_config.get("path_wiki_train_batches", None)
    num_collection_passes = experiment_config["artm_model_params"]["num_collection_passes"]
    average_rubric_size = int(len(train_grnti) / len(set(train_grnti.values())))
    print(f'На каждой эпохе используется по {average_rubric_size} документа ' +
          f'для каждой из {experiment_config["num_rubric"]} рубрик.')
    # тут нужно визуализировать "распределение" документов по модальностям (рубрикам)

    for iteration in tqdm(range(num_collection_passes)):
        _train_iteration(model, experiment_config, train_grnti, docs_of_rubrics,
                         path_balanced_train, path_to_batches, path_batches_wiki)
        if path_to_dump_model.exists():
            _recursively_unlink(path_to_dump_model)
        model.dump_artm_model(str(path_to_dump_model))
        search_metrics = calculate_search_quality(experiment_config)
        # тут нужно визуализировать итерацию iteration
        # тут нужно визуализировать метрики search_metrics
        modality_list = experiment_config["LANGUAGES_MAIN"]
        modality_distribution = _get_modality_distribution(modality_list, path_balanced_train)
        pprint(modality_distribution)
        # тут нужно визуализировать распределение документов по модальностям modality_distribution


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = _get_args()
    for config in load(args.config, always_iter=True):
        fit_topic_model(config)
