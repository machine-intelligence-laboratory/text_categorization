import artm
import numpy as np
import pandas as pd
import typing as tp

from pathlib import Path
from scipy.spatial.distance import cosine


def _get_text_dist(vw_texts, phi):
    text_dists = []

    for vw_line in vw_texts:
        text_dist = pd.DataFrame(index=phi.index.values, dtype=float)

        vw_line_lang_list = vw_line.strip().split(' |@')
        num, *texts = vw_line_lang_list

        for vw_line_lang in texts:
            lang, *tokens_with_counter = vw_line_lang.split()
            if lang == 'ru':
                text = pd.DataFrame.from_dict(_get_text_tokens(tokens_with_counter), orient='index')[0]

                text_dist['dist'] = (text / text.sum(axis=0))
                text_dist = text_dist.fillna(0)

                text_dists.append(text_dist)

    return text_dists


def _get_text_tokens(tokens_with_counter: tp.List[str]):
    res = {}
    for token_with_counter in tokens_with_counter:
        token, counter = token_with_counter.split(':')
        res[token] = float(counter)

    return res


def _get_topics(vw_texts, theta, phi, n, tmp_file):
    cosines = []
    text_dists = _get_text_dist(vw_texts, phi)

    topics = {}

    for i, text_dist in enumerate(text_dists):
        text = theta.columns[i]
        topic_from = f'topic_{theta[text].argmax()}'

        topic_to = None
        max_cos = -1

        for topic in theta.index.values:
            if topic != topic_from:
                cos = 1 - cosine(phi[topic] - phi[topic_from], text_dist)
                if cos > max_cos:
                    topic_to = topic
                    max_cos = cos

        cosines.append(max_cos)

        topics[text] = [topic_from, topic_to]

    cosines = np.array(cosines)
    max_cosines_ind = np.argpartition(cosines, -n)[-n:]
    texts = theta.columns[max_cosines_ind]

    res = {}
    for text in texts:
        res[text] = topics[text]

    with open(tmp_file, 'w') as file:
        for vw_line in vw_texts:
            vw_line_lang_list = vw_line.strip().split(' |@')
            title, *_ = vw_line_lang_list
            if title in res:
                file.write(vw_line)

    return res


def _get_important_tokens(text_dist, num_top_tokens=5):
    new_diff = text_dist[text_dist.index.str.len() > 4]['new_diff']
    added = new_diff[new_diff > 0].nlargest(num_top_tokens).index.values
    removed = new_diff[new_diff < 0].nsmallest(num_top_tokens).index.values

    return added, removed


def _mutate_text(vw_texts, phi, topics, need_change, multiplier, num_top_tokens=5,
                 tmp_file='data/change_topic/tmp.txt'):
    changed = {}

    with open(tmp_file, 'w') as file:
        text_dist = pd.DataFrame(index=phi.index.values, dtype=float)

        for vw_line in vw_texts:
            vw_line_lang_list = vw_line.strip().split(' |@')
            title, *texts = vw_line_lang_list

            if need_change[title]:
                topic_from, topic_to = topics[title]
                string_to_write = title
                for vw_line_lang in texts:
                    lang, *tokens_with_counter = vw_line_lang.split()

                    if lang == 'ru':
                        new_tokens = []

                        text = pd.DataFrame.from_dict(_get_text_tokens(tokens_with_counter), orient='index')[0]

                        token_count = text.sum()
                        text_dist['dist'] = (text / token_count)
                        text_dist = text_dist.fillna(0)

                        text_dist['diff'] = (phi[topic_from] - phi[topic_to]) * multiplier

                        text_dist['new'] = np.maximum(text_dist['dist'] - text_dist['diff'], 0)
                        text_dist['new'] /= text_dist['new'].sum()

                        text_dist['new_diff'] = text_dist['new'] - text_dist['dist']

                        added, removed = _get_important_tokens(text_dist, num_top_tokens)
                        changed[title] = [added, removed]

                        text_dist['new'] *= token_count

                        for token, value in text_dist['new'].astype(np.float16).items():
                            if value > 0:
                                new_tokens.append(f'{token}:{value}')

                        text = ' '.join(new_tokens)
                        string_to_write += ' |@' + lang + ' ' + text + '\n'
                        file.write(string_to_write)
            else:
                file.write(vw_line)

    return changed


def _check_change(model, topics, need_change, changed, tmp_file, log_file='data/change_topic/log.txt'):
    batch_vectorizer = artm.BatchVectorizer(data_path=tmp_file, data_format='vowpal_wabbit',
                                            target_folder='data/batches_tmp2', batch_size=20)
    theta = model.transform(batch_vectorizer)
    texts = theta.columns
    with open(log_file, 'a') as log:
        for text in texts:
            if need_change[text]:
                top_topic = f'topic_{theta[text].argmax()}'
                topic_from, topic_to = topics[text]
                if top_topic == topic_to:
                    need_change[text] = False
                    added, removed = changed[text]
                    log.write(
                        f'{text}: changed from {topic_from} to {topic_to}\n' +
                        f'Added: {", ".join(added)}\n' +
                        f'Removed: {", ".join(removed)}\n\n')

    return need_change, not True in need_change.values()


def augment_text(model, input_text: str, target_folder: str, n: int, num_top_tokens: int = 5):
    """
    Визуализация предсказаний обученной модели.

        Args:
            model (artm.ARTM): путь до обученной модели
            input_text (str): путь до входного текста для визуализации предсказания модели
                на ru языке в формате vowpal wabbit
            target_folder (str): путь для временного хранения даннных
            n (int): количество текстов для анализа
            num_top_tokens (int): максимальное число добавленных и удаленных топ-токенов
    """

    with open(input_text) as file:
        vw_texts = file.readlines()
    # TODO: заменить n на len(vw_texts) ?

    target_folder = Path(target_folder)
    target_folder.mkdir(exist_ok=True)
    tmp_batches = target_folder.joinpath('batches')
    tmp_batches.mkdir(exist_ok=True)
    batch_vectorizer = artm.BatchVectorizer(data_path=input_text, data_format='vowpal_wabbit',
                                            target_folder=str(tmp_batches), batch_size=20)
    change_topic = target_folder.joinpath('change_topic')
    change_topic.mkdir(exist_ok=True)
    tmp_file = change_topic.joinpath('tmp.txt')

    # model = artm.load_artm_model(model_path)

    theta = model.transform(batch_vectorizer)
    phi = model.get_phi(class_ids="@ru")

    topics = _get_topics(vw_texts, theta, phi, n, tmp_file)

    with open(tmp_file) as file:
        vw_texts = file.readlines()

    need_change = {title: True for title in topics}

    for multiplier in np.logspace(-1.5, 0, 5):
        changed = _mutate_text(vw_texts, phi, topics, need_change, multiplier, num_top_tokens, tmp_file)
        need_change, stop = _check_change(model, topics, need_change, changed, tmp_file)
        if stop:
            break
