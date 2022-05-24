import itertools
import json
import logging
import typing

import requests
from aiohttp import web
from prometheus_client import start_http_server, Counter

logging.basicConfig(level=logging.DEBUG)

METRICS = {}


def send_metric(key, action, value):
    """
    Устанавливает значение метрики через REST сервис

    Args:
        key: ключ метрики
        action: действие inc или set
        value: значение

    Returns:

    """
    requests.post('http://localhost:8080/set', json={'key': key, 'action': action, 'value': value})


def set_metric(key, value):
    """
    Устанавливает значение метрики
    Args:
        key: Ключ метрики
        value:  Значение
    """
    send_metric(key, 'set', value)


def inc_metric(key, value):
    """
    Инкрементирует значение метрики

    Args:
        key: ключ метрики
        value: значение
    """
    send_metric(key, 'inc', value)


async def handle(request):
    """
    REST обработчик

    Args:
        request: REST запрос

    Returns:
        REST ответ
    """
    data = await request.json()
    logging.debug(json.dumps(data))
    metric = METRICS.get(data['key'])
    return web.Response()


def run_metrics_server(config: typing.Dict):
    """
    Запускает сервер метрик

    Args:
        config: конфиг эксперимента
    """
    try:
        from prometheus_client import Gauge

        logging.info('Starting metrics server')

        start_http_server(8000, addr='0.0.0.0')
        METRICS.update({
            'added_docs': Counter('added_docs', 'Number of added documents'),
            'training_iteration': Gauge('training_iteration', 'Current training iteration'),
            'average_rubric_size': Gauge('average_rubric_size', 'Average rubric size'),
            'num_rubric': Gauge('num_rubric', 'Number of rubrics'),
            'train_size_bytes': Gauge('train_size_bytes', 'Size of training data in bytes'),
            'train_size_docs': Gauge('train_size_docs', 'Size of training data in documents'),
            'perplexity_score_en': Gauge('perplexity_score_en', 'Perplexity score english'),
            'perplexity_score_ru': Gauge('perplexity_score_ru', 'Perplexity score russian'),
            'num_topics': Gauge('num_topics', 'Number of topics'),
            'num_bcg_topics': Gauge('num_bcg_topics', 'Number of background topics'),
            'num_modalities': Gauge('num_modalities', 'Number of background topics'),
            'num_collection_passes': Gauge('num_collection_passes', 'Number of background topics'),
            'tau_DecorrelatorPhi': Gauge('tau_DecorrelatorPhi', 'tau DecorrelatorPhi'),
            'tau_SmoothTheta': Gauge('tau_SmoothTheta', 'tau SmoothTheta'),
            'tau_SparseTheta': Gauge('tau_SparseTheta', 'tau SparseTheta'),
        })

        for modality in itertools.chain(config["MODALITIES_TRAIN"].keys(), config["LANGUAGES_TRAIN"].keys()):
            METRICS[f'modality_distribution_{modality}'] = Gauge(f'modality_distribution_{modality}',
                                                                 f'Modality distribution {modality}')

        app = web.Application()
        app.add_routes([web.post('/set', handle)])

        web.run_app(app, host='0.0.0.0', port=8080)
    except Exception as ex:
        logging.exception(ex)
