import json
import logging

import requests
from aiohttp import web
from prometheus_client import start_http_server, Counter

logging.basicConfig(level=logging.DEBUG)

METRICS = dict()


def send_metric(key, action, value):
    requests.post('http://localhost:8080/set', json={'key': key, 'action': action, 'value': value})


def set_metric(key, value):
    send_metric(key, 'set', value)


def inc_metric(key, value):
    send_metric(key, 'inc', value)


async def handle(request):
    data = await request.json()
    logging.debug(json.dumps(data))
    getattr(METRICS[data['key']], data['action'])(data['value'])
    return web.Response()


def run_metrics_server():
    from prometheus_client import Gauge

    start_http_server(8000, addr='0.0.0.0')
    METRICS.update({
        'added_docs': Counter('added_docs', 'Number of added documents'),
        'training_iteration': Gauge('training_iteration', 'Current training iteration'),
        'average_rubric_size': Gauge('average_rubric_size', 'Average rubric size'),
        'num_rubric': Gauge('num_rubric', 'Number of rubrics')
    })

    app = web.Application()
    app.add_routes([web.post('/set', handle)])
    web.run_app(app)
