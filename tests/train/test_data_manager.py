import yaml

from concurrent import futures

from ap.train.data_manager import ModelDataManager
from ap.train.metrics import run_metrics_server


def test_init():
    data_dir = 'tests/data/'
    train_conf = 'tests/data/test_config.yml'
    with open(train_conf, "r") as file:
        config = yaml.safe_load(file)

    executor = futures.ProcessPoolExecutor(max_workers=3)
    executor.submit(run_metrics_server, config)
    data_manager = ModelDataManager(data_dir, train_conf)

    assert isinstance(data_manager, ModelDataManager)


def test_get_modality_distribution():
    data_dir = 'tests/data/'
    train_conf = 'tests/data/test_config.yml'
    with open(train_conf, "r") as file:
        config = yaml.safe_load(file)

    executor = futures.ProcessPoolExecutor(max_workers=3)
    executor.submit(run_metrics_server, config)
    data_manager = ModelDataManager(data_dir, train_conf)
    modality_distribution = data_manager.get_modality_distribution()

    with open('tests/data/train.txt') as file:
        train_data = file.read()

    assert modality_distribution['UDK'] == train_data.count('|@UDK')
    assert modality_distribution['GRNTI'] == train_data.count('|@GRNTI')
    assert modality_distribution['ru'] == train_data.count('|@ru')
    assert modality_distribution['zh'] == train_data.count('|@zh')
