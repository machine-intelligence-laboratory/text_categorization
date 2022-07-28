import yaml

from datetime import datetime
from pathlib import Path

import artm

from ap.topic_model.v1.TopicModelTrain_pb2 import StartTrainTopicModelRequest
from ap.train.data_manager import ModelDataManager
from ap.train.trainer import ModelTrainer
from ap.utils.general import recursively_unlink

from ap.utils.bpe import load_bpe_models
from ap.utils.vowpal_wabbit_bpe import VowpalWabbitBPE


def get_trainer():
    data_dir = 'tests/data'
    config_path = 'tests/data/test_config.yml'
    data_manager = ModelDataManager(data_dir, config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    bpe_models = load_bpe_models(config["BPE_models"])
    vw = VowpalWabbitBPE(bpe_models)
    docs = {'0_0': {'en': 'a:1 the:1 one:2', 'GRNTI': '64:1'},
            '0_1': {'en': 'two:1 none:1 real:2', 'GRNTI': '1:1'},
            '1_0': {'en': 'three:1 four:1 for:2', 'GRNTI': '1:1'}}
    with open('tests/data/train.txt', 'w') as file:
        file.writelines([doc_id + ' ' + ' '.join([f'|@{mod}' + ' ' + line for mod, line in vw_dict.items()]) + '\n'
                         for doc_id, vw_dict in docs.items()])
    data_manager.write_new_docs(vw, docs)
    trainer = ModelTrainer(data_manager)
    return trainer


def test_create_initial_model():
    trainer = get_trainer()
    trainer.model = trainer._create_initial_model()
    assert isinstance(trainer.model, artm.ARTM)


def test_load_model():
    trainer = get_trainer()
    train_type = StartTrainTopicModelRequest.TrainType.FULL
    trainer._load_model(train_type)
    assert isinstance(trainer.model, artm.ARTM)


def test_train_model():
    trainer = get_trainer()
    train_type = StartTrainTopicModelRequest.TrainType.FULL
    trainer._load_model(train_type)
    trainer.train_model(train_type)

    config_path = 'tests/data/test_config.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    assert Path(config['path_experiment']).exists
    recursively_unlink(Path(config['path_experiment']))
    with open('tests/data/train.txt', 'w') as file:
        pass


def test_model_scores_value():
    trainer = get_trainer()
    train_type = StartTrainTopicModelRequest.TrainType.FULL
    trainer._load_model(train_type)
    scores = trainer.model_scores_value
    assert scores == {}

    trainer.train_model(train_type)
    scores = trainer.model_scores_value
    assert 'SparsityThetaScore' in scores


def test_generate_model_name():
    trainer = get_trainer()
    model_name = trainer.generate_model_name()
    date_formatter = "%Y%m%d_%H%M%S"
    model_creation_time_info = datetime.strptime(model_name, date_formatter)
    now = datetime.now()
    assert model_creation_time_info.year == now.year
    assert model_creation_time_info.day == now.day
