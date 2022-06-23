export CONFIG=./ap/utils/experiment_config.yml
export DATA_PATH=/data/datasets/Antiplagiat/

docker-compose -f docker-compose-train.yml build && docker-compose -f docker-compose-train.yml up
