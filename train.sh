export CONFIG=./ap/utils/experiment_docker_config.yml
export DATA_PATH=./data

docker-compose -f docker-compose-train.yml build && docker-compose -f docker-compose-train.yml up
