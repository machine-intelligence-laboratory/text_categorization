export MODELS_PATH=/home/antiplagiat/models
export DATA_PATH=/home/antiplagiat/data
export BPE_PATH=/home/antiplagiat/bpe_models

docker-compose -f docker-compose-train.yml build && docker-compose -f docker-compose-train.yml up
