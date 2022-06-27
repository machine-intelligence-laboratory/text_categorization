export MODEL_PATH=/model
export BPE_PATH=./data/BPE_models
export RUBRIC_PATH=./data

docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up
