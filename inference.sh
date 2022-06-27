export MODEL_PATH=./best_model_avr_rubric_size_25/20220526_211543
export BPE_PATH=./data/BPE_models
export RUBRIC_PATH=./data

docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up
