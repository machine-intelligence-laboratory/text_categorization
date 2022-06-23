export MODEL_PATH=./best_model_avr_rubric_size_25/20220526_211543

docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up
