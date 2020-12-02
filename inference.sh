export MODEL_PATH=/home/antiplagiat/models/PLSA_L14_V20000_TOP100_E1_G0_T10
export BPE_PATH=/home/antiplagiat/bpe_models

docker-compose -f docker-compose.yml build && docker-compose -f docker-compose.yml up
