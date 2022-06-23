export CONFIG=/home/antiplagiat/config
export DATA_PATH=/home/antiplagiat/data

docker-compose -f docker-compose-train.yml build && docker-compose -f docker-compose-train.yml up
