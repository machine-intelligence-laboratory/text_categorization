FROM python:3.8.5-slim

COPY ./data ./data
COPY ./best_model_avr_rubric_size_25/20220526_211543 /model

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY ./protos ./protos
COPY ./generate.sh ./generate.sh

COPY ap ./ap
RUN /bin/sh -c /generate.sh


# ENTRYPOINT python -m ap.inference.server --model=/model --bpe=/bpe --rubric=/rubric
# ENTRYPOINT python -m ap.train.server --config=/config --data=/data