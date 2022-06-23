FROM python:3.8.5-slim


COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY ./protos ./protos
COPY ./generate.sh ./generate.sh
RUN /bin/sh -c bash ./generate.sh

COPY ap ./ap


ENTRYPOINT python -m ap.inference.server --model=/model
