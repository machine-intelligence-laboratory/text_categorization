FROM python:3.8.5-slim


COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY ./protos /protos
COPY ./generate.sh /generate.sh

COPY ap ./ap
RUN /bin/sh -c /generate.sh
