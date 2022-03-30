#!/usr/bin/env bash
if [ -d ap/topic_model ]; then rm -Rf ap/topic_model; fi

python -m grpc_tools.protoc -I ./protos --python_out=./ --grpc_python_out=./ ./protos/ap/topic_model/v1/*.proto
find ap/topic_model -type d -exec touch {}/__init__.py \;
