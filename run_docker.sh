#!/usr/bin/env bash

CURR_DIR=$(pwd)

docker run --detach --rm \
  --name deeplab_v3 \
  -p 8500:8500 -p 8501:8501 \
  -v "${CURR_DIR}/versions:/models/deeplab_v3" \
  -e MODEL_NAME=deeplab_v3 \
  -e TF_CPP_MIN_VLOG_LEVEL=1 \
  tensorflow/serving:latest

