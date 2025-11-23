#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"


export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

python ${BASE_DIR}/runner/entropy_model/train.py