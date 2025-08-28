#! /bin/bash

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"


export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=3

python ${BASE_DIR}/trainer/entropy_model/entropy_process.py \
  --in_json ${DATA_DIR}/zeroshot/pampa/data_brics.jsonl \
  --out_json ${DATA_DIR}/zeroshot/pampa/data_brics_entropy.jsonl \
  --ckpt_dir ${BASE_DIR}/checkpoints/entropy_model/checkpoint-4929 \
  --vocab ${BASE_DIR}/trainer/entropy_model/vocab.txt \
  --batch_size 256 --q 0.75
