#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6,7

python ${BASE_DIR}/stage1.py \
    --train_config_path ${BASE_DIR}/configs/stage1_dqformer_combine/train_config.yaml \
    --data_config_path ${BASE_DIR}/configs/stage1_dqformer_combine/data_config.yaml \
    --test_mode