#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

python ${BASE_DIR}/stage2.py \
    --train_config_path ${BASE_DIR}/configs/stage2_dqw2d/train_config.yaml \
    --data_config_path ${BASE_DIR}/configs/stage2_dqw2d/data_config.yaml