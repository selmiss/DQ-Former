#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

python ${BASE_DIR}/evaluation/molecule_gen.py \
  --train_config_path ${BASE_DIR}/configs/molecule_property/dqw2d/train_config.yaml \
  --data_config_path ${BASE_DIR}/configs/molecule_property/dqw2d/data_config.yaml

