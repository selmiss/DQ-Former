#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=1

python ${BASE_DIR}/evaluation/inference.py \
    --pretrained_model_name_or_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
    --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
    --data_dir ${DATA_DIR} \
    --task_name pampa \
    --prompt_type default