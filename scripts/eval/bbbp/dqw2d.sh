#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
        --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
        --data_dir ${DATA_DIR} \
        --task_name bbbp \
        --qformer_path ${BASE_DIR}/checkpoints/stage2_dqw2d/epoch=01.ckpt \
        --prompt_type ${prompt_type} \
        --output_name dqw2d \
        --use_dq_encoder \
        --enable_blending
done