#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
        --tokenizer_path ${BASE_DIR}/checkpoints/edt_former_s2_large/final_model \
        --data_dir ${DATA_DIR} \
        --task_name bace \
        --qformer_path ${BASE_DIR}/checkpoints/edt_former_s2_large/final_model/model.safetensors \
        --prompt_type ${prompt_type} \
        --output_name edt_former \
        --use_dq_encoder \
        --freeze_llm \
        --enable_blending
done