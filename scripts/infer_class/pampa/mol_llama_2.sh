#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=7

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/llama-2-7b-chat \
        --tokenizer_path DongkiKim/Mol-Llama-2-7b-chat \
        --data_dir ${DATA_DIR} \
        --task_name pampa \
        --qformer_path hf_home/hub/models--DongkiKim--Mol-Llama-2-7b-chat/snapshots/30631d7bf0de1409bc48dc9a5baa833c851ab76a/model.safetensors \
        --prompt_type ${prompt_type} \
        --output_name molllama_2 \
        --baseline_type mollama \
        --freeze_llm \
        --enable_blending
done