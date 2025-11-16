#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
        --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
        --data_dir ${DATA_DIR} \
        --task_name bbbp \
        --qformer_path /home/UWO/zjing29/proj/DQ-Former/hf_home/hub/models--DongkiKim--Mol-Llama-3.1-8B-Instruct/snapshots/dd99e6ea328e01d713ac31aaad017074a6126483/model.safetensors \
        --prompt_type ${prompt_type} \
        --output_name molllama \
        --baseline_type mollama \
        --freeze_llm \
        --enable_blending
done