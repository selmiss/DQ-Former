#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/llama-2-7b-chat \
        --tokenizer_path unsloth/llama-2-7b-chat \
        --data_dir ${DATA_DIR} \
        --task_name pampa \
        --lora_path zjunlp/llama2-molinst-molecule-7b \
        --prompt_type ${prompt_type} \
        --output_name molinstructions_2 \
        --baseline_type llm_lora \
        --only_llm
done