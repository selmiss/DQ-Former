#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=7

for prompt_type in default rationale task_info; do

    python ${BASE_DIR}/evaluation/inference.py \
        --pretrained_model_name_or_path unsloth/llama-2-7b-chat \
        --tokenizer_path unsloth/llama-2-7b-chat \
        --data_dir ${DATA_DIR} \
        --task_name pampa \
        --qformer_path none \
        --prompt_type ${prompt_type} \
        --output_name llama2 \
        --only_llm
done