#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
: "${HF_HOME:?Environment variable HF_HOME not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=6

# List of tasks to evaluate
TASKS=${TASKS:-"bace bbbp pampa clintox hia pgp ames dili herg"}

# List of prompt types to use (all 13 prompts)
PROMPT_TYPES=${PROMPT_TYPES:-"default_variant_1 default_variant_2 default_variant_3 rationale_variant_1 rationale_variant_2 task_info_variant_1 task_info_variant_2 binary_instruction confidence_instruction checklist_instruction"}

for task_name in ${TASKS}; do
    echo "=================================================="
    echo "Processing task: ${task_name}"
    echo "=================================================="
    
    for prompt_type in ${PROMPT_TYPES}; do
        echo "Running ${task_name} with prompt type: ${prompt_type}"
        
        python ${BASE_DIR}/evaluation/inference.py \
            --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
            --tokenizer_path DongkiKim/Mol-Llama-3.1-8B-Instruct \
            --data_dir ${DATA_DIR} \
            --task_name ${task_name} \
            --qformer_path ${HF_HOME}/hub/models--DongkiKim--Mol-Llama-3.1-8B-Instruct/snapshots/dd99e6ea328e01d713ac31aaad017074a6126483/model.safetensors \
            --prompt_type ${prompt_type} \
            --output_name molllama \
            --baseline_type mollama \
            --freeze_llm \
            --enable_blending
    done
    
    echo "Completed task: ${task_name}"
    echo ""
done

echo "All tasks completed!"