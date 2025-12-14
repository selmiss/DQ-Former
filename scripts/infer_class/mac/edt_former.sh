#! /bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=7

# List of tasks to evaluate
TASKS=${TASKS:-"mac"}

# List of prompt types to use (all 13 prompts)
PROMPT_TYPES=${PROMPT_TYPES:-"default rationale task_info default_variant_1 default_variant_2 default_variant_3 rationale_variant_1 rationale_variant_2 task_info_variant_1 task_info_variant_2 binary_instruction confidence_instruction checklist_instruction"}

for task_name in ${TASKS}; do
    echo "=================================================="
    echo "Processing task: ${task_name}"
    echo "=================================================="
    
    for prompt_type in ${PROMPT_TYPES}; do
        echo "Running ${task_name} with prompt type: ${prompt_type}"
        python ${BASE_DIR}/evaluation/inference.py \
            --pretrained_model_name_or_path unsloth/Llama-3.1-8B-Instruct \
            --tokenizer_path ${BASE_DIR}/checkpoints/edt_former_s2_large/final_model \
            --data_dir ${DATA_DIR} \
            --task_name ${task_name} \
            --qformer_path ${BASE_DIR}/checkpoints/edt_former_s2_large/final_model/model.safetensors \
            --prompt_type ${prompt_type} \
            --output_name edt_former_s2_large \
            --use_dq_encoder \
            --freeze_llm \
            --enable_blending
    done
    
    echo "Completed task: ${task_name}"
    echo ""
done

echo "All tasks completed!"