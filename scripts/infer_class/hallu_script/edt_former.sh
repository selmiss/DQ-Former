#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"
: "${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY not set}"

export PYTHONPATH="${BASE_DIR}:${PYTHONPATH:-}"

MODEL_NAME="${MODEL_NAME:-unsloth/Llama-3.1-8B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_DIR}/checkpoints/edt_former_s2_large/final_model}"
QFORMER_PATH="${QFORMER_PATH:-${TOKENIZER_PATH}/model.safetensors}"
OUTPUT_NAME="${OUTPUT_NAME:-edt_former}"
CHATGPT_MODEL="${CHATGPT_MODEL:-gpt-5-mini}"
CHATGPT_TEMPERATURE="${CHATGPT_TEMPERATURE:-1.0}"
CHATGPT_MAX_RETRIES="${CHATGPT_MAX_RETRIES:-3}"
CHATGPT_RETRY_BACKOFF="${CHATGPT_RETRY_BACKOFF:-2.0}"

TASK="${TASK:-hallu_fg}"
PROMPT_TYPE="${PROMPT_TYPE:-functional_group}"

echo "Using model:            ${MODEL_NAME}"
echo "Tokenizer path:         ${TOKENIZER_PATH}"
echo "Q-Former checkpoint:    ${QFORMER_PATH}"
echo "Output name suffix:     ${OUTPUT_NAME}"
echo "ChatGPT judge model:    ${CHATGPT_MODEL}"
echo "Task:                   ${TASK}"
echo "Prompt type:            ${PROMPT_TYPE}"
echo ""

echo "=================================================="
echo "Processing task: ${TASK}"
echo "=================================================="

CUDA_VISIBLE_DEVICES=4 python "${BASE_DIR}/evaluation/chatgpt_hallu_eval.py" \
    --pretrained_model_name_or_path "${MODEL_NAME}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --data_dir "${DATA_DIR}" \
    --task_name "${TASK}" \
    --prompt_type "${PROMPT_TYPE}" \
    --qformer_path "${QFORMER_PATH}" \
    --output_name "${OUTPUT_NAME}" \
    --use_dq_encoder \
    --freeze_llm \
    --enable_blending \
    --openai_api_key "${OPENAI_API_KEY}" \
    --chatgpt_model "${CHATGPT_MODEL}" \
    --chatgpt_temperature "${CHATGPT_TEMPERATURE}" \
    --chatgpt_max_retries "${CHATGPT_MAX_RETRIES}" \
    --chatgpt_retry_backoff "${CHATGPT_RETRY_BACKOFF}"

echo "Completed task: ${TASK}"
echo ""

echo "All hallucination evaluation jobs completed!"

