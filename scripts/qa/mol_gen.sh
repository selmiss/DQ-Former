#!/bin/bash
# Molecule Generation/Captioning Training Script
# Usage: bash scripts/qa/mol_gen.sh [deepspeed_stage] [--clear-cache]
# Example: bash scripts/qa/mol_gen.sh 2
# Example with cache clearing: bash scripts/qa/mol_gen.sh 2 --clear-cache

# Set BASE_DIR for DeepSpeed config paths
export BASE_DIR=$(pwd)

# Get deepspeed stage from argument or default to 2
DEEPSPEED_STAGE=${1:-2}

# Check for --clear-cache flag
CLEAR_CACHE_FLAG=""
if [[ "$*" == *"--clear-cache"* ]]; then
    CLEAR_CACHE_FLAG="--clear_cache"
    echo "Cache clearing enabled"
fi

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "=========================================="
echo "Molecule Generation DQ-Former Training"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: $NUM_GPUS"
echo "DeepSpeed Stage: $DEEPSPEED_STAGE"
echo "=========================================="

# Launch training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29502 \
    runner/qa_finetuning.py \
    --model_config_path configs/qa/mol_gen/model_config.yaml \
    --training_config_path configs/qa/mol_gen/training_config.yaml \
    --data_config_path configs/qa/mol_gen/data_config_preprocessed.yaml \
    --deepspeed_stage $DEEPSPEED_STAGE \
    $CLEAR_CACHE_FLAG

echo "=========================================="
echo "Training completed!"
echo "=========================================="


