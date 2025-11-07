#!/bin/bash
# MoleculeQA Finetuning Training Script
# Usage: bash scripts/moleculeqa/train_dqw2d.sh [deepspeed_stage]
# Example: bash scripts/moleculeqa/train_dqw2d.sh 2

# Set BASE_DIR for DeepSpeed config paths
export BASE_DIR=$(pwd)

# Get deepspeed stage from argument or default to 2
DEEPSPEED_STAGE=${1:-2}

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Number of GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

echo "=========================================="
echo "MoleculeQA DQ-Former Training"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_GPUS: $NUM_GPUS"
echo "DeepSpeed Stage: $DEEPSPEED_STAGE"
echo "=========================================="

# Launch training with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29501 \
    runner/qa_finetuning.py \
    --model_config_path configs/moleculeqa/dqw2d/model_config.yaml \
    --training_config_path configs/moleculeqa/dqw2d/training_config.yaml \
    --data_config_path configs/moleculeqa/dqw2d/data_config.yaml \
    --deepspeed_stage $DEEPSPEED_STAGE

echo "=========================================="
echo "Training completed!"
echo "=========================================="

