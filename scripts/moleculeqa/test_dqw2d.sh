#!/bin/bash
# MoleculeQA Quick Test Script (single GPU, small dataset)
# Usage: bash scripts/moleculeqa/test_dqw2d.sh

# Set BASE_DIR for config paths
export BASE_DIR=$(pwd)

# Set visible GPUs (single GPU for testing)
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "MoleculeQA Test Mode (Small Dataset)"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Launch training in test mode (no DeepSpeed, small dataset)
python trainer/moleculeqa_finetuning.py \
    --model_config_path configs/moleculeqa/dqw2d/model_config.yaml \
    --training_config_path configs/moleculeqa/dqw2d/training_config.yaml \
    --data_config_path configs/moleculeqa/dqw2d/data_config.yaml \
    --test_mode

echo "=========================================="
echo "Test completed!"
echo "=========================================="

