#!/bin/bash

# Source environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_ROOT}/local.env.sh" ]; then
    echo "Sourcing local.env.sh..."
    source "${PROJECT_ROOT}/local.env.sh"
else
    echo "Warning: local.env.sh not found. Please ensure environment variables are set."
    : "${BASE_DIR:?Environment variable BASE_DIR not set}"
    : "${DATA_DIR:?Environment variable DATA_DIR not set}"
    export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
fi

# Configuration
export GPUs="5,6"  # GPU IDs to use for finetuning
export MASTER_PORT=29501  # Master port for distributed training (different from stage1)

# Launch with DeepSpeed - FULL TRAINING (no test mode)
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} \
    ${BASE_DIR}/finetuning_hf.py \
    --train_config_path ${BASE_DIR}/configs/stage2_dqw2d/train_config.yaml \
    --data_config_path ${BASE_DIR}/configs/stage2_dqw2d/data_config.yaml

