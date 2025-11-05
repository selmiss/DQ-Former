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
export GPUs="4,5,6,7"  # GPU IDs to use
export MASTER_PORT=29500  # Master port for distributed training

# Launch with DeepSpeed
deepspeed --master_port ${MASTER_PORT} --include localhost:${GPUs} \
    ${BASE_DIR}/stage1_hf.py \
    --train_config_path ${BASE_DIR}/configs/stage1_dqw2d/train_config.yaml \
    --data_config_path ${BASE_DIR}/configs/stage1_dqw2d/data_config.yaml

