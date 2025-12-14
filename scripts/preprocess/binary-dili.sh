#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

# Set thread limits to avoid overwhelming shared servers
# Adjust these values based on your system's resources
export RDKIT_NUM_THREADS=${RDKIT_NUM_THREADS:-16}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-16}

# Generic preprocessing script for binary classification datasets
# Supports both standard directory structure with train.csv, valid.csv, test.csv
# AND molnet structure with raw/train_*.csv, raw/valid_*.csv, raw/test_*.csv
#
# Thread Limits:
#   By default, uses 16 CPU threads to avoid overwhelming shared servers.
#   To customize, set environment variables before running:
#     RDKIT_NUM_THREADS=8 OMP_NUM_THREADS=8 bash binary-dili.sh
#
# Usage examples:
#   # CLINTOX (default)
#   DATASET_VARIANT=clintox_1 DATASET_NAME=clintox bash binary.sh
#
#   # BACE
#   DATASET_VARIANT=bace_1 DATASET_NAME=bace SMILES_COL=mol TARGET_COL=Class bash binary.sh
#
#   # SIDER
#   DATASET_VARIANT=sider_1 DATASET_NAME=sider SMILES_COL=mol TARGET_COL=Class bash binary.sh
#
#   # HIV
#   DATASET_VARIANT=hiv DATASET_NAME=hiv SMILES_COL=smiles TARGET_COL=label bash binary.sh
#
#   # Custom dataset with different columns
#   DATASET_VARIANT=custom_1 DATASET_NAME=custom SMILES_COL=smiles \
#   TARGET_COL=label ANSWER_MAP='{"1":"Positive","0":"Negative"}' bash binary.sh

# Configuration - set these variables before running
DATASET_VARIANT=${DATASET_VARIANT:-DILI}
DATASET_NAME=${DATASET_NAME:-dili}
SMILES_COL=${SMILES_COL:-smiles}
TARGET_COL=${TARGET_COL:-Y}
ANSWER_MAP=${ANSWER_MAP:-'{"1":"Active","0":"Inactive"}'}

INPUT_DIR=${BASE_DIR}/data/raw_sets/${DATASET_VARIANT}
OUT_DIR=${DATA_DIR}/zeroshot/${DATASET_NAME}
mkdir -p "${OUT_DIR}"

echo "Processing ${DATASET_NAME^^} dataset..."
echo "Thread limits: RDKIT=${RDKIT_NUM_THREADS}, OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}"

python ${BASE_DIR}/data_provider/generate_from_csv.py \
  --csv_dir ${INPUT_DIR} \
  --smiles_col ${SMILES_COL} \
  --target_col ${TARGET_COL} \
  --answer_map "${ANSWER_MAP}" \
  --output_dir ${OUT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --enable_graph_features \
  --enable_brics_gids \
  --enable_entropy_gids

echo "${DATASET_NAME^^} data JSONL written to ${OUT_DIR}"
echo "Now updating meta.json with custom prompts..."

# Update the generated meta.json with our custom prompts
python -c "
import json
import os

meta_path = '${OUT_DIR}/${DATASET_NAME}_meta.json'
custom_meta_path = '${BASE_DIR}/data/zeroshot/${DATASET_NAME}/meta.json'

# Read the generated meta.json
with open(meta_path, 'r') as f:
    meta = json.load(f)

# Read our custom prompts
if os.path.exists(custom_meta_path):
    with open(custom_meta_path, 'r') as f:
        custom = json.load(f)
    # Update with custom prompts and labels
    meta['prompts'] = custom['prompts']
    if 'positive_label' in custom:
        meta['positive_label'] = custom['positive_label']
    if 'negative_label' in custom:
        meta['negative_label'] = custom['negative_label']
    # Write back
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Updated {meta_path} with custom prompts')
else:
    print(f'Custom meta not found at {custom_meta_path}, using defaults')
"

echo "${DATASET_NAME^^} processing complete. Output in ${OUT_DIR}"