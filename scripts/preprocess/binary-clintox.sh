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

# CLINTOX (Clinical Toxicity) preprocessing script
# CLINTOX has two targets: FDA_APPROVED and CT_TOX (Clinical Trial Toxicity)
# By default uses CT_TOX for toxicity prediction
# Supports molnet structure with raw/train_*.csv, raw/valid_*.csv, raw/test_*.csv
#
# Thread Limits:
#   By default, uses 16 CPU threads to avoid overwhelming shared servers.
#   To customize, set environment variables before running:
#     RDKIT_NUM_THREADS=8 OMP_NUM_THREADS=8 bash binary-clintox.sh
#
# Usage examples:
#   # Default CLINTOX_1 with CT_TOX (toxicity)
#   bash binary-clintox.sh
#
#   # Use FDA_APPROVED instead
#   TARGET_COL=FDA_APPROVED ANSWER_MAP='{"1":"Approved","0":"Not approved"}' bash binary-clintox.sh
#
#   # Custom variant
#   DATASET_VARIANT=clintox_2 bash binary-clintox.sh
#
#   # Custom thread limits (if system is heavily loaded)
#   RDKIT_NUM_THREADS=4 bash binary-clintox.sh

# Configuration - set these variables before running
DATASET_VARIANT=${DATASET_VARIANT:-clintox_1}
DATASET_NAME=${DATASET_NAME:-clintox}
SMILES_COL=${SMILES_COL:-smiles}
# Default to CT_TOX (Clinical Trial Toxicity), can override to use FDA_APPROVED
TARGET_COL=${TARGET_COL:-CT_TOX}
ANSWER_MAP=${ANSWER_MAP:-'{"1":"Toxic","0":"Non-toxic"}'}

# CLINTOX is in data/molnet/ not data/raw_sets/
INPUT_DIR=${BASE_DIR}/data/molnet/${DATASET_VARIANT}
OUT_DIR=${DATA_DIR}/zeroshot/${DATASET_NAME}
mkdir -p "${OUT_DIR}"

echo "Processing ${DATASET_NAME^^} dataset..."
echo "Target column: ${TARGET_COL}"
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

