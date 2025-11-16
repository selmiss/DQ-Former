#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

# Process only HIV test split
echo "Processing HIV test split only..."

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}

# Run the generation using single CSV file (test only)
python ${BASE_DIR}/data_provider/generate_from_csv.py \
  --csv ${BASE_DIR}/data/molnet/hiv/test_hiv_1.csv \
  --smiles_col smiles \
  --target_col label \
  --answer_map '{"1":"Active","0":"Inactive"}' \
  --output_dir ${DATA_DIR}/zeroshot/hiv \
  --dataset_name hiv \
  --train_ratio 0.0 \
  --val_ratio 0.0 \
  --test_ratio 1.0

echo "HIV test data JSONL written to ${DATA_DIR}/zeroshot/hiv"
echo "Now updating meta.json with custom prompts..."

# Update the generated meta.json with our custom prompts
python -c "
import json
import os

meta_path = '${DATA_DIR}/zeroshot/hiv/hiv_meta.json'
custom_meta_path = '${BASE_DIR}/data/zeroshot/hiv/meta.json'

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

echo "HIV test-only processing complete. Output in ${DATA_DIR}/zeroshot/hiv"

