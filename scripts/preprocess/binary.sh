#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

# Example preprocessing for the BBBP dataset (binary classification: blood-brain barrier penetration)
# Expects directory with train.csv, valid.csv (or val.csv / validation.csv), and test.csv

INPUT_DIR=/data/lab_ph/zihao/Nips/dataset/sft_tdc/BBB_Martins
OUT_DIR=${DATA_DIR}/zeroshot/bbbp
mkdir -p "${OUT_DIR}"

python ${BASE_DIR}/zeroshot/generate_from_csv.py \
  --csv_dir ${INPUT_DIR} \
  --smiles_col smiles --target_col Y \
  --answer_map '{"1":"Penetrant","0":"Non-penetrant"}' \
  --output_dir ${OUT_DIR} \
  --dataset_name bbbp \
  --prompts_json ${BASE_DIR}/prompts/bbbp/task.json

echo "BBBP meta and JSONL written to ${OUT_DIR}"