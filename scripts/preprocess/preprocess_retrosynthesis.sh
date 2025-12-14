#!/bin/bash
#
# Preprocess Retrosynthesis Dataset
# 
# This script processes the retrosynthesis dataset from Molecule-oriented_Instructions
# into mol_qa-like JSONL format with multi-molecule support.
#
# Usage:
#   bash scripts/preprocess_retrosynthesis.sh
#
# Options (set as environment variables):
#   MAX_TRAIN_SAMPLES - Limit training samples (default: all)
#   OUTPUT_DIR - Output directory (default: data/mol_instructions_processed)
#

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

echo "=================================================="
echo "Preprocessing Retrosynthesis Dataset"
echo "=================================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Configuration
INPUT_JSON="${PROJECT_ROOT}/data/Molecule-oriented_Instructions/retrosynthesis.json"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/mol_instructions_processed}"
TASK_NAME="retrosynthesis"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"

# Check if input file exists
if [ ! -f "${INPUT_JSON}" ]; then
    echo "❌ Error: Input file not found: ${INPUT_JSON}"
    echo "Please ensure the Molecule-oriented_Instructions dataset is downloaded."
    exit 1
fi

# Check if entropy model checkpoint exists
if [ ! -d "${PROJECT_ROOT}/checkpoints/entropy_model" ]; then
    echo "⚠️  Warning: Entropy model checkpoint not found at checkpoints/entropy_model"
    echo "Please download the entropy model checkpoint before running this script."
    exit 1
fi

# Navigate to project root
cd "${PROJECT_ROOT}"

# Build command
CMD="python data_provider/preprocess_multi_mol_instructions.py \
    --input_json ${INPUT_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --task_name ${TASK_NAME}"

# Add max_train_samples if specified
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
    CMD="${CMD} --max_train_samples ${MAX_TRAIN_SAMPLES}"
    echo "Max training samples: ${MAX_TRAIN_SAMPLES}"
else
    echo "Processing all training samples"
fi

echo ""
echo "Command: ${CMD}"
echo ""
echo "=================================================="
echo "Starting preprocessing..."
echo "=================================================="
echo ""

# Run preprocessing
eval ${CMD}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Preprocessing completed successfully!"
    echo "=================================================="
    echo "Output location: ${OUTPUT_DIR}/${TASK_NAME}/"
    echo ""
    echo "Files created:"
    ls -lh "${OUTPUT_DIR}/${TASK_NAME}/"*.jsonl 2>/dev/null || echo "  (no files found)"
    echo ""
    echo "To load the processed data:"
    echo "  from datasets import load_dataset"
    echo "  dataset = load_dataset('json', data_dir='${OUTPUT_DIR}/${TASK_NAME}')"
    echo ""
else
    echo ""
    echo "=================================================="
    echo "❌ Preprocessing failed!"
    echo "=================================================="
    exit 1
fi

