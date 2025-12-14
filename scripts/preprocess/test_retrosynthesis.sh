#!/bin/bash
#
# TEST: Preprocess Retrosynthesis Dataset (Small Sample)
# 
# This script processes a SMALL SAMPLE of the retrosynthesis dataset
# for testing and training script development.
#
# Usage:
#   bash scripts/preprocess/test_retrosynthesis.sh
#
# Output: data/mol_instructions_test/retrosynthesis/
#   - train.jsonl (~100 examples)
#   - test.jsonl (~50 examples)
#
# Expected time: ~1-2 minutes
#

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

echo "=================================================="
echo "TEST: Retrosynthesis Dataset (Small Sample)"
echo "=================================================="
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Configuration
INPUT_JSON="${PROJECT_ROOT}/data/Molecule-oriented_Instructions/retrosynthesis.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/mol_instructions_test"
TASK_NAME="retrosynthesis"
TRAIN_SAMPLES=100  # Small sample for testing

echo "Configuration:"
echo "  Input: ${INPUT_JSON}"
echo "  Output: ${OUTPUT_DIR}/${TASK_NAME}/"
echo "  Train samples: ${TRAIN_SAMPLES}"
echo "  Test samples: all (will process full test set)"
echo ""

# Check prerequisites
if [ ! -f "${INPUT_JSON}" ]; then
    echo "❌ Error: Input file not found: ${INPUT_JSON}"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/checkpoints/entropy_model" ]; then
    echo "❌ Error: Entropy model checkpoint not found"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment: edtformer"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate edtformer

# Navigate to project root
cd "${PROJECT_ROOT}"

echo ""
echo "=================================================="
echo "Starting test preprocessing..."
echo "=================================================="
echo ""

# Run preprocessing with limited samples
python data_provider/preprocess_multi_mol_instructions.py \
    --input_json ${INPUT_JSON} \
    --output_dir ${OUTPUT_DIR} \
    --task_name ${TASK_NAME} \
    --max_train_samples ${TRAIN_SAMPLES}

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Test preprocessing completed!"
    echo "=================================================="
    echo ""
    
    # Show output statistics
    echo "Output files:"
    ls -lh "${OUTPUT_DIR}/${TASK_NAME}/"*.jsonl
    
    echo ""
    echo "File statistics:"
    for file in "${OUTPUT_DIR}/${TASK_NAME}/"*.jsonl; do
        if [ -f "$file" ]; then
            count=$(wc -l < "$file")
            echo "  $(basename $file): ${count} examples"
        fi
    done
    
    echo ""
    echo "Example record (first line of train.jsonl):"
    echo "---"
    head -n 1 "${OUTPUT_DIR}/${TASK_NAME}/train.jsonl" | python3 -m json.tool | head -30
    echo "..."
    
    echo ""
    echo "✅ Demo output ready for training script development!"
    echo ""
    echo "To load in Python:"
    echo "  from datasets import load_dataset"
    echo "  dataset = load_dataset('json', data_dir='${OUTPUT_DIR}/${TASK_NAME}')"
    echo ""
else
    echo ""
    echo "❌ Test preprocessing failed!"
    exit 1
fi

