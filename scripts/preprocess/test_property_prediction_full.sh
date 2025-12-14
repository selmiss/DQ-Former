#!/bin/bash

##############################################################################
# Test Script: Property Prediction (Full) - Small Demo Output
##############################################################################
# This script generates a small demo dataset for property prediction (full)
# to facilitate training script development.
#
# Demo size:
#   - Train: 1000 samples (out of 354,272)
#   - Test: All test samples (1,946)
#
# Usage:
#   bash scripts/preprocess/test_property_prediction_full.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "TEST: Property Prediction Full Dataset (Small Sample)"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_DIR="$PROJECT_ROOT/data/Molecule-oriented_Instructions/property_prediction_full"
OUTPUT_DIR="$PROJECT_ROOT/data/mol_instructions_test/property_prediction_full"
MAX_TRAIN_SAMPLES=1000

echo ""
echo "Configuration:"
echo "  Input dir: $INPUT_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Train samples: $MAX_TRAIN_SAMPLES"
echo "  Test samples: all"

# Activate conda environment
echo ""
echo "Activating conda environment: edtformer"
eval "$(conda shell.bash hook)"
conda activate edtformer

echo ""
echo "=================================================="
echo "Starting test preprocessing..."
echo "=================================================="

# Run preprocessing for property prediction full
python3 "$PROJECT_ROOT/data_provider/preprocess_property_prediction.py" \
    --input_json_train "$INPUT_DIR/train.json" \
    --input_json_test "$INPUT_DIR/test.json" \
    --output_dir "$OUTPUT_DIR" \
    --task_name "property_prediction_full" \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --skip_failures

echo ""
echo "=================================================="
echo "✅ Test preprocessing completed!"
echo "=================================================="

# Show output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.jsonl

# Show statistics
echo ""
echo "File statistics:"
for f in "$OUTPUT_DIR"/*.jsonl; do
    echo "  $(basename $f): $(wc -l < $f) examples"
done

# Show first example
echo ""
echo "Example record (first line of train.jsonl):"
echo "---"
python3 -c "
import json
with open('$OUTPUT_DIR/train.jsonl') as f:
    record = json.loads(f.readline())
    print(f\"CID: {record['cid']}\")
    print(f\"SMILES: {record.get('smiles', 'N/A')}\")
    print(f\"Has graph_data: {'graph_data' in record}\")
    print(f\"Has brics_gids: {'brics_gids' in record}\")
    print(f\"Has entropy_gids: {'entropy_gids' in record}\")
    print(f\"Conversation user: {record['conversations'][0]['user'][:150]}...\")
    print(f\"Conversation assistant: {record['conversations'][0]['assistant']}\")
"

echo ""
echo "✅ Demo output ready for training script development!"
echo ""
echo "To load in Python:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('json', data_dir='$OUTPUT_DIR')"
echo ""

