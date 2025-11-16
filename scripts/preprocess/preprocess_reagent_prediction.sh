#!/bin/bash

##############################################################################
# Full Preprocessing Script: Reagent Prediction
##############################################################################
# This script preprocesses the complete reagent prediction dataset.
#
# Dataset size:
#   - Train: 124,384 samples
#   - Test: 1,000 samples
#
# Usage:
#   bash scripts/preprocess/preprocess_reagent_prediction.sh
#
# To limit training samples:
#   MAX_TRAIN_SAMPLES=10000 bash scripts/preprocess/preprocess_reagent_prediction.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "FULL PREPROCESSING: Reagent Prediction"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_JSON="$PROJECT_ROOT/data/Molecule-oriented_Instructions/reagent_prediction.json"
OUTPUT_DIR="$PROJECT_ROOT/data/Molecule-oriented_Instructions_jsonl/reagent_prediction"

# Allow override of max train samples via environment variable
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}  # Empty means process all

if [ -z "$MAX_TRAIN_SAMPLES" ]; then
    echo ""
    echo "Configuration:"
    echo "  Input: $INPUT_JSON"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train samples: ALL (124,384)"
    echo "  Test samples: ALL (1,000)"
else
    echo ""
    echo "Configuration:"
    echo "  Input: $INPUT_JSON"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train samples: $MAX_TRAIN_SAMPLES (limited)"
    echo "  Test samples: ALL (1,000)"
fi

echo ""
echo "⚠️  Note: This will process the FULL dataset"
echo "    Estimated time: ~8-12 hours for full training set"
echo "    (depending on CPU cores and molecule complexity)"


echo ""
echo "=================================================="
echo "Starting full preprocessing..."
echo "=================================================="

# Build command
CMD="python3 \"$PROJECT_ROOT/data_provider/preprocess_multi_mol_instructions.py\" \
    --input_json \"$INPUT_JSON\" \
    --output_dir \"$OUTPUT_DIR\" \
    --task_name \"reagent_prediction\" \
    --skip_failures"

# Add max_train_samples if set
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    CMD="$CMD --max_train_samples $MAX_TRAIN_SAMPLES"
fi

# Execute
eval $CMD

echo ""
echo "=================================================="
echo "✅ Full preprocessing completed!"
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

echo ""
echo "✅ Dataset ready for training!"
echo ""
echo "Output location: $OUTPUT_DIR"
echo ""

