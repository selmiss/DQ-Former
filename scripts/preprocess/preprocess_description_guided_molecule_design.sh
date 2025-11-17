#!/bin/bash

##############################################################################
# Full Preprocessing Script: Description-Guided Molecule Design
##############################################################################
# This script preprocesses the complete description-guided molecule design dataset.
#
# NOTE: This task has NO molecular graph input - only text descriptions.
#       The input is a text description, output is a molecule (SELFIES).
#       Therefore, NO graph_data is generated for this task.
#
# Dataset size:
#   - Train: 297,319 samples
#   - Test: 1,000 samples
#
# Usage:
#   bash scripts/preprocess/preprocess_description_guided_molecule_design.sh
#
# To limit training samples:
#   MAX_TRAIN_SAMPLES=50000 bash scripts/preprocess/preprocess_description_guided_molecule_design.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "FULL PREPROCESSING: Description-Guided Molecule Design"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_JSON="$PROJECT_ROOT/data/Molecule-oriented_Instructions/description_guided_molecule_design.json"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/mol_instructions_processed}"

# Allow override of max train samples via environment variable
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}  # Empty means process all

if [ -z "$MAX_TRAIN_SAMPLES" ]; then
    echo ""
    echo "Configuration:"
    echo "  Input: $INPUT_JSON"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train samples: ALL (297,319)"
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
echo "⚠️  Note: This task has NO graph input (text description only)"
echo "    Output format will NOT include graph_data"
echo "    Processing is much faster (~100-200 it/s)"

echo ""
echo "⏱️  Estimated time: ~30-60 minutes for full training set"

echo "=================================================="
echo "Starting full preprocessing..."
echo "=================================================="

# Build command
CMD="python3 \"$PROJECT_ROOT/data_provider/preprocess_text_to_mol.py\" \
    --input_json \"$INPUT_JSON\" \
    --output_dir \"$OUTPUT_DIR\" \
    --task_name \"description_guided_molecule_design\" \
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

