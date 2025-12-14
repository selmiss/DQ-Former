#!/bin/bash

##############################################################################
# Full Preprocessing Script: Open Question
##############################################################################
# This script preprocesses the complete open question dataset.
#
# NOTE: This task has NO molecular graph input - only text.
#       The input is a question (instruction + input), output is a text answer.
#       Therefore, NO graph_data is generated for this task.
#
# Dataset size:
#   - Train: TBD samples
#   - Test: TBD samples
#
# Usage:
#   bash scripts/preprocess/preprocess_open_question.sh
#
# To limit training samples:
#   MAX_TRAIN_SAMPLES=50000 bash scripts/preprocess/preprocess_open_question.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "FULL PREPROCESSING: Open Question"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_JSON="$PROJECT_ROOT/data/Biomolecular_Text_Instructions/open_question.json"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/mol_instructions_processed}"

# Allow override of max train samples via environment variable
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-}  # Empty means process all

if [ -z "$MAX_TRAIN_SAMPLES" ]; then
    echo ""
    echo "Configuration:"
    echo "  Input: $INPUT_JSON"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train samples: ALL"
    echo "  Test samples: ALL"
else
    echo ""
    echo "Configuration:"
    echo "  Input: $INPUT_JSON"
    echo "  Output: $OUTPUT_DIR"
    echo "  Train samples: $MAX_TRAIN_SAMPLES (limited)"
    echo "  Test samples: ALL"
fi

echo ""
echo "⚠️  Note: This task has NO graph input (text-to-text only)"
echo "    Input: instruction + input (question)"
echo "    Output: text answer"
echo "    Output format will NOT include graph_data"
echo "    Processing is much faster (~100-200 it/s)"

echo ""
echo "⏱️  Estimated time: varies based on dataset size"

echo "=================================================="
echo "Starting full preprocessing..."
echo "=================================================="

# Build command
CMD="python3 \"$PROJECT_ROOT/data_provider/preprocess_text_to_mol.py\" \
    --input_json \"$INPUT_JSON\" \
    --output_dir \"$OUTPUT_DIR\" \
    --task_name \"open_question\" \
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

