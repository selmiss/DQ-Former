#!/bin/bash
# Convenient shell script for preprocessing molecular datasets

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source environment if exists
if [ -f "local.env.sh" ]; then
    source local.env.sh
fi

# Default paths
DATA_DIR="${DATA_DIR:-./data}"
INPUT_JSON="${INPUT_JSON:-$DATA_DIR/Mol-LLaMA-Instruct/pubchem-molecules_brics_entropy_gids.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/pretrain/}"

# Default parameters
ENCODER_TYPES="${ENCODER_TYPES:-unimol moleculestm}"
MAX_ATOMS="${MAX_ATOMS:-512}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "================================="
echo "Preprocessing Molecular Dataset"
echo "================================="
echo "Project root:    $PROJECT_ROOT"
echo "Input JSON:      $INPUT_JSON"
echo "Output DIR:      $OUTPUT_DIR"
echo "  - train.jsonl"
echo "  - val.jsonl"
echo "  - test.jsonl"
echo "Encoder types:   $ENCODER_TYPES"
echo "Max atoms:       $MAX_ATOMS"
echo "================================="
echo ""

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "❌ Error: Input file not found: $INPUT_JSON"
    echo ""
    echo "Please set INPUT_JSON environment variable or place your data at:"
    echo "  $DATA_DIR/Mol-LLaMA-Instruct/pubchem-molecules.json"
    exit 1
fi

# Run preprocessing
python runner/dataset_creator/preprocess_pretrain_data.py \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --encoder_types $ENCODER_TYPES \
    --max_atoms "$MAX_ATOMS"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "================================="
    echo "✅ Preprocessing complete!"
    echo "================================="
    echo "Output directory: $OUTPUT_DIR"
    echo "  - train.jsonl: $(ls -lh $OUTPUT_DIR/train.jsonl 2>/dev/null | awk '{print $5}')"
    echo "  - val.jsonl:   $(ls -lh $OUTPUT_DIR/val.jsonl 2>/dev/null | awk '{print $5}')"
    echo "  - test.jsonl:  $(ls -lh $OUTPUT_DIR/test.jsonl 2>/dev/null | awk '{print $5}')"
    echo ""
    echo "Update your data config to use:"
    echo "  use_preprocessed: true"
    echo "  preprocessed_data: $OUTPUT_DIR"
else
    echo ""
    echo "❌ Preprocessing failed! Check the error messages above."
    exit 1
fi

