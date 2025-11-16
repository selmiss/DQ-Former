#!/bin/bash

##############################################################################
# Test Script: Description Guided Molecule Design - Small Demo Output
##############################################################################
# This script generates a small demo dataset for description-guided molecule
# design to facilitate training script development.
#
# NOTE: This task has NO molecular graph input - only text descriptions.
#       The input is a text description, output is a molecule (SELFIES).
#       Therefore, NO graph_data is generated for this task.
#
# Demo size:
#   - Train: 1000 samples (out of 297,319)
#   - Test: All test samples (1,000)
#
# Usage:
#   bash scripts/preprocess/test_description_guided_molecule_design.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "TEST: Description Guided Molecule Design (Small Sample)"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_JSON="$PROJECT_ROOT/data/Molecule-oriented_Instructions/description_guided_molecule_design.json"
OUTPUT_DIR="$PROJECT_ROOT/data/mol_instructions_test/description_guided_molecule_design"
MAX_TRAIN_SAMPLES=1000

echo ""
echo "Configuration:"
echo "  Input: $INPUT_JSON"
echo "  Output: $OUTPUT_DIR"
echo "  Train samples: $MAX_TRAIN_SAMPLES"
echo "  Test samples: all"
echo ""
echo "⚠️  Note: This task has NO graph input (text description only)"
echo "    Output format will NOT include graph_data"

echo ""
echo "=================================================="
echo "Starting test preprocessing..."
echo "=================================================="

# Run preprocessing for description-guided molecule design
python3 "$PROJECT_ROOT/data_provider/preprocess_text_to_mol.py" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --task_name "description_guided_molecule_design" \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --skip_failures

echo ""
echo "=================================================="
echo "✅ Test preprocessing completed!"
echo "=================================================="

# Show output files
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "No JSONL files found yet"

# Show statistics
echo ""
echo "File statistics:"
for f in "$OUTPUT_DIR"/*.jsonl 2>/dev/null; do
    [ -f "$f" ] && echo "  $(basename $f): $(wc -l < $f) examples"
done

# Show first example if exists
if [ -f "$OUTPUT_DIR/train.jsonl" ]; then
    echo ""
    echo "Example record (first line of train.jsonl):"
    echo "---"
    python3 -c "
import json
with open('$OUTPUT_DIR/train.jsonl') as f:
    record = json.loads(f.readline())
    print(f\"CID: {record['cid']}\")
    print(f\"Has graph_data: {'graph_data' in record}\")
    print(f\"Conversation user: {record['conversations'][0]['user'][:150]}...\")
    print(f\"Conversation assistant (SELFIES output): {record['conversations'][0]['assistant'][:100]}...\")
"
fi

echo ""
echo "✅ Demo output ready for training script development!"
echo ""
echo "To load in Python:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('json', data_dir='$OUTPUT_DIR')"
echo ""

