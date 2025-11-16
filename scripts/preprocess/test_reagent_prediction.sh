
#!/bin/bash

##############################################################################
# Test Script: Reagent Prediction - Small Demo Output
##############################################################################
# This script generates a small demo dataset for reagent prediction
# to facilitate training script development.
#
# NOTE: Input format is: reactants >> products (reaction SMILES)
#       We process BOTH reactants and products for graph data.
#
# Demo size:
#   - Train: 100 samples (out of 124,384)
#   - Test: All test samples (1,000)
#
# Usage:
#   bash scripts/preprocess/test_reagent_prediction.sh
##############################################################################

set -e  # Exit on error

echo "=================================================="
echo "TEST: Reagent Prediction Dataset (Small Sample)"
echo "=================================================="

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Configuration
INPUT_JSON="$PROJECT_ROOT/data/Molecule-oriented_Instructions/reagent_prediction.json"
OUTPUT_DIR="$PROJECT_ROOT/data/mol_instructions_test/reagent_prediction"
MAX_TRAIN_SAMPLES=100

echo ""
echo "Configuration:"
echo "  Input: $INPUT_JSON"
echo "  Output: $OUTPUT_DIR"
echo "  Train samples: $MAX_TRAIN_SAMPLES"
echo "  Test samples: all (will process full test set)"
echo ""
echo "⚠️  Note: Reagent prediction has reaction format (reactants >> products)"
echo "    Processing may be slower due to multiple molecules"


echo ""
echo "=================================================="
echo "Starting test preprocessing..."
echo "=================================================="

# Run preprocessing for reagent prediction
python3 "$PROJECT_ROOT/data_provider/preprocess_multi_mol_instructions.py" \
    --input_json "$INPUT_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --task_name "reagent_prediction" \
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
    if isinstance(record.get('smiles'), list):
        print(f\"Number of input molecules: {len(record['smiles'])}\")
        print(f\"SMILES (first 3): {record['smiles'][:3]}\")
        print(f\"graph_data elements: {len(record.get('graph_data', []))}\")
    else:
        print(f\"SMILES: {record['smiles'][:100]}...\")
    print(f\"Conversation user: {record['conversations'][0]['user'][:100]}...\")
    print(f\"Conversation assistant (output): {record['conversations'][0]['assistant'][:100]}...\")
"

echo ""
echo "✅ Demo output ready for training script development!"
echo ""
echo "To load in Python:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('json', data_dir='$OUTPUT_DIR')"
echo ""

