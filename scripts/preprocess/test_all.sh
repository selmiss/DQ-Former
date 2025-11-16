#!/bin/bash
#
# TEST: Preprocess All Datasets (Small Samples)
# 
# This script processes SMALL SAMPLES of both datasets
# for testing and training script development.
#
# Usage:
#   bash scripts/preprocess/test_all.sh
#
# Output: data/mol_instructions_test/
#   - retrosynthesis/
#   - forward_reaction_prediction/
#
# Expected time: ~3-5 minutes
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================================="
echo "TEST: Processing All Datasets (Small Samples)"
echo "=================================================="
echo ""
echo "This will create demo datasets for:"
echo "  1. Retrosynthesis (~100 train + test)"
echo "  2. Forward Reaction Prediction (~100 train + test)"
echo ""
echo "Expected time: ~3-5 minutes"
echo ""

# Run retrosynthesis test
echo "=================================================="
echo "Step 1/2: Retrosynthesis"
echo "=================================================="
bash "${SCRIPT_DIR}/test_retrosynthesis.sh"

if [ $? -ne 0 ]; then
    echo "❌ Retrosynthesis test failed!"
    exit 1
fi

echo ""
echo "=================================================="
echo "Step 2/2: Forward Reaction Prediction"
echo "=================================================="
bash "${SCRIPT_DIR}/test_forward_reaction.sh"

if [ $? -ne 0 ]; then
    echo "❌ Forward reaction test failed!"
    exit 1
fi

# Summary
echo ""
echo "=================================================="
echo "✅ ALL TEST PREPROCESSING COMPLETE!"
echo "=================================================="
echo ""
echo "Output location: data/mol_instructions_test/"
echo ""
echo "Available datasets:"
echo "  - retrosynthesis/"
echo "  - forward_reaction_prediction/"
echo ""

# Show directory structure
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
OUTPUT_DIR="${PROJECT_ROOT}/data/mol_instructions_test"

if [ -d "${OUTPUT_DIR}" ]; then
    echo "Directory structure:"
    ls -lh "${OUTPUT_DIR}"/*/*.jsonl 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
fi

echo "Quick verification script:"
echo "---"
cat << 'EOF'
python3 << 'PYEOF'
from datasets import load_dataset

# Load datasets
retro = load_dataset('json', data_dir='data/mol_instructions_test/retrosynthesis')
forward = load_dataset('json', data_dir='data/mol_instructions_test/forward_reaction_prediction')

print("✅ Datasets loaded successfully!")
print(f"\nRetrosynthesis:")
print(f"  Train: {len(retro['train'])} examples")
print(f"  Test: {len(retro['test'])} examples")
print(f"\nForward Reaction:")
print(f"  Train: {len(forward['train'])} examples")
print(f"  Test: {len(forward['test'])} examples")

# Show example
print(f"\nRetrosynthesis example:")
ex = retro['train'][0]
print(f"  Input molecules: {len(ex['smiles'])}")
print(f"  graph_data: {len(ex['graph_data'])}")
print(f"  Keys: {list(ex.keys())}")
PYEOF
EOF
echo "---"
echo ""
echo "You can now start developing your training scripts!"

