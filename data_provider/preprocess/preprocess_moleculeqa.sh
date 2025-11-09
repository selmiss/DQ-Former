#!/bin/bash
# Convenient shell script for preprocessing MoleculeQA datasets

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source environment if exists
if [ -f "local.env.sh" ]; then
    source local.env.sh
fi

# Default paths
DATA_DIR="${DATA_DIR:-./data}"
MOL_QA_DIR="${MOL_QA_DIR:-$DATA_DIR/Molecule-oriented_Instructions/property_prediction_full}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$DATA_DIR/mol_prop}"

# Default parameters
ENCODER_TYPES="${ENCODER_TYPES:-unimol moleculestm}"
MAX_ATOMS="${MAX_ATOMS:-512}"

# Print configuration
echo "================================="
echo "Preprocessing MoleculeQA Datasets"
echo "================================="
echo "Project root:    $PROJECT_ROOT"
echo "MoleculeQA dir:  $MOL_QA_DIR"
echo "Output dir:      $OUTPUT_BASE_DIR"
echo "Encoder types:   $ENCODER_TYPES"
echo "Max atoms:       $MAX_ATOMS"
echo "================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# Check if MoleculeQA directory exists
if [ ! -d "$MOL_QA_DIR" ]; then
    echo "❌ Error: MoleculeQA directory not found: $MOL_QA_DIR"
    echo ""
    echo "Please set MOL_QA_DIR environment variable or place your data at:"
    echo "  $DATA_DIR/MoleculeQA/"
    exit 1
fi

SUCCESS_COUNT=0
FAIL_COUNT=0

# # Process train split
# echo ""
# echo "---------------------------------"
# echo "Processing: train split"
# echo "---------------------------------"

# TRAIN_MOL_JSON="$MOL_QA_DIR/train_mol.json"
# TRAIN_INSTRUCTION_JSON="$MOL_QA_DIR/train.json"
# TRAIN_OUTPUT="$OUTPUT_BASE_DIR/train.jsonl"

# if [ ! -f "$TRAIN_MOL_JSON" ]; then
#     echo "⚠️  Warning: Train molecule file not found: $TRAIN_MOL_JSON"
#     echo "   Skipping train split..."
#     FAIL_COUNT=$((FAIL_COUNT + 1))
# elif [ ! -f "$TRAIN_INSTRUCTION_JSON" ]; then
#     echo "⚠️  Warning: Train instruction file not found: $TRAIN_INSTRUCTION_JSON"
#     echo "   Skipping train split..."
#     FAIL_COUNT=$((FAIL_COUNT + 1))
# else
#     python data_provider/preprocess/preprocess_moleculeqa_data.py \
#         --mol_json "$TRAIN_MOL_JSON" \
#         --instruction_json "$TRAIN_INSTRUCTION_JSON" \
#         --output_jsonl "$TRAIN_OUTPUT" \
#         --encoder_types $ENCODER_TYPES \
#         --max_atoms "$MAX_ATOMS"
    
#     if [ $? -eq 0 ]; then
#         echo "✅ Success: train split"
#         SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
#         # Print file size
#         if [ -f "$TRAIN_OUTPUT" ]; then
#             SIZE=$(du -h "$TRAIN_OUTPUT" | cut -f1)
#             echo "   File size: $SIZE"
#         fi
#     else
#         echo "❌ Failed: train split"
#         FAIL_COUNT=$((FAIL_COUNT + 1))
#     fi
# fi

# Process test split
echo ""
echo "---------------------------------"
echo "Processing: test split"
echo "---------------------------------"

TEST_MOL_JSON="$MOL_QA_DIR/test_mol.json"
TEST_INSTRUCTION_JSON="$MOL_QA_DIR/test.json"
TEST_OUTPUT="$OUTPUT_BASE_DIR/test.jsonl"

if [ ! -f "$TEST_MOL_JSON" ]; then
    echo "⚠️  Warning: Test molecule file not found: $TEST_MOL_JSON"
    echo "   Skipping test split..."
    FAIL_COUNT=$((FAIL_COUNT + 1))
elif [ ! -f "$TEST_INSTRUCTION_JSON" ]; then
    echo "⚠️  Warning: Test instruction file not found: $TEST_INSTRUCTION_JSON"
    echo "   Skipping test split..."
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    python data_provider/preprocess/preprocess_moleculeqa_data.py \
        --mol_json "$TEST_MOL_JSON" \
        --instruction_json "$TEST_INSTRUCTION_JSON" \
        --output_jsonl "$TEST_OUTPUT" \
        --encoder_types $ENCODER_TYPES \
        --max_atoms "$MAX_ATOMS"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: test split"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        
        # Print file size
        if [ -f "$TEST_OUTPUT" ]; then
            SIZE=$(du -h "$TEST_OUTPUT" | cut -f1)
            echo "   File size: $SIZE"
        fi
    else
        echo "❌ Failed: test split"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# Final summary
echo ""
echo "================================="
echo "Preprocessing Summary"
echo "================================="
echo "✅ Successful: $SUCCESS_COUNT"
echo "❌ Failed:     $FAIL_COUNT"
echo "================================="

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo ""
    echo "Output directory: $OUTPUT_BASE_DIR"
    echo ""
    echo "Files created:"
    [ -f "$TRAIN_OUTPUT" ] && echo "  ✓ train.jsonl"
    [ -f "$TEST_OUTPUT" ] && echo "  ✓ test.jsonl"
    echo ""
    echo "To use the preprocessed data in your training:"
    echo "  1. Update your data config:"
    echo "     use_preprocessed: true"
    echo "     preprocessed_data: $OUTPUT_BASE_DIR"
    echo ""
    echo "  2. Or upload to HuggingFace Hub:"
    echo "     python data_provider/preprocess/upload_to_hub.py \\"
    echo "       --data_dir $OUTPUT_BASE_DIR \\"
    echo "       --repo_id username/moleculeqa-preprocessed \\"
    echo "       --dataset_type moleculeqa"
fi

# Exit with error if all failed
if [ $SUCCESS_COUNT -eq 0 ]; then
    exit 1
fi

