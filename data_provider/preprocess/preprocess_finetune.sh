#!/bin/bash
# Convenient shell script for preprocessing finetuning instruction datasets

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Source environment if exists
if [ -f "local.env.sh" ]; then
    source local.env.sh
fi

# Default paths
DATA_DIR="${DATA_DIR:-./data}"
MOL_JSON="${MOL_JSON:-$DATA_DIR/Mol-LLaMA-Instruct/pubchem-molecules_brics_entropy_gids.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$DATA_DIR/finetune}"

# Default parameters
ENCODER_TYPES="${ENCODER_TYPES:-unimol moleculestm}"
MAX_ATOMS="${MAX_ATOMS:-512}"
VAL_RATIO="${VAL_RATIO:-0.01}"
RANDOM_SEED="${RANDOM_SEED:-42}"

# You can specify multiple instruction files to preprocess
# Default: preprocess some common ones
INSTRUCTION_FILES="${INSTRUCTION_FILES:-comprehensive_conversations.json detailed_structural_descriptions.json structure2chemical_features_relationships.json structure2biological_features_relationships.json}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "================================="
echo "Preprocessing Finetuning Datasets"
echo "================================="
echo "Project root:    $PROJECT_ROOT"
echo "Molecule JSON:   $MOL_JSON"
echo "Output base dir: $OUTPUT_BASE_DIR"
echo "Encoder types:   $ENCODER_TYPES"
echo "Max atoms:       $MAX_ATOMS"
echo "Val ratio:       $VAL_RATIO"
echo "Random seed:     $RANDOM_SEED"
echo "Instruction files: $INSTRUCTION_FILES"
echo "================================="
echo ""

# Check if molecule file exists
if [ ! -f "$MOL_JSON" ]; then
    echo "❌ Error: Molecule file not found: $MOL_JSON"
    echo ""
    echo "Please set MOL_JSON environment variable or place your data at:"
    echo "  $DATA_DIR/Mol-LLaMA-Instruct/pubchem-molecules_brics_entropy_gids.json"
    exit 1
fi

# Process each instruction file
SUCCESS_COUNT=0
FAIL_COUNT=0

for inst_file in $INSTRUCTION_FILES; do
    echo ""
    echo "---------------------------------"
    echo "Processing: $inst_file"
    echo "---------------------------------"
    
    INPUT_JSON="$DATA_DIR/Mol-LLaMA-Instruct/$inst_file"
    
    # Check if instruction file exists
    if [ ! -f "$INPUT_JSON" ]; then
        echo "⚠️  Warning: File not found: $INPUT_JSON"
        echo "   Skipping..."
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    # Generate output directory for this instruction file
    BASE_NAME=$(basename "$inst_file" .json)
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$BASE_NAME"
    
    mkdir -p "$OUTPUT_DIR"
    
    # Run preprocessing
    python runner/ft_datasets/preprocess_finetune_data.py \
        --input_json "$INPUT_JSON" \
        --mol_json "$MOL_JSON" \
        --output_dir "$OUTPUT_DIR" \
        --encoder_types $ENCODER_TYPES \
        --max_atoms "$MAX_ATOMS" \
        --val_ratio "$VAL_RATIO" \
        --random_seed "$RANDOM_SEED"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ Success: $BASE_NAME"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Failed: $BASE_NAME"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

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
    echo "Output base directory: $OUTPUT_BASE_DIR"
    echo ""
    echo "Each instruction file has its own directory with train.jsonl and val.jsonl:"
    echo "  $OUTPUT_BASE_DIR/comprehensive_conversations/"
    echo "    ├── train.jsonl"
    echo "    └── val.jsonl"
    echo ""
    echo "Update your data config to use:"
    echo "  use_preprocessed: true"
    echo "  preprocessed_data: $OUTPUT_BASE_DIR/your-instruction-name"
fi

# Exit with error if all failed
if [ $SUCCESS_COUNT -eq 0 ]; then
    exit 1
fi

