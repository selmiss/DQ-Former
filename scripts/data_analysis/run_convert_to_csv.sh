#!/bin/bash

# Script to run convert_to_csv.py with configurable input/output paths
# Usage: ./run_convert_to_csv.sh <dataset_name> [input_dir] [output_dir]
#
# Example:
#   ./run_convert_to_csv.sh bace
#   ./run_convert_to_csv.sh bace /path/to/custom/input /path/to/custom/output

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Default paths
UTILS_DIR="$PROJECT_ROOT/utils"
DATA_RESULTS_DIR="$PROJECT_ROOT/data/results"
OUTPUT_DIR="$SCRIPT_DIR"

# Check if dataset name is provided
if [ -z "$1" ]; then
    echo "Error: Dataset name is required"
    echo "Usage: $0 <dataset_name> [input_dir] [output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 bace"
    echo "  $0 bace /custom/input/path /custom/output/path"
    exit 1
fi

DATASET_NAME="$1"

# Use custom input directory if provided, otherwise use default
if [ -n "$2" ]; then
    INPUT_DIR="$2"
else
    INPUT_DIR="$DATA_RESULTS_DIR/$DATASET_NAME"
fi

# Use custom output directory if provided, otherwise use default (this script's directory)
if [ -n "$3" ]; then
    OUTPUT_DIR="$3"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Set output file path
OUTPUT_FILE="$OUTPUT_DIR/${DATASET_NAME}_results_table.csv"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if convert_to_csv.py exists
if [ ! -f "$UTILS_DIR/convert_to_csv.py" ]; then
    echo "Error: convert_to_csv.py not found at: $UTILS_DIR/convert_to_csv.py"
    exit 1
fi

# Print configuration
echo "========================================"
echo "Converting Results to CSV"
echo "========================================"
echo "Dataset: $DATASET_NAME"
echo "Input directory: $INPUT_DIR"
echo "Output file: $OUTPUT_FILE"
echo "========================================"
echo ""

# Run the conversion script
python3 "$UTILS_DIR/convert_to_csv.py" \
    --input_dir "$INPUT_DIR" \
    --output_file "$OUTPUT_FILE" \
    --dataset_name "$DATASET_NAME"

# Check if the script was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Success! CSV file created at:"
    echo "$OUTPUT_FILE"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Error: Conversion failed"
    echo "========================================"
    exit 1
fi

