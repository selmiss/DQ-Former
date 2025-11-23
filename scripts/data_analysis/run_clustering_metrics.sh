#!/bin/bash

# Script to calculate ARI and NMI between FOUR clustering methods:
# 1. Key1 (e.g., entropy_gids)
# 2. Key2 (e.g., brics_gids)
# 3. RECAP (computed from SMILES using RDKit with 3s timeout)
# 4. Random baseline (shuffled version of Key2)
#
# The script calculates all pairwise comparisons and outputs mean/median matrices
# Note: Molecules that take >3s for RECAP will be automatically skipped

# Base directory (adjust if needed)
BASE_DIR="/home/UWO/zjing29/proj/DQ-Former"
UTILS_DIR="${BASE_DIR}/utils"

# Main comparison: entropy_gids vs brics_gids (+ RECAP + Random)
echo "Comparing: entropy_gids, brics_gids, RECAP (from SMILES, timeout=3s), and Random baseline"
python ${UTILS_DIR}/calculate_clustering_metrics.py \
    --file ${BASE_DIR}/data/finetune/comprehensive_conversations-preprocessed.jsonl \
    --key1 entropy_gids \
    --key2 brics_gids

# Example 2: Compare other clustering methods
# echo ""
# echo "Comparing different methods"
# python ${UTILS_DIR}/calculate_clustering_metrics.py \
#     --file ${BASE_DIR}/data/train.jsonl \
#     --key1 method_a_ids \
#     --key2 method_b_ids

