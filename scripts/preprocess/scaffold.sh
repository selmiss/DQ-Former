#!/bin/bash
: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"



python utils/create_scaffold_split.py \
    --train data/mol_qa/train.jsonl \
    --test data/mol_qa/test.jsonl \
    --output data/mol_qa_scaffold \
    --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 \
    --seed 42