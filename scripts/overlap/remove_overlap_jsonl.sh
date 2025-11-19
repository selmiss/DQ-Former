#!/bin/bash

python utils/remove_overlap_jsonl.py \
  --jsonl \
    data/mol_qa/test.jsonl \
    data/mol_prop/test.jsonl \
    data/mol_gen/test.jsonl \
  --train \
    data/overlap/training/finetune.txt \
  --output data

