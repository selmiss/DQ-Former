#!/bin/bash

python utils/calculate_overlap.py \
  --test \
    data/mol_instructions_processed/forward_reaction_prediction/test.jsonl \
    data/mol_instructions_processed/reagent_prediction/test.jsonl \
    data/mol_instructions_processed/retrosynthesis/test.jsonl \
  --train \
    data/overlap/training/finetune.txt \
  --output data/overlap/results/3_more_mol_instructions.txt

