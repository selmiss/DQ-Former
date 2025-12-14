#!/bin/bash

python utils/calculate_overlap.py \
  --test \
    data/zeroshot/ames/ames.jsonl \
    data/zeroshot/bace/bace.jsonl \
    data/zeroshot/bbbp/bbbp.jsonl \
    data/zeroshot/clintox/clintox.jsonl \
    data/zeroshot/dili/dili.jsonl \
    data/zeroshot/hiv/hiv.jsonl \
    data/zeroshot/herg/herg.jsonl \
    data/zeroshot/dili/dili.jsonl \
    data/zeroshot/hia/hia.jsonl \
    data/zeroshot/pampa/data.jsonl \
    data/zeroshot/pgp/pgp.jsonl \
    data/mol_qa/test.jsonl \
    data/mol_prop/test.jsonl \
    data/mol_gen/test.jsonl \
  --train \
    data/overlap/training/finetune.txt \
  --output data/overlap/results/all_overlap.txt

