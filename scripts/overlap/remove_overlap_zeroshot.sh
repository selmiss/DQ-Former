#!/bin/bash

python utils/remove_overlap.py \
  --dataset \
    data/zeroshot/ames \
    data/zeroshot/bace \
    data/zeroshot/bbbp \
    data/zeroshot/clintox \
    data/zeroshot/dili \
    data/zeroshot/hiv \
    data/zeroshot/herg \
    data/zeroshot/dili \
    data/zeroshot/hia \
    data/zeroshot/pampa \
    data/zeroshot/pgp \
  --train \
    data/overlap/training/finetune.txt \
  --output data/zeroshot

