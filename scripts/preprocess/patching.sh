: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

python ${BASE_DIR}/utils/patching_preprocess.py \
  --input ${DATA_DIR}/zeroshot/pampa/data.jsonl \
  --output ${DATA_DIR}/zeroshot/pampa/data_brics.jsonl \
  --fallback-len 3
