: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

python ${BASE_DIR}/utils/patching_preprocess.py \
  --input ${DATA_DIR}/Mol-LLaMA-Instruct/pubchem-molecules.json \
  --output ${DATA_DIR}/Mol-LLaMA-Instruct/pubchem-molecules_brics.json \
  --fallback-len 3
