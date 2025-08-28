: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

python ${BASE_DIR}/utils/patching_preprocess.py \
  --input ${DATA_DIR}/moleculeqa/train_mol.json \
  --output ${DATA_DIR}/moleculeqa/train_mol_brics.json \
  --fallback-len 3
