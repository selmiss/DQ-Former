: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

export PYTHONPATH=${BASE_DIR}:${PYTHONPATH}
export CUDA_VISIBLE_DEVICES=0

python ${BASE_DIR}/data_provider/precess_mol_instructions.py \
    --input_json ${DATA_DIR}/Molecule-oriented_Instructions/property_prediction.json \
    --output_dir ${DATA_DIR}/Molecule-oriented_Instructions/property_prediction/
