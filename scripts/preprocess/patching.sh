: "${BASE_DIR:?Environment variable BASE_DIR not set}"
: "${DATA_DIR:?Environment variable DATA_DIR not set}"

# Usage:
#   ./patching.sh [relative_json_path ...]
# If no arguments are provided, it defaults to the original single file.

if [ "$#" -gt 0 ]; then
  FILE_LIST=("$@")
else
  FILE_LIST=("data/Molecule-oriented_Instructions/mol_gen_full/train_mol.json" \
  "data/Molecule-oriented_Instructions/property_prediction_full/train_mol.json" \
  "data/Molecule-oriented_Instructions/mol_gen_full/test_mol.json" \
  "data/Molecule-oriented_Instructions/property_prediction_full/test_mol.json")
fi

for REL_PATH in "${FILE_LIST[@]}"; do
  IN_PATH="${BASE_DIR}/${REL_PATH}"
  OUT_PATH="${IN_PATH%.json}_brics.json"

  python ${BASE_DIR}/utils/patching_preprocess.py \
    --input "${IN_PATH}" \
    --output "${OUT_PATH}" \
    --fallback-len 3

  mv -f "${OUT_PATH}" "${IN_PATH}"
done
