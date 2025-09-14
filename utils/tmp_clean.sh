python ./utils/preprocess_smiles.py \
  --input data/Molecule-oriented_Instructions/property_prediction_full/train.json \
  --output data/Molecule-oriented_Instructions/property_prediction_full/train_clean.json \
  --log data/Molecule-oriented_Instructions/property_prediction_full/train_invalid.log \
  --smiles-key smiles

python ./utils/preprocess_smiles.py \
  --input data/Molecule-oriented_Instructions/property_prediction_full/test.json \
  --output data/Molecule-oriented_Instructions/property_prediction_full/test_clean.json \
  --log data/Molecule-oriented_Instructions/property_prediction_full/test_invalid.log \
  --smiles-key smiles