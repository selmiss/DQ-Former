#!/usr/bin/env python3
"""
Preprocessing script for Stage 1 pretraining data.
Converts raw molecular data to preprocessed JSONL format with graph representations.

Usage:
    python preprocess_pretrain_data.py \
        --input_json /path/to/pubchem-molecules.json \
        --output_jsonl /path/to/preprocessed_data.jsonl \
        --encoder_types unimol moleculestm \
        --max_atoms 512 \
        --batch_size 100
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from scipy.spatial import distance_matrix
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_provider.ogb_features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph representation.
    Returns node features, edge indices, and edge features as lists.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    
    # atoms
    atom_features_list = []
    if mol is not None:
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
        x = np.array(atom_features_list, dtype=np.int64)
    else:
        x = np.zeros((1, 9), dtype=np.int64)
    
    # bonds
    num_bond_features = 3
    if mol is not None and len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        
        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)
    
    return {
        'node_feat': x.tolist(),
        'edge_index': edge_index.tolist(),
        'edge_attr': edge_attr.tolist()
        # num_nodes is auto-inferred by PyG from len(node_feat)
    }


def get_unimol_data(atoms, coordinates, dictionary, max_atoms=512, remove_Hs=True):
    """
    Processes atoms and coordinates for UniMol encoder.
    Returns tokenized atoms, edge types, and distance matrix as lists.
    """
    atoms = np.array(atoms)
    coordinates = np.array(coordinates)
    
    assert len(atoms) == len(coordinates) and len(atoms) > 0
    assert coordinates.shape[1] == 3
    
    # Remove Hydrogen atoms
    if remove_Hs:
        mask_hydrogen = atoms != "H"
        if sum(mask_hydrogen) > 0:
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]
    
    # Randomly sample atoms if too many
    if len(atoms) > max_atoms:
        index = np.random.permutation(len(atoms))[:max_atoms]
        atoms = atoms[index]
        coordinates = coordinates[index]
    
    assert 0 < len(atoms) <= max_atoms
    
    # Tokenize atoms
    atom_vec = dictionary.vec_index(atoms).astype(np.int64)
    atom_vec = np.concatenate([
        [dictionary.bos()],
        atom_vec,
        [dictionary.eos()]
    ])
    
    # Center coordinates and add padding
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = np.concatenate([
        np.zeros((1, 3)),
        coordinates,
        np.zeros((1, 3))
    ], axis=0)
    
    # Compute edge types and distances
    edge_type = atom_vec.reshape(-1, 1) * len(dictionary) + atom_vec.reshape(1, -1)
    dist = distance_matrix(coordinates, coordinates).astype(np.float32)
    
    return {
        'src_tokens': atom_vec.tolist(),
        'src_edge_type': edge_type.tolist(),
        'src_distance': dist.tolist()
    }


def load_unimol_dictionary():
    """Load UniMol dictionary from HuggingFace."""
    from huggingface_hub import hf_hub_download
    from utils.unicore import Dictionary
    
    logger.info("Loading UniMol dictionary from HuggingFace...")
    unimol_dictionary_path = hf_hub_download(
        repo_id='dptech/Uni-Mol-Models',
        filename='mol.dict.txt',
    )
    unimol_dictionary = Dictionary.load(unimol_dictionary_path)
    unimol_dictionary.add_symbol("[MASK]", is_special=True)
    logger.info(f"✅ Loaded UniMol dictionary with {len(unimol_dictionary)} symbols")
    
    return unimol_dictionary


def process_molecule(data, unimol_dictionary, encoder_types, max_atoms=512):
    """
    Process a single molecule into graph representations.
    
    Args:
        data: Dictionary with molecule data (cid, smiles, atoms, coordinates, etc.)
        unimol_dictionary: UniMol dictionary for tokenization
        encoder_types: List of encoder types to process
        max_atoms: Maximum number of atoms for UniMol
        
    Returns:
        Dictionary with processed data ready for JSONL serialization
    """
    result = {
        'cid': data['cid'],
        'split': data['split'],
        'iupac_name': data['iupac_name'],
        'smiles': data['smiles'],
    }
    
    # Add optional fields
    if 'brics_gids' in data:
        result['brics_gids'] = data['brics_gids']
    if 'entropy_gids' in data:
        result['entropy_gids'] = data['entropy_gids']
    
    # Process graph representations
    graph_data = {}
    
    if 'unimol' in encoder_types:
        atoms = data['atoms']
        coordinates = data['coordinates'][0] if isinstance(data['coordinates'][0], list) else data['coordinates']
        unimol_data = get_unimol_data(
            atoms, coordinates, unimol_dictionary, max_atoms, remove_Hs=True
        )
        graph_data['unimol'] = unimol_data
    
    if 'moleculestm' in encoder_types:
        smiles = data['smiles']
        graph = smiles2graph(smiles)
        graph_data['moleculestm'] = graph
    
    result['graph_data'] = graph_data
    
    return result


def preprocess_dataset(
    input_json,
    output_dir,
    encoder_types=['unimol', 'moleculestm'],
    max_atoms=512,
    batch_size=100
):
    """
    Preprocess molecular dataset and save to separate JSONL files by split.
    
    Args:
        input_json: Path to input JSON file with raw molecular data
        output_dir: Directory to save output JSONL files (train.jsonl, val.jsonl)
        encoder_types: List of encoder types to process
        max_atoms: Maximum number of atoms for UniMol
        batch_size: Batch size for progress reporting
    """
    logger.info(f"Starting preprocessing of {input_json}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load UniMol dictionary if needed
    unimol_dictionary = None
    if 'unimol' in encoder_types:
        unimol_dictionary = load_unimol_dictionary()
    
    # Load raw data
    logger.info(f"Loading raw data from {input_json}")
    with open(input_json, 'r') as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} molecules")
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Open separate files for each split
    train_file = output_path / 'train.jsonl'
    val_file = output_path / 'val.jsonl'
    test_file = output_path / 'test.jsonl'
    
    logger.info("Processing molecules and splitting by 'split' field...")
    processed_count = {'train': 0, 'val': 0, 'test': 0}
    error_count = 0
    
    with open(train_file, 'w') as f_train, open(val_file, 'w') as f_val, open(test_file, 'w') as f_test:
        for i, data in enumerate(tqdm(raw_data, desc="Processing")):
            try:
                processed = process_molecule(
                    data, unimol_dictionary, encoder_types, max_atoms
                )
                
                # Write to appropriate file based on split
                split = data.get('split', 'pretrain')
                
                # Handle common split name variations
                if split in ['pretrain', 'train', 'training']:
                    f_train.write(json.dumps(processed) + '\n')
                    processed_count['train'] += 1
                elif split in ['valid', 'validation', 'val']:
                    f_val.write(json.dumps(processed) + '\n')
                    processed_count['val'] += 1
                elif split in ['test', 'testing']:
                    f_test.write(json.dumps(processed) + '\n')
                    processed_count['test'] += 1
                else:
                    logger.warning(f"Unknown split '{split}' for molecule {data.get('cid', i)}, skipping")
                    
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing molecule {data.get('cid', i)}: {e}")
                continue
    
    logger.info(f"✅ Preprocessing complete!")
    logger.info(f"   Train samples: {processed_count['train']}")
    logger.info(f"   Val samples: {processed_count['val']}")
    logger.info(f"   Test samples: {processed_count['test']}")
    logger.info(f"   Errors: {error_count} molecules")
    logger.info(f"   Output files:")
    logger.info(f"     - {train_file}")
    logger.info(f"     - {val_file}")
    logger.info(f"     - {test_file}")
    
    # Print file sizes
    train_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    val_size_mb = os.path.getsize(val_file) / (1024 * 1024)
    test_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    logger.info(f"   File sizes:")
    logger.info(f"     - train.jsonl: {train_size_mb:.2f} MB")
    logger.info(f"     - val.jsonl: {val_size_mb:.2f} MB")
    logger.info(f"     - test.jsonl: {test_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess molecular data for Stage 1 pretraining"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON file with raw molecular data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save output JSONL files (train.jsonl and val.jsonl)'
    )
    parser.add_argument(
        '--encoder_types',
        nargs='+',
        default=['unimol', 'moleculestm'],
        choices=['unimol', 'moleculestm'],
        help='Encoder types to process (default: unimol moleculestm)'
    )
    parser.add_argument(
        '--max_atoms',
        type=int,
        default=512,
        help='Maximum number of atoms for UniMol (default: 512)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size for progress reporting (default: 100)'
    )
    
    args = parser.parse_args()
    
    preprocess_dataset(
        input_json=args.input_json,
        output_dir=args.output_dir,
        encoder_types=args.encoder_types,
        max_atoms=args.max_atoms,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()

