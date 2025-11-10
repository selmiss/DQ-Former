#!/usr/bin/env python3
"""
Preprocessing script for MoleculeQA data.
Converts raw instruction + molecule data to preprocessed JSONL format with graph representations.

Usage:
    python preprocess_moleculeqa_data.py \
        --mol_json data/MoleculeQA/train_mol.json \
        --instruction_json data/MoleculeQA/train.json \
        --output_jsonl data/moleculeqa_preprocessed/train.jsonl \
        --encoder_types unimol moleculestm \
        --max_atoms 512
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from scipy.spatial import distance_matrix

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
    """Converts SMILES string to graph representation for MoleculeSTM."""
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
    """Processes atoms and coordinates for UniMol encoder."""
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
    
    # Randomly sample atoms if too many (consistent with finetuning)
    if len(atoms) > max_atoms:
        index = np.random.permutation(len(atoms))[:max_atoms]
        atoms = atoms[index]
        coordinates = coordinates[index]
    
    assert 0 < len(atoms) <= max_atoms
    
    # Tokenize atoms using vec_index (consistent with finetuning)
    atom_vec = dictionary.vec_index(atoms).astype(np.int64)
    atom_vec = np.concatenate([
        [dictionary.bos()],
        atom_vec,
        [dictionary.eos()]
    ])
    
    # Center coordinates and add padding (consistent with finetuning)
    coordinates = coordinates - coordinates.mean(axis=0)
    coordinates = np.concatenate([
        np.zeros((1, 3)),
        coordinates,
        np.zeros((1, 3))
    ], axis=0)
    
    # Compute edge types and distances (consistent with finetuning)
    edge_type = atom_vec.reshape(-1, 1) * len(dictionary) + atom_vec.reshape(1, -1)
    dist = distance_matrix(coordinates, coordinates).astype(np.float32)
    
    return {
        'src_tokens': atom_vec.tolist(),
        'src_edge_type': edge_type.tolist(),
        'src_distance': dist.tolist()
    }


def process_moleculeqa_sample(
    instruction_data,
    mol_data,
    unimol_dictionary,
    encoder_types,
    max_atoms=512
):
    """
    Process a single MoleculeQA instruction sample with its molecule data.
    
    Args:
        instruction_data: Dictionary with instruction/conversation data
        mol_data: Molecule data (atoms, coordinates, etc.) - aligned by index
        unimol_dictionary: UniMol dictionary for atom tokenization
        encoder_types: List of encoder types to process
        max_atoms: Maximum number of atoms for UniMol
    
    Returns:
        Dictionary with instruction data and serialized graph representations
    """
    cid = instruction_data['cid']
    
    # Process graph data based on encoder types
    graph_data = {}
    
    if 'unimol' in encoder_types:
        # coordinates is a list of conformations, use the first one
        coordinates = mol_data['coordinates'][0] if mol_data['coordinates'] else []
        unimol_data = get_unimol_data(
            mol_data['atoms'],
            coordinates,
            unimol_dictionary,
            max_atoms=max_atoms
        )
        graph_data['unimol'] = unimol_data
    
    if 'moleculestm' in encoder_types:
        moleculestm_data = smiles2graph(instruction_data['smiles'])
        graph_data['moleculestm'] = moleculestm_data
    
    # Create processed sample
    processed = {
        'cid': cid,
        'smiles': instruction_data['smiles'],
        'selfies': instruction_data.get('selfies', ''),
        'system': instruction_data['system'],
        'conversations': instruction_data['conversations'],
        'category': instruction_data.get('category', ''),
        'graph_data': graph_data,  # Keep as nested dict like finetuning data
    }
    
    # Add brics_gids and entropy_gids if they exist
    if 'brics_gids' in mol_data:
        processed['brics_gids'] = mol_data['brics_gids']
    if 'entropy_gids' in mol_data:
        processed['entropy_gids'] = mol_data['entropy_gids']
    
    return processed


def preprocess_moleculeqa_dataset(
    mol_json,
    instruction_json,
    output_jsonl,
    encoder_types=['unimol', 'moleculestm'],
    max_atoms=512
):
    """
    Main preprocessing function for MoleculeQA dataset.
    
    Args:
        mol_json: Path to molecule JSON file
        instruction_json: Path to instruction JSON file
        output_jsonl: Path to output JSONL file
        encoder_types: List of encoder types to process
        max_atoms: Maximum number of atoms for UniMol
    """
    logger.info("=" * 80)
    logger.info("MoleculeQA Data Preprocessing")
    logger.info("=" * 80)
    logger.info(f"Molecule JSON: {mol_json}")
    logger.info(f"Instruction JSON: {instruction_json}")
    logger.info(f"Output JSONL: {output_jsonl}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info(f"Max atoms: {max_atoms}")
    
    # Load UniMol dictionary if needed
    unimol_dictionary = None
    if 'unimol' in encoder_types:
        from huggingface_hub import hf_hub_download
        from utils.unicore import Dictionary
        
        logger.info("Loading UniMol dictionary from HuggingFace...")
        unimol_dictionary_path = hf_hub_download(
            repo_id='dptech/Uni-Mol-Models',
            filename='mol.dict.txt',
        )
        unimol_dictionary = Dictionary.load(unimol_dictionary_path)
        unimol_dictionary.add_symbol("[MASK]", is_special=True)
        logger.info(f"UniMol dictionary loaded with {len(unimol_dictionary)} tokens")
    
    # Load molecule data
    logger.info(f"Loading molecule data from {mol_json}")
    with open(mol_json, 'r') as f:
        mol_data_list = json.load(f)
    logger.info(f"Loaded {len(mol_data_list)} molecules")
    
    # Load instruction data
    logger.info(f"Loading instruction data from {instruction_json}")
    with open(instruction_json, 'r') as f:
        instruction_data_list = json.load(f)
    logger.info(f"Loaded {len(instruction_data_list)} instruction samples")
    
    # Verify data alignment
    if len(mol_data_list) != len(instruction_data_list):
        raise ValueError(
            f"Molecule data and instruction data must have the same length. "
            f"Got {len(mol_data_list)} molecules and {len(instruction_data_list)} instructions."
        )
    logger.info(f"✓ Data alignment verified: {len(mol_data_list)} samples")
    
    # Create output directory
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process and write (using index-based alignment)
    logger.info("Processing instruction samples...")
    logger.info("Note: Using index-based alignment (instruction[i] <-> molecule[i])")
    processed_count = 0
    error_count = 0
    
    with open(output_jsonl, 'w') as f:
        for idx, (instruction_data, mol_data) in enumerate(tqdm(
            zip(instruction_data_list, mol_data_list), 
            total=len(instruction_data_list),
            desc="Processing"
        )):
            try:
                processed = process_moleculeqa_sample(
                    instruction_data,
                    mol_data,
                    unimol_dictionary,
                    encoder_types,
                    max_atoms
                )
                f.write(json.dumps(processed) + '\n')
                processed_count += 1
            except Exception as e:
                error_count += 1
                import traceback
                logger.error(f"Error processing sample at index {idx} (CID={instruction_data.get('cid', '?')}): {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                if error_count <= 3:  # Show detailed error for first 3 failures
                    logger.error(f"Sample data: {json.dumps(instruction_data, indent=2)}")
                continue
    
    logger.info("=" * 80)
    logger.info("✅ Preprocessing complete!")
    logger.info(f"   Processed: {processed_count} samples")
    logger.info(f"   Errors: {error_count} samples")
    logger.info(f"   Output: {output_jsonl}")
    
    # Print file size
    file_size_mb = os.path.getsize(output_jsonl) / (1024 * 1024)
    logger.info(f"   File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MoleculeQA instruction data"
    )
    parser.add_argument(
        '--mol_json',
        type=str,
        required=True,
        help='Path to molecule JSON file'
    )
    parser.add_argument(
        '--instruction_json',
        type=str,
        required=True,
        help='Path to instruction JSON file'
    )
    parser.add_argument(
        '--output_jsonl',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--encoder_types',
        nargs='+',
        default=['unimol', 'moleculestm'],
        choices=['unimol', 'moleculestm'],
        help='Encoder types to process'
    )
    parser.add_argument(
        '--max_atoms',
        type=int,
        default=512,
        help='Maximum number of atoms for UniMol'
    )
    
    args = parser.parse_args()
    
    preprocess_moleculeqa_dataset(
        mol_json=args.mol_json,
        instruction_json=args.instruction_json,
        output_jsonl=args.output_jsonl,
        encoder_types=args.encoder_types,
        max_atoms=args.max_atoms
    )


if __name__ == '__main__':
    main()

