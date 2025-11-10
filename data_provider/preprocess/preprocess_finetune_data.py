#!/usr/bin/env python3
"""
Preprocessing script for Stage 2 finetuning data.
Converts raw instruction data to preprocessed JSONL format with graph representations.

The key difference from pretraining:
- Messages/conversations are stored as-is (not tokenized)
- Tokenization happens during training (depends on LLM version)
- Graph data is preprocessed and stored

Usage:
    python preprocess_finetune_data.py \
        --input_json data/Mol-LLaMA-Instruct/instruction_data.json \
        --mol_json data/Mol-LLaMA-Instruct/pubchem-molecules.json \
        --output_jsonl data/preprocessed/finetune_data.jsonl \
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
import numpy as np
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
    """Converts SMILES string to graph representation."""
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


def create_cid_to_mol_mapping(mol_data_list):
    """Create CID to molecule data mapping."""
    cid_to_mol = {}
    for mol_data in mol_data_list:
        cid_to_mol[mol_data['cid']] = mol_data
    return cid_to_mol


def process_instruction_sample(
    instruction_data,
    cid_to_mol,
    unimol_dictionary,
    encoder_types,
    max_atoms=512
):
    """
    Process a single instruction sample.
    
    Args:
        instruction_data: Instruction data with cid, system, conversations
        cid_to_mol: Mapping from CID to molecule data
        unimol_dictionary: UniMol dictionary
        encoder_types: List of encoder types
        max_atoms: Maximum atoms for UniMol
    
    Returns:
        Dictionary with preprocessed data ready for JSONL
    """
    cid = instruction_data['cid']
    
    # Get molecule data
    if cid not in cid_to_mol:
        raise ValueError(f"CID {cid} not found in molecule data")
    
    mol_data = cid_to_mol[cid]
    
    # Build messages list (don't tokenize - that depends on LLM version)
    # Just store the messages as-is
    messages = []
    messages.append({
        "role": "system",
        "content": instruction_data['system']
    })
    
    for turn in instruction_data['conversations']:
        messages.append({
            "role": "user",
            "content": turn['user']  # Keep <mol> tokens as-is
        })
        messages.append({
            "role": "assistant",
            "content": turn['assistant']
        })
    
    # Process graph representations
    graph_data = {}
    
    if 'unimol' in encoder_types:
        atoms = mol_data['atoms']
        coordinates = mol_data['coordinates'][0] if isinstance(mol_data['coordinates'][0], list) else mol_data['coordinates']
        unimol_data = get_unimol_data(
            atoms, coordinates, unimol_dictionary, max_atoms, remove_Hs=True
        )
        graph_data['unimol'] = unimol_data
    
    if 'moleculestm' in encoder_types:
        smiles = mol_data['smiles']
        graph = smiles2graph(smiles)
        graph_data['moleculestm'] = graph
    
    # Build result
    result = {
        'cid': cid,
        'smiles': mol_data['smiles'],
        'iupac_name': mol_data['iupac_name'],
        'messages': messages,
        'graph_data': graph_data,
    }
    
    # Add optional fields
    if 'brics_gids' in mol_data:
        result['brics_gids'] = mol_data['brics_gids']
    if 'entropy_gids' in mol_data:
        result['entropy_gids'] = mol_data['entropy_gids']
    
    # Add instruction-specific metadata
    if 'type' in instruction_data:
        result['task_type'] = instruction_data['type']
    
    return result


def preprocess_finetune_dataset(
    input_json,
    mol_json,
    output_jsonl,
    encoder_types=['unimol', 'moleculestm'],
    max_atoms=512
):
    """
    Preprocess finetuning instruction dataset.
    
    Args:
        input_json: Path to instruction JSON file
        mol_json: Path to molecule JSON file
        output_jsonl: Path to output JSONL file
        encoder_types: List of encoder types
        max_atoms: Maximum atoms for UniMol
    """
    logger.info("=" * 80)
    logger.info("Preprocessing Finetuning Dataset")
    logger.info("=" * 80)
    logger.info(f"Instruction file: {input_json}")
    logger.info(f"Molecule file: {mol_json}")
    logger.info(f"Output file: {output_jsonl}")
    logger.info(f"Encoder types: {encoder_types}")
    logger.info("=" * 80)
    
    # Load UniMol dictionary if needed
    unimol_dictionary = None
    if 'unimol' in encoder_types:
        unimol_dictionary = load_unimol_dictionary()
    
    # Load molecule data
    logger.info(f"Loading molecule data from {mol_json}")
    with open(mol_json, 'r') as f:
        mol_data_list = json.load(f)
    logger.info(f"Loaded {len(mol_data_list)} molecules")
    
    # Create CID mapping
    cid_to_mol = create_cid_to_mol_mapping(mol_data_list)
    logger.info(f"Created CID mapping for {len(cid_to_mol)} molecules")
    
    # Load instruction data
    logger.info(f"Loading instruction data from {input_json}")
    with open(input_json, 'r') as f:
        instruction_data_list = json.load(f)
    logger.info(f"Loaded {len(instruction_data_list)} instruction samples")
    
    # Create output directory
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process and write
    logger.info("Processing instruction samples...")
    processed_count = 0
    error_count = 0
    
    with open(output_jsonl, 'w') as f:
        for instruction_data in tqdm(instruction_data_list, desc="Processing"):
            try:
                processed = process_instruction_sample(
                    instruction_data,
                    cid_to_mol,
                    unimol_dictionary,
                    encoder_types,
                    max_atoms
                )
                f.write(json.dumps(processed) + '\n')
                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.warning(f"Error processing sample {instruction_data.get('cid', '?')}: {e}")
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
        description="Preprocess finetuning instruction data"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to instruction JSON file'
    )
    parser.add_argument(
        '--mol_json',
        type=str,
        required=True,
        help='Path to molecule JSON file'
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
    
    preprocess_finetune_dataset(
        input_json=args.input_json,
        mol_json=args.mol_json,
        output_jsonl=args.output_jsonl,
        encoder_types=args.encoder_types,
        max_atoms=args.max_atoms
    )


if __name__ == '__main__':
    main()

