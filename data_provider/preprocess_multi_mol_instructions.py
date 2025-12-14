#!/usr/bin/env python3
"""
Preprocess Molecule-oriented Instructions datasets to mol_qa-like JSONL format.

This script handles datasets with multiple molecules in the input (e.g., retrosynthesis,
forward reaction prediction). It processes each molecule in the INPUT to generate graph data,
while keeping the OUTPUT as SELFIES strings.

Key differences from precess_mol_instructions.py:
- Supports MULTIPLE molecules per input (split by '.')
- graph_data is a LIST of dicts (one per input molecule)
- Only processes INPUT molecules, OUTPUT remains as SELFIES
- Outputs JSONL format (one JSON object per line)

Usage:
    python preprocess_multi_mol_instructions.py \
        --input_json data/Molecule-oriented_Instructions/retrosynthesis.json \
        --output_dir data/mol_instructions_processed \
        --task_name retrosynthesis
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm

# Set thread limits to avoid overwhelming shared servers
# Can be overridden by setting environment variables before running
if 'RDKIT_NUM_THREADS' not in os.environ:
    os.environ['RDKIT_NUM_THREADS'] = '16'
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '16'
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = '16'
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = '16'

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import exact functions from precess_mol_instructions.py
from data_provider.precess_mol_instructions import (
    selfies_to_smiles,
    normalize_split
)
from data_provider.generate_from_csv import generate_conformer
from utils.patching_preprocess import brics_ids_from_smiles
from runner.entropy_model.entropy_process import group_ids_by_entropy

# For moleculestm graph generation
from data_provider.mol_dataset import smiles2graph

# Global UniMol dictionary (loaded once)
_UNIMOL_DICTIONARY = None


def get_unimol_dictionary():
    """
    Load and cache the UniMol dictionary from HuggingFace.
    This is called once and cached globally for efficiency.
    """
    global _UNIMOL_DICTIONARY
    
    if _UNIMOL_DICTIONARY is None:
        from huggingface_hub import hf_hub_download
        from utils.unicore import Dictionary
        
        print("Loading UniMol dictionary from HuggingFace...")
        unimol_dictionary_path = hf_hub_download(
            repo_id='dptech/Uni-Mol-Models',
            filename='mol.dict.txt',
        )
        _UNIMOL_DICTIONARY = Dictionary.load(unimol_dictionary_path)
        _UNIMOL_DICTIONARY.add_symbol("[MASK]", is_special=True)
        print(f"✅ Loaded UniMol dictionary with {len(_UNIMOL_DICTIONARY)} symbols")
    
    return _UNIMOL_DICTIONARY


def split_selfies_molecules(selfies_string: str) -> List[str]:
    """
    Split a SELFIES string containing multiple molecules separated by '.'.
    
    Args:
        selfies_string: SELFIES string like "[C][O].[N][H]"
    
    Returns:
        List of individual SELFIES strings
    """
    # Split by '.' but need to be careful - '.' is also used in SELFIES tokens
    # In the datasets, molecules are separated by '.' that's NOT inside brackets
    molecules = []
    current_mol = []
    i = 0
    
    while i < len(selfies_string):
        if selfies_string[i] == '[':
            # Find closing bracket
            j = i + 1
            while j < len(selfies_string) and selfies_string[j] != ']':
                j += 1
            # Add token including brackets
            current_mol.append(selfies_string[i:j+1])
            i = j + 1
        elif selfies_string[i] == '.':
            # This is a molecule separator
            if current_mol:
                molecules.append(''.join(current_mol))
                current_mol = []
            i += 1
        else:
            # Unexpected character (shouldn't happen in valid SELFIES)
            i += 1
    
    # Add last molecule
    if current_mol:
        molecules.append(''.join(current_mol))
    
    return molecules


def generate_graph_data_for_molecule(smiles: str, unimol_dictionary) -> Optional[Dict]:
    """
    Generate graph data (unimol + moleculestm) for a single molecule.
    
    Args:
        smiles: SMILES string
        unimol_dictionary: Pre-loaded UniMol dictionary
    
    Returns:
        Dict with 'unimol' and 'moleculestm' keys, or None if failed
    """
    try:
        # Generate 3D conformer for unimol
        atoms, coords = generate_conformer(smiles)
        if atoms is None or coords is None:
            return None
        
        # Generate 2D graph for moleculestm
        graph_2d = smiles2graph(smiles)
        if graph_2d is None:
            return None
        
        # Process unimol data
        from data_provider.preprocess.preprocess_moleculeqa_data import get_unimol_data
        unimol_data = get_unimol_data(atoms, coords, unimol_dictionary)
        
        # Convert numpy arrays to lists for JSON serialization
        unimol_serializable = {}
        for key, value in unimol_data.items():
            if hasattr(value, 'tolist'):
                unimol_serializable[key] = value.tolist()
            else:
                unimol_serializable[key] = value
        
        # Process moleculestm data (convert to serializable format)
        moleculestm_data = {
            'node_feat': graph_2d['node_feat'].tolist(),
            'edge_index': graph_2d['edge_index'].tolist(),
            'edge_feat': graph_2d['edge_feat'].tolist(),
        }
        
        return {
            'unimol': unimol_serializable,
            'moleculestm': moleculestm_data
        }
    
    except Exception as e:
        print(f"  Error generating graph data: {e}")
        return None


def process_single_item(item: Dict, cid: int, task_name: str, unimol_dictionary) -> Optional[Dict]:
    """
    Process a single instruction item with potentially multiple input molecules.
    
    Args:
        item: Original data item with instruction, input, output, metadata
        cid: Unique compound ID
        task_name: Name of the task (e.g., 'retrosynthesis')
        unimol_dictionary: Pre-loaded UniMol dictionary
    
    Returns:
        Processed record in mol_qa-like format, or None if processing failed
    """
    instruction = str(item.get("instruction", "")).strip()
    input_selfies = str(item.get("input", "")).strip()
    output_selfies = str(item.get("output", "")).strip()
    metadata = item.get("metadata", {}) or {}
    split = normalize_split(metadata.get("split", "train"))
    
    # Split input SELFIES into individual molecules
    input_molecules_selfies = split_selfies_molecules(input_selfies)
    
    if not input_molecules_selfies:
        print(f"  Warning: No molecules found in input for cid {cid}")
        return None
    
    # Process each input molecule
    smiles_list = []
    graph_data_list = []
    brics_gids_list = []
    entropy_gids_list = []
    
    for idx, mol_selfies in enumerate(input_molecules_selfies):
        try:
            # Convert SELFIES to SMILES
            smiles = selfies_to_smiles(mol_selfies)
            smiles_list.append(smiles)
            
            # Generate graph data
            graph_data = generate_graph_data_for_molecule(smiles, unimol_dictionary)
            if graph_data is None:
                print(f"  Failed to generate graph data for molecule {idx+1}/{len(input_molecules_selfies)} in cid {cid}")
                return None
            graph_data_list.append(graph_data)
            
            # Generate BRICS IDs
            brics_ids = brics_ids_from_smiles(smiles)
            if brics_ids is None:
                print(f"  Failed to generate BRICS IDs for molecule {idx+1} in cid {cid}")
                return None
            brics_gids_list.append(brics_ids)
            
            # Generate entropy IDs
            entropy_result, _ = group_ids_by_entropy(
                [smiles],
                ckpt_dir="checkpoints/entropy_model",
                vocab_path="runner/entropy_model/vocab.txt"
            )
            if entropy_result is None or len(entropy_result) == 0:
                print(f"  Failed to generate entropy IDs for molecule {idx+1} in cid {cid}")
                return None
            entropy_gids_list.append(entropy_result[0])
            
        except Exception as e:
            print(f"  Error processing molecule {idx+1}/{len(input_molecules_selfies)} in cid {cid}: {e}")
            return None
    
    # Create user prompt with <mol> placeholders
    # Replace each molecule with <mol>
    user_prompt = instruction + "\n"
    for idx in range(len(input_molecules_selfies)):
        if idx == 0:
            user_prompt += f"Molecule {idx+1}: <mol>\n"
        else:
            user_prompt += f"Molecule {idx+1}: <mol>\n"
    user_prompt = user_prompt.strip()
    
    # Create record in mol_qa-like format
    record = {
        "cid": str(cid),
        "smiles": smiles_list,  # List of SMILES for input molecules
        "selfies": input_molecules_selfies,  # List of SELFIES for input molecules
        "system": "",  # Can be customized per task if needed
        "conversations": [
            {
                "user": user_prompt,
                "assistant": output_selfies  # Keep output as SELFIES
            }
        ],
        "category": task_name,
        "graph_data": graph_data_list,  # List of graph data dicts
        "brics_gids": brics_gids_list,  # List of BRICS IDs
        "entropy_gids": entropy_gids_list,  # List of entropy IDs
    }
    
    return record


def process_dataset(
    input_json: str,
    task_name: str,
    max_train_samples: Optional[int] = None,
    skip_failures: bool = True
) -> Dict[str, List[Dict]]:
    """
    Process a full dataset JSON file.
    
    Args:
        input_json: Path to input JSON file
        task_name: Name of the task
        max_train_samples: Maximum number of training samples (None = all)
        skip_failures: If True, skip failed items; if False, raise error
    
    Returns:
        Dict mapping split names to lists of processed records
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_json)}")
    print(f"Task: {task_name}")
    print(f"{'='*60}\n")
    
    # Load input data
    with open(input_json, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of dicts")
    
    print(f"Total items in dataset: {len(items)}")
    
    # Organize by split
    splits_input = defaultdict(list)
    for item in items:
        metadata = item.get("metadata", {}) or {}
        split = normalize_split(metadata.get("split", "train"))
        splits_input[split].append(item)
    
    print(f"Split distribution:")
    for split, split_items in splits_input.items():
        print(f"  {split}: {len(split_items)} items")
    
    # Sample training data if needed
    if max_train_samples and 'train' in splits_input and len(splits_input['train']) > max_train_samples:
        import random
        random.seed(42)
        splits_input['train'] = random.sample(splits_input['train'], max_train_samples)
        print(f"\nSampled {max_train_samples} training items")
    
    # Load UniMol dictionary once (for efficiency)
    print("\nLoading UniMol dictionary...")
    unimol_dictionary = get_unimol_dictionary()
    
    # Process each split
    splits_output = defaultdict(list)
    cid = 0
    
    for split in ['train', 'valid', 'test']:
        if split not in splits_input:
            continue
        
        print(f"\nProcessing {split} split...")
        items_to_process = splits_input[split]
        
        successful = 0
        failed = 0
        
        for item in tqdm(items_to_process, desc=f"  {split}"):
            record = process_single_item(item, cid, task_name, unimol_dictionary)
            
            if record is not None:
                splits_output[split].append(record)
                successful += 1
                cid += 1
            else:
                failed += 1
                if not skip_failures:
                    raise RuntimeError(f"Failed to process item {cid} in {split} split")
        
        print(f"  {split}: {successful} successful, {failed} failed")
    
    return splits_output


def write_jsonl_output(splits: Dict[str, List[Dict]], output_dir: str, task_name: str):
    """
    Write processed splits to JSONL files.
    
    Args:
        splits: Dict mapping split names to lists of records
        output_dir: Output directory
        task_name: Task name (used for subdirectory)
    """
    # Create task-specific output directory
    task_output_dir = os.path.join(output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Writing output to: {task_output_dir}")
    print(f"{'='*60}\n")
    
    for split, records in splits.items():
        output_file = os.path.join(task_output_dir, f"{split}.jsonl")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  {split:>6}: {len(records):>6} records -> {output_file}")
        print(f"           Size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess multi-molecule instruction datasets to mol_qa-like JSONL format"
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/mol_instructions_processed',
        help='Output directory for processed JSONL files'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        default=None,
        help='Task name (default: infer from filename)'
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum number of training samples to process (default: all)'
    )
    parser.add_argument(
        '--skip_failures',
        action='store_true',
        default=True,
        help='Skip items that fail processing (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")
    
    # Infer task name from filename if not provided
    if args.task_name is None:
        args.task_name = os.path.splitext(os.path.basename(args.input_json))[0]
    
    print(f"\n{'='*60}")
    print("Multi-Molecule Instructions Preprocessor")
    print(f"{'='*60}")
    print(f"Input: {args.input_json}")
    print(f"Output: {args.output_dir}/{args.task_name}/")
    print(f"Max train samples: {args.max_train_samples or 'all'}")
    print(f"Skip failures: {args.skip_failures}")
    print(f"Thread limit: {os.environ.get('RDKIT_NUM_THREADS', 'default')} cores")
    print(f"  (Set RDKIT_NUM_THREADS env var to change)")
    
    # Process dataset
    splits = process_dataset(
        args.input_json,
        args.task_name,
        max_train_samples=args.max_train_samples,
        skip_failures=args.skip_failures
    )
    
    # Write output
    write_jsonl_output(splits, args.output_dir, args.task_name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    total_records = sum(len(records) for records in splits.values())
    print(f"Total records processed: {total_records}")
    for split, records in splits.items():
        print(f"  {split}: {len(records)} records")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

