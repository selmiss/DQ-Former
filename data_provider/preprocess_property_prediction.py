#!/usr/bin/env python3
"""
Preprocess property prediction datasets.

This script handles property prediction datasets that have a specific format:
  - "cid": compound ID
  - "smiles": SMILES string (single molecule)
  - "conversations": [{"user": "...<mol>...", "assistant": "property value"}]
  - "system": system prompt

The user message already contains <mol> placeholder. We need to generate:
  - graph_data (UniMol + MoleculeSTM)
  - brics_gids
  - entropy_gids

Usage:
    python preprocess_property_prediction.py \
        --input_json_train data/Molecule-oriented_Instructions/property_prediction_full/train.json \
        --input_json_test data/Molecule-oriented_Instructions/property_prediction_full/test.json \
        --output_dir data/mol_instructions_processed \
        --task_name property_prediction_full
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from tqdm import tqdm

# Set thread limits
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

from data_provider.generate_from_csv import generate_conformer
from utils.patching_preprocess import brics_ids_from_smiles
from runner.entropy_model.entropy_process import group_ids_by_entropy
from data_provider.mol_dataset import smiles2graph

# Global UniMol dictionary (loaded once)
_UNIMOL_DICTIONARY = None


def get_unimol_dictionary():
    """Load and cache the UniMol dictionary from HuggingFace."""
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


def generate_graph_data_for_molecule(smiles: str, unimol_dictionary) -> Optional[Dict]:
    """Generate graph data (UniMol + MoleculeSTM) for a single molecule."""
    try:
        # Generate 3D conformer
        atoms, coords = generate_conformer(smiles)
        
        # Generate 2D graph for MoleculeSTM
        graph_2d = smiles2graph(smiles)
        
        # Generate UniMol data
        from data_provider.preprocess.preprocess_moleculeqa_data import get_unimol_data
        unimol_data = get_unimol_data(atoms, coords, unimol_dictionary)
        
        # Convert to serializable format
        unimol_serializable = {}
        for key, value in unimol_data.items():
            if hasattr(value, 'tolist'):
                unimol_serializable[key] = value.tolist()
            else:
                unimol_serializable[key] = value
        
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


def process_single_item(item: Dict, unimol_dictionary, skip_failures: bool = True) -> Optional[Dict]:
    """
    Process a single property prediction item.
    
    Args:
        item: Original data item with cid, smiles, conversations, system
        unimol_dictionary: Pre-loaded UniMol dictionary
        skip_failures: If True, return None on failure; if False, raise error
    
    Returns:
        Processed record with graph_data, brics_gids, entropy_gids added
    """
    cid = item.get("cid")
    smiles = item.get("smiles", "").strip()
    conversations = item.get("conversations", [])
    system = item.get("system", "")
    
    if not smiles or not conversations:
        if skip_failures:
            return None
        raise ValueError(f"Missing smiles or conversations for cid {cid}")
    
    try:
        # Generate graph data
        graph_data = generate_graph_data_for_molecule(smiles, unimol_dictionary)
        if graph_data is None:
            if skip_failures:
                return None
            raise ValueError(f"Failed to generate graph data for cid {cid}")
        
        # Generate BRICS IDs
        brics_ids = brics_ids_from_smiles(smiles)
        if brics_ids is None:
            if skip_failures:
                return None
            raise ValueError(f"Failed to generate BRICS IDs for cid {cid}")
        
        # Generate entropy IDs
        entropy_result, _ = group_ids_by_entropy(
            [smiles],
            ckpt_dir="checkpoints/entropy_model",
            vocab_path="runner/entropy_model/vocab.txt"
        )
        if entropy_result is None or len(entropy_result) == 0:
            if skip_failures:
                return None
            raise ValueError(f"Failed to generate entropy IDs for cid {cid}")
        entropy_ids = entropy_result[0]
        
        # Create enhanced record
        record = {
            "cid": str(cid),
            "smiles": smiles,
            "system": system,
            "conversations": conversations,
            "category": "property_prediction",
            "graph_data": graph_data,
            "brics_gids": brics_ids,
            "entropy_gids": entropy_ids,
        }
        
        return record
    
    except Exception as e:
        if skip_failures:
            print(f"  Error processing cid {cid}: {e}")
            return None
        raise


def process_dataset(
    input_json: str,
    task_name: str,
    max_samples: Optional[int] = None,
    skip_failures: bool = True
) -> List[Dict]:
    """Process a property prediction dataset file."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_json)}")
    print(f"Task: {task_name}")
    print(f"{'='*60}\n")
    
    # Load data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total items: {len(data)}")
    
    # Sample if requested
    if max_samples and len(data) > max_samples:
        import random
        random.seed(42)
        data = random.sample(data, max_samples)
        print(f"Sampled {max_samples} items")
    
    # Load UniMol dictionary once
    unimol_dictionary = get_unimol_dictionary()
    
    # Process items
    print("\nProcessing items...")
    processed_records = []
    failed_count = 0
    
    with tqdm(data, desc="  Processing") as pbar:
        for item in pbar:
            record = process_single_item(item, unimol_dictionary, skip_failures)
            if record:
                processed_records.append(record)
            else:
                failed_count += 1
            pbar.set_postfix({"failed": failed_count})
    
    print(f"  {len(processed_records)} successful, {failed_count} failed")
    
    return processed_records


def write_jsonl(records: List[Dict], output_path: str):
    """Write records to JSONL file."""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Preprocess property prediction datasets')
    parser.add_argument('--input_json_train', type=str, required=False, help='Path to train JSON')
    parser.add_argument('--input_json_test', type=str, required=False, help='Path to test JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--task_name', type=str, required=True, help='Task name')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Max train samples')
    parser.add_argument('--skip_failures', action='store_true', default=True, help='Skip failures')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Property Prediction Preprocessor")
    print("="*60)
    print(f"Output: {args.output_dir}")
    print(f"Max train samples: {args.max_train_samples or 'all'}")
    print(f"Skip failures: {args.skip_failures}")
    print(f"Thread limit: {os.environ.get('RDKIT_NUM_THREADS', 'unset')} cores")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process train set
    if args.input_json_train:
        train_records = process_dataset(
            args.input_json_train,
            args.task_name,
            args.max_train_samples,
            args.skip_failures
        )
        train_path = os.path.join(args.output_dir, "train.jsonl")
        write_jsonl(train_records, train_path)
        print(f"\n✅ Train: {len(train_records)} records -> {train_path}")
    
    # Process test set
    if args.input_json_test:
        test_records = process_dataset(
            args.input_json_test,
            args.task_name,
            None,  # Process all test samples
            args.skip_failures
        )
        test_path = os.path.join(args.output_dir, "test.jsonl")
        write_jsonl(test_records, test_path)
        print(f"✅ Test: {len(test_records)} records -> {test_path}")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

