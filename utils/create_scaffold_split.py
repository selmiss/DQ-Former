#!/usr/bin/env python3
"""
Create scaffold-based splits for molecular QA datasets.
Each line in the input JSONL files is a dict with a 'smiles' key.
Molecules with the same scaffold are assigned to the same split.

Example:
    python utils/create_scaffold_split.py \
        --train data/mol_qa/train.jsonl \
        --test data/mol_qa/test.jsonl \
        --output data/mol_qa_scaffold \
        --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    sys.stderr.write(
        "[ERROR] RDKit is required. Install with: conda install -c conda-forge rdkit\n"
    )
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable=None, desc=""):
        return iterable if iterable is not None else []


def get_scaffold(smiles: str) -> Optional[str]:
    """Generate Murcko scaffold from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        return scaffold_smiles
    except Exception:
        return None


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file line by line."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num} in {filepath}: {e}")
                continue
    return data


def scaffold_split(
    all_data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split data based on scaffolds.
    
    Args:
        all_data: All data to split
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle scaffolds before splitting
        seed: Random seed for shuffling
    
    Returns:
        Tuple of (train_data, val_data, test_data) based on scaffold split
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Group molecules by scaffold
    scaffold_to_molecules = defaultdict(list)
    invalid_count = 0
    
    print("Generating scaffolds...")
    for idx, item in enumerate(tqdm(all_data, desc="Processing molecules")):
        smiles = item.get('smiles', '')
        if not smiles:
            invalid_count += 1
            scaffold = f"__INVALID__{invalid_count}"
        else:
            scaffold = get_scaffold(smiles)
            if scaffold is None:
                invalid_count += 1
                scaffold = f"__INVALID__{invalid_count}"
        
        scaffold_to_molecules[scaffold].append(item)
    
    # Get unique scaffolds
    scaffolds = list(scaffold_to_molecules.keys())
    print(f"Found {len(scaffolds)} unique scaffolds")
    print(f"Total molecules: {len(all_data)}")
    if invalid_count > 0:
        print(f"Warning: {invalid_count} molecules had invalid/failed scaffolds")
    
    # Sort scaffolds by number of molecules (descending) for more balanced splits
    scaffolds.sort(key=lambda s: len(scaffold_to_molecules[s]), reverse=True)
    
    # Shuffle if requested
    if shuffle:
        if seed is not None:
            random.seed(seed)
        # Shuffle while preserving the sorted order within groups
        scaffold_groups = defaultdict(list)
        for scaffold in scaffolds:
            size = len(scaffold_to_molecules[scaffold])
            scaffold_groups[size].append(scaffold)
        
        shuffled_scaffolds = []
        for size in sorted(scaffold_groups.keys(), reverse=True):
            group = scaffold_groups[size]
            random.shuffle(group)
            shuffled_scaffolds.extend(group)
        scaffolds = shuffled_scaffolds
    
    # Split scaffolds into train/val/test
    num_scaffolds = len(scaffolds)
    num_train_scaffolds = int(num_scaffolds * train_ratio)
    num_val_scaffolds = int(num_scaffolds * val_ratio)
    # Remaining scaffolds go to test
    
    train_scaffolds = set(scaffolds[:num_train_scaffolds])
    val_scaffolds = set(scaffolds[num_train_scaffolds:num_train_scaffolds + num_val_scaffolds])
    test_scaffolds = set(scaffolds[num_train_scaffolds + num_val_scaffolds:])
    
    # Assign molecules to splits based on their scaffold
    train_split = []
    val_split = []
    test_split = []
    
    for scaffold, molecules in scaffold_to_molecules.items():
        if scaffold in train_scaffolds:
            train_split.extend(molecules)
        elif scaffold in val_scaffolds:
            val_split.extend(molecules)
        elif scaffold in test_scaffolds:
            test_split.extend(molecules)
    
    print(f"\nSplit statistics:")
    print(f"Train: {len(train_split)} molecules ({len(train_scaffolds)} scaffolds)")
    print(f"Val: {len(val_split)} molecules ({len(val_scaffolds)} scaffolds)")
    print(f"Test: {len(test_split)} molecules ({len(test_scaffolds)} scaffolds)")
    
    return train_split, val_split, test_split


def write_jsonl(data: List[Dict], filepath: str):
    """Write data to JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create scaffold-based splits for molecular QA datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for scaffold-based splits"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio for training set"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio for validation set"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling scaffolds"
    )
    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Don't shuffle scaffolds before splitting (default: shuffle)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load data
    print("Loading data...")
    train_data = load_jsonl(args.train)
    test_data = load_jsonl(args.test)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(test_data)} test samples")
    
    # Combine all data
    all_data = train_data + test_data
    
    # Create scaffold-based splits
    train_split, val_split, test_split = scaffold_split(
        all_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shuffle=not args.no_shuffle,
        seed=args.seed
    )
    
    # Write output files
    print("\nWriting output files...")
    write_jsonl(train_split, os.path.join(args.output, "train.jsonl"))
    write_jsonl(val_split, os.path.join(args.output, "val.jsonl"))
    write_jsonl(test_split, os.path.join(args.output, "test.jsonl"))
    
    print(f"\nDone! Output written to {args.output}/")


if __name__ == "__main__":
    main()

