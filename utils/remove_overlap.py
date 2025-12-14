#!/usr/bin/env python3
"""
Remove overlapping data from test splits and save cleaned datasets.

This script:
1. Loads training SMILES from TXT files
2. Identifies overlapping SMILES in test splits only (ignores train/val splits)
3. Removes overlapping entries from test splits
4. Updates meta.json with new split indices
5. Saves cleaned data to new directory(ies)

Supports processing multiple datasets in a single run.

Example (single dataset):
  python utils/remove_overlap.py \
    --dataset data/zeroshot/ames \
    --train data/train1.txt data/train2.txt \
    --output data/zeroshot/ames_clean

Example (multiple datasets):
  python utils/remove_overlap.py \
    --dataset data/zeroshot/ames data/zeroshot/bace data/zeroshot/bbbp \
    --train data/train1.txt data/train2.txt \
    --output data/zeroshot
    # Creates: data/zeroshot/ames_clean, data/zeroshot/bace_clean, etc.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Set, Optional, Dict, Union, Any

try:
    from tqdm import tqdm
except ImportError:
    sys.stderr.write(
        "[WARNING] tqdm not installed. Install it with: pip install tqdm\n"
        "Progress bars will be disabled.\n"
    )
    # Fallback: create a dummy tqdm that does nothing
    def tqdm(iterable=None, *args, **kwargs):
        if iterable is None:
            class DummyTqdm:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return DummyTqdm()
        return iterable

try:
    from rdkit import Chem  # type: ignore
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    sys.stderr.write(
        "[ERROR] RDKit is required for canonicalization. Install it via conda:\n"
        "        conda install -c conda-forge rdkit\n"
    )
    sys.exit(1)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
    
    Returns:
        Canonicalized SMILES string, or None if invalid
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical
    except Exception:
        return None


def load_training_smiles(txt_files: List[str]) -> Set[str]:
    """Load and canonicalize SMILES from training TXT files.
    
    Args:
        txt_files: List of paths to TXT files (one SMILES per line)
    
    Returns:
        Set of canonicalized SMILES strings
    """
    training_set = set()
    
    print(f"Loading training SMILES from {len(txt_files)} file(s)...")
    for txt_file in txt_files:
        if not Path(txt_file).exists():
            sys.stderr.write(f"[WARNING] Training file not found: {txt_file}, skipping\n")
            continue
        
        file_name = Path(txt_file).name
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        initial_size = len(training_set)
        print(f"  Processing {file_name} ({len(lines)} SMILES)...")
        for line in tqdm(lines, desc=f"  Canonicalizing {file_name}", unit="SMILES", leave=False):
            smiles = line.strip()
            if smiles:
                canonical = canonicalize_smiles(smiles)
                if canonical is not None:
                    training_set.add(canonical)
        
        added_count = len(training_set) - initial_size
        print(f"  Added {added_count:,} unique SMILES from {file_name}")
    
    print(f"\nTotal unique training SMILES: {len(training_set):,}")
    return training_set


def load_jsonl_smiles(jsonl_path: str, indices: List[int], smiles_key: str = "smiles") -> Dict[int, str]:
    """Load SMILES from specific indices in a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        indices: List of line indices (0-indexed) to extract
        smiles_key: Key name for SMILES in each JSON object (default: "smiles")
    
    Returns:
        Dictionary mapping index to canonicalized SMILES (or None if invalid)
    """
    index_set = set(indices)
    max_index = max(indices) if indices else -1
    
    smiles_dict = {}
    file_name = Path(jsonl_path).name
    
    print(f"Loading SMILES from {file_name} for {len(indices)} indices...")
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, desc=f"Reading {file_name}", unit="lines", total=max_index+1 if max_index >= 0 else None)):
            if line_idx > max_index:
                break
            
            if line_idx not in index_set:
                continue
            
            line = line.strip()
            if not line:
                smiles_dict[line_idx] = None
                continue
            
            try:
                record = json.loads(line)
                if smiles_key in record:
                    smiles_value = record[smiles_key]
                    if isinstance(smiles_value, list):
                        # Handle list of SMILES: canonicalize all
                        canonicalized_list = []
                        for item in smiles_value:
                            if isinstance(item, str) and item.strip():
                                canonical = canonicalize_smiles(item.strip())
                                if canonical is not None:
                                    canonicalized_list.append(canonical)
                        # Store as a list of canonicalized SMILES
                        # Overlap checking will require ALL SMILES to overlap (matching calculate_overlap.py)
                        smiles_dict[line_idx] = canonicalized_list if canonicalized_list else None
                    elif isinstance(smiles_value, str) and smiles_value.strip():
                        canonical = canonicalize_smiles(smiles_value.strip())
                        smiles_dict[line_idx] = canonical
                    elif smiles_value is not None:
                        canonical = canonicalize_smiles(str(smiles_value))
                        smiles_dict[line_idx] = canonical
                    else:
                        smiles_dict[line_idx] = None
                else:
                    smiles_dict[line_idx] = None
            except json.JSONDecodeError:
                smiles_dict[line_idx] = None
            except Exception:
                smiles_dict[line_idx] = None
    
    return smiles_dict


def check_overlap(smiles_data: Optional[Union[str, List[str]]], training_set: Set[str]) -> bool:
    """Check if SMILES data overlaps with training set.
    
    Args:
        smiles_data: Single SMILES string, list of SMILES strings, or None
        training_set: Set of canonicalized training SMILES
    
    Returns:
        True if overlaps, False otherwise
        For list of SMILES: returns True only if ALL SMILES overlap (matching calculate_overlap.py behavior)
    """
    if smiles_data is None:
        return False
    
    if isinstance(smiles_data, list):
        # For list of SMILES, require ALL to overlap (matching calculate_overlap.py behavior)
        if len(smiles_data) == 0:
            return False
        return all(smiles in training_set for smiles in smiles_data)
    else:
        # Single SMILES string
        return smiles_data in training_set


def process_dataset(
    dataset_dir: Path,
    training_set: Set[str],
    output_dir: Path,
    smiles_key: str,
    meta_filename: str
) -> Dict[str, Any]:
    """Process a single dataset to remove overlapping test samples.
    
    Args:
        dataset_dir: Path to dataset directory containing meta.json and jsonl file
        training_set: Set of canonicalized training SMILES
        output_dir: Path to output directory for cleaned dataset
        smiles_key: Key name for SMILES in JSON objects
        meta_filename: Name of meta JSON file
    
    Returns:
        Dictionary with processing results
    """
    # Validate input directory
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    meta_path = dataset_dir / meta_filename
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    
    # Find JSONL file
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL file found in {dataset_dir}")
    if len(jsonl_files) > 1:
        sys.stderr.write(f"[WARNING] Multiple JSONL files found in {dataset_dir}, using: {jsonl_files[0]}\n")
    jsonl_path = jsonl_files[0]
    
    # Load meta.json
    print(f"\nProcessing: {dataset_dir.name}")
    print("-" * 80)
    print(f"Loading meta.json from {meta_path}...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    if "split" not in meta:
        raise ValueError(f"meta.json does not contain 'split' field: {meta_path}")
    
    splits = meta["split"]
    if "test" not in splits:
        raise ValueError(f"meta.json does not contain 'test' split: {meta_path}")
    
    test_indices = splits["test"]
    print(f"Found {len(test_indices)} test samples")
    
    # Load test SMILES
    test_smiles = load_jsonl_smiles(jsonl_path, test_indices, smiles_key)
    
    # Identify overlapping indices
    print("Identifying overlapping samples...")
    overlapping_indices = set()
    overlap_count = 0
    
    for idx in tqdm(test_indices, desc="Checking overlaps", unit="samples", leave=False):
        smiles_data = test_smiles.get(idx)
        if check_overlap(smiles_data, training_set):
            overlapping_indices.add(idx)
            overlap_count += 1
    
    print(f"Found {overlap_count:,} overlapping test samples out of {len(test_indices):,}")
    
    # Create cleaned test split
    cleaned_test_indices = [idx for idx in test_indices if idx not in overlapping_indices]
    removed_count = len(test_indices) - len(cleaned_test_indices)
    
    print(f"Removed {removed_count:,} overlapping samples")
    print(f"Remaining test samples: {len(cleaned_test_indices):,}")
    
    if removed_count == 0:
        print("No overlaps found. Skipping output.")
        return {
            "dataset": dataset_dir.name,
            "original_count": len(test_indices),
            "removed_count": 0,
            "cleaned_count": len(test_indices),
            "overlap_rate": 0.0,
            "output_dir": None
        }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update meta.json with cleaned splits
    cleaned_meta = meta.copy()
    cleaned_meta["split"] = splits.copy()
    cleaned_meta["split"]["test"] = cleaned_test_indices
    
    # Save cleaned meta.json
    output_meta_path = output_dir / meta_filename
    print(f"Saving cleaned meta.json to {output_meta_path}...")
    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_meta, f, indent=2)
    
    # Copy JSONL file (unchanged, since we only modify indices)
    output_jsonl_path = output_dir / jsonl_path.name
    print(f"Copying JSONL file to {output_jsonl_path}...")
    shutil.copy2(jsonl_path, output_jsonl_path)
    
    overlap_rate = removed_count / len(test_indices) if len(test_indices) > 0 else 0.0
    
    return {
        "dataset": dataset_dir.name,
        "original_count": len(test_indices),
        "removed_count": removed_count,
        "cleaned_count": len(cleaned_test_indices),
        "overlap_rate": overlap_rate,
        "output_dir": str(output_dir)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove overlapping data from test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        required=True,
        help="Path(s) to dataset directory(ies) containing meta.json and jsonl file(s)"
    )
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Path(s) to training TXT file(s) (one SMILES per line)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Base path to output directory. For multiple datasets, subdirectories will be created."
    )
    parser.add_argument(
        "--smiles-key",
        default="smiles",
        help="Key name for SMILES in JSON objects (default: 'smiles')"
    )
    parser.add_argument(
        "--meta-filename",
        default="meta.json",
        help="Name of meta JSON file (default: 'meta.json')"
    )
    parser.add_argument(
        "--output-suffix",
        default="_clean",
        help="Suffix to append to dataset name for output directory (default: '_clean')"
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Path to output file for summary table and detailed results (default: data/overlap/results/remove_overlap_results.txt)"
    )
    
    args = parser.parse_args()
    
    # Set default output table path if not provided
    if args.output_table is None:
        args.output_table = "data/overlap/results/remove_overlap_results.txt"
    
    # Validate training files exist
    for txt_path in args.train:
        if not Path(txt_path).exists():
            sys.stderr.write(f"[ERROR] Training file not found: {txt_path}\n")
            return 1
    
    # Validate dataset directories exist
    dataset_dirs = []
    for dataset_path in args.dataset:
        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            sys.stderr.write(f"[ERROR] Dataset directory not found: {dataset_dir}\n")
            return 1
        dataset_dirs.append(dataset_dir)
    
    # Load training SMILES (once for all datasets)
    print(f"{'='*80}")
    print("LOADING TRAINING SMILES")
    print(f"{'='*80}")
    training_set = load_training_smiles(args.train)
    
    if len(training_set) == 0:
        sys.stderr.write("[ERROR] No valid training SMILES loaded\n")
        return 1
    
    # Process each dataset
    print(f"\n{'='*80}")
    print("PROCESSING DATASETS")
    print(f"{'='*80}")
    
    results = []
    base_output_dir = Path(args.output)
    
    for dataset_dir in dataset_dirs:
        try:
            # Determine output directory
            if len(dataset_dirs) == 1:
                # Single dataset: use output path as-is
                output_dir = base_output_dir
            else:
                # Multiple datasets: create subdirectory
                dataset_name = dataset_dir.name
                output_dir = base_output_dir / f"{dataset_name}{args.output_suffix}"
            
            result = process_dataset(
                dataset_dir,
                training_set,
                output_dir,
                args.smiles_key,
                args.meta_filename
            )
            results.append(result)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to process {dataset_dir}: {e}\n")
            results.append({
                "dataset": dataset_dir.name,
                "error": str(e)
            })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Dataset':<30} {'Original':>12} {'Removed':>12} {'Cleaned':>12} {'Rate':>10}")
    print("-" * 80)
    
    for result in results:
        if "error" in result:
            print(f"{result['dataset']:<30} {'ERROR':>12}")
        elif result.get("removed_count", 0) == 0:
            print(
                f"{result['dataset']:<30} {result['original_count']:>12,} "
                f"{'0':>12} {result['cleaned_count']:>12,} {'0.00%':>10}"
            )
        else:
            print(
                f"{result['dataset']:<30} {result['original_count']:>12,} "
                f"{result['removed_count']:>12,} {result['cleaned_count']:>12,} "
                f"{result['overlap_rate']:>9.2%}"
            )
    print("=" * 80)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nDataset: {result['dataset']}")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Original test samples:  {result['original_count']:,}")
            print(f"  Overlapping samples:    {result['removed_count']:,}")
            print(f"  Cleaned test samples:   {result['cleaned_count']:,}")
            if result['original_count'] > 0:
                print(f"  Overlap rate:           {result['overlap_rate']:.4%}")
            if result.get('output_dir'):
                print(f"  Output directory:      {result['output_dir']}")
    
    print(f"\n{'='*80}")
    
    # Write results to file
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("OVERLAP REMOVAL RESULTS")
    output_lines.append("=" * 80)
    output_lines.append(f"\nTraining sets: {len(args.train)} file(s)")
    for train_file in args.train:
        output_lines.append(f"  - {train_file}")
    output_lines.append(f"\nTotal unique training SMILES: {len(training_set):,}")
    
    # Summary table
    output_lines.append("\n" + "=" * 80)
    output_lines.append("SUMMARY TABLE")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Dataset':<30} {'Original':>12} {'Removed':>12} {'Cleaned':>12} {'Rate':>10}")
    output_lines.append("-" * 80)
    
    for result in results:
        if "error" in result:
            output_lines.append(f"{result['dataset']:<30} {'ERROR':>12}")
        elif result.get("removed_count", 0) == 0:
            output_lines.append(
                f"{result['dataset']:<30} {result['original_count']:>12,} "
                f"{'0':>12} {result['cleaned_count']:>12,} {'0.00%':>10}"
            )
        else:
            output_lines.append(
                f"{result['dataset']:<30} {result['original_count']:>12,} "
                f"{result['removed_count']:>12,} {result['cleaned_count']:>12,} "
                f"{result['overlap_rate']:>9.2%}"
            )
    output_lines.append("=" * 80)
    
    # Detailed results
    output_lines.append("\n" + "=" * 80)
    output_lines.append("DETAILED RESULTS")
    output_lines.append("=" * 80)
    
    for result in results:
        output_lines.append(f"\nDataset: {result['dataset']}")
        if "error" in result:
            output_lines.append(f"  ERROR: {result['error']}")
        else:
            output_lines.append(f"  Original test samples:  {result['original_count']:,}")
            output_lines.append(f"  Overlapping samples:    {result['removed_count']:,}")
            output_lines.append(f"  Cleaned test samples:   {result['cleaned_count']:,}")
            if result['original_count'] > 0:
                output_lines.append(f"  Overlap rate:           {result['overlap_rate']:.4%}")
            if result.get('output_dir'):
                output_lines.append(f"  Output directory:      {result['output_dir']}")
    
    output_lines.append("\n" + "=" * 80)
    
    output_text = "\n".join(output_lines)
    
    output_path = Path(args.output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\nResults written to: {args.output_table}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

