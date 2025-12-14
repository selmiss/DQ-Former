#!/usr/bin/env python3
"""
Remove overlapping data from JSONL files (without meta.json splits).

This script:
1. Loads training SMILES from TXT files
2. Reads all lines from JSONL file(s)
3. Checks each line for SMILES overlap with training set
4. Removes overlapping lines entirely
5. Saves cleaned JSONL file(s)

Supports processing multiple JSONL files in a single run.

Example (single file):
  python utils/remove_overlap_jsonl.py \
    --jsonl data/mol_qa/test.jsonl \
    --train data/train1.txt data/train2.txt \
    --output data
    # Creates: data/mol_qa_clean/test.jsonl

Example (multiple files):
  python utils/remove_overlap_jsonl.py \
    --jsonl data/mol_qa/test.jsonl data/mol_prop/test.jsonl \
    --train data/train1.txt \
    --output data
    # Creates: data/mol_qa_clean/test.jsonl, data/mol_prop_clean/test.jsonl
"""

import argparse
import json
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


def extract_smiles_from_record(record: Dict, smiles_key: str) -> Optional[Union[str, List[str]]]:
    """Extract SMILES from a JSON record.
    
    Args:
        record: JSON record dictionary
        smiles_key: Key name for SMILES in JSON object
    
    Returns:
        Single SMILES string, list of SMILES strings, or None
    """
    if smiles_key not in record:
        return None
    
    smiles_value = record[smiles_key]
    if isinstance(smiles_value, list):
        # Handle list of SMILES: canonicalize all
        canonicalized_list = []
        for item in smiles_value:
            if isinstance(item, str) and item.strip():
                canonical = canonicalize_smiles(item.strip())
                if canonical is not None:
                    canonicalized_list.append(canonical)
        return canonicalized_list if canonicalized_list else None
    elif isinstance(smiles_value, str) and smiles_value.strip():
        canonical = canonicalize_smiles(smiles_value.strip())
        return canonical
    elif smiles_value is not None:
        canonical = canonicalize_smiles(str(smiles_value))
        return canonical
    
    return None


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


def process_jsonl_file(
    jsonl_path: Path,
    training_set: Set[str],
    output_path: Path,
    smiles_key: str
) -> Dict[str, Any]:
    """Process a single JSONL file to remove overlapping lines.
    
    Args:
        jsonl_path: Path to input JSONL file
        training_set: Set of canonicalized training SMILES
        output_path: Path to output cleaned JSONL file
        smiles_key: Key name for SMILES in JSON objects
    
    Returns:
        Dictionary with processing results
    """
    file_name = jsonl_path.name
    
    print(f"\nProcessing: {file_name}")
    print("-" * 80)
    
    # Count total lines first
    total_lines = 0
    try:
        with open(jsonl_path, "rb") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        pass
    
    print(f"Reading {file_name} ({total_lines:,} lines)...")
    
    # Read all lines and check for overlaps
    cleaned_lines = []
    overlapping_count = 0
    invalid_count = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Checking overlaps", unit="lines", total=total_lines if total_lines > 0 else None)):
            line = line.strip()
            if not line:
                # Keep empty lines
                cleaned_lines.append(line)
                continue
            
            try:
                record = json.loads(line)
                smiles_data = extract_smiles_from_record(record, smiles_key)
                
                if smiles_data is None:
                    # No SMILES found, keep the line
                    cleaned_lines.append(line)
                elif check_overlap(smiles_data, training_set):
                    # Overlapping SMILES found, skip this line
                    overlapping_count += 1
                else:
                    # No overlap, keep the line
                    cleaned_lines.append(line)
            except json.JSONDecodeError:
                # Invalid JSON, keep the line but count as invalid
                invalid_count += 1
                cleaned_lines.append(line)
            except Exception:
                # Other error, keep the line
                invalid_count += 1
                cleaned_lines.append(line)
    
    removed_count = overlapping_count
    kept_count = len(cleaned_lines) - invalid_count
    
    print(f"Total lines:           {total_lines:,}")
    print(f"Overlapping lines:     {removed_count:,}")
    print(f"Kept lines:            {kept_count:,}")
    print(f"Invalid JSON lines:    {invalid_count:,}")
    
    if removed_count == 0:
        print("No overlaps found. Skipping output.")
        return {
            "file": file_name,
            "original_count": total_lines,
            "removed_count": 0,
            "kept_count": kept_count,
            "overlap_rate": 0.0,
            "output_file": None
        }
    
    # Write cleaned JSONL file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing cleaned file to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")
    
    overlap_rate = removed_count / total_lines if total_lines > 0 else 0.0
    
    return {
        "file": file_name,
        "original_count": total_lines,
        "removed_count": removed_count,
        "kept_count": kept_count,
        "overlap_rate": overlap_rate,
        "output_file": str(output_path)
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove overlapping data from JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="Path(s) to JSONL file(s) to process"
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
        help="Base output directory. Creates *_clean subdirectories preserving input structure."
    )
    parser.add_argument(
        "--smiles-key",
        default="smiles",
        help="Key name for SMILES in JSON objects (default: 'smiles')"
    )
    parser.add_argument(
        "--output-suffix",
        default="_clean",
        help="Suffix to append to directory name for output (default: '_clean')"
    )
    parser.add_argument(
        "--output-table",
        default=None,
        help="Path to output file for summary table and detailed results (default: data/overlap/results/remove_overlap_jsonl_results.txt)"
    )
    
    args = parser.parse_args()
    
    # Set default output table path if not provided
    if args.output_table is None:
        args.output_table = "data/overlap/results/remove_overlap_jsonl_results.txt"
    
    # Validate training files exist
    for txt_path in args.train:
        if not Path(txt_path).exists():
            sys.stderr.write(f"[ERROR] Training file not found: {txt_path}\n")
            return 1
    
    # Validate JSONL files exist
    jsonl_files = []
    for jsonl_path in args.jsonl:
        jsonl_file = Path(jsonl_path)
        if not jsonl_file.exists():
            sys.stderr.write(f"[ERROR] JSONL file not found: {jsonl_file}\n")
            return 1
        jsonl_files.append(jsonl_file)
    
    # Load training SMILES (once for all files)
    print(f"{'='*80}")
    print("LOADING TRAINING SMILES")
    print(f"{'='*80}")
    training_set = load_training_smiles(args.train)
    
    if len(training_set) == 0:
        sys.stderr.write("[ERROR] No valid training SMILES loaded\n")
        return 1
    
    # Process each JSONL file
    print(f"\n{'='*80}")
    print("PROCESSING JSONL FILES")
    print(f"{'='*80}")
    
    results = []
    output_base = Path(args.output)
    
    for jsonl_file in jsonl_files:
        try:
            # Determine output path: preserve directory structure with _clean suffix
            # Example: data/mol_qa/test.jsonl -> data/mol_qa_clean/test.jsonl
            input_dir = jsonl_file.parent
            input_filename = jsonl_file.name
            dir_name = input_dir.name
            
            # Create output directory: output_base / (dir_name + suffix) / filename
            # This creates: data/mol_qa_clean/test.jsonl when output_base is "data"
            output_dir = output_base / f"{dir_name}{args.output_suffix}"
            output_path = output_dir / input_filename
            
            result = process_jsonl_file(
                jsonl_file,
                training_set,
                output_path,
                args.smiles_key
            )
            results.append(result)
        except Exception as e:
            sys.stderr.write(f"[ERROR] Failed to process {jsonl_file}: {e}\n")
            results.append({
                "file": jsonl_file.name,
                "error": str(e)
            })
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'File':<40} {'Original':>12} {'Removed':>12} {'Kept':>12} {'Rate':>10}")
    print("-" * 80)
    
    for result in results:
        if "error" in result:
            print(f"{result['file']:<40} {'ERROR':>12}")
        elif result.get("removed_count", 0) == 0:
            print(
                f"{result['file']:<40} {result['original_count']:>12,} "
                f"{'0':>12} {result['kept_count']:>12,} {'0.00%':>10}"
            )
        else:
            print(
                f"{result['file']:<40} {result['original_count']:>12,} "
                f"{result['removed_count']:>12,} {result['kept_count']:>12,} "
                f"{result['overlap_rate']:>9.2%}"
            )
    print("=" * 80)
    
    # Print detailed results
    print(f"\n{'='*80}")
    print("DETAILED RESULTS")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nFile: {result['file']}")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Original lines:        {result['original_count']:,}")
            print(f"  Overlapping lines:     {result['removed_count']:,}")
            print(f"  Kept lines:            {result['kept_count']:,}")
            if result['original_count'] > 0:
                print(f"  Overlap rate:          {result['overlap_rate']:.4%}")
            if result.get('output_file'):
                print(f"  Output file:           {result['output_file']}")
    
    print(f"\n{'='*80}")
    
    # Write results to file
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("OVERLAP REMOVAL RESULTS (JSONL)")
    output_lines.append("=" * 80)
    output_lines.append(f"\nTraining sets: {len(args.train)} file(s)")
    for train_file in args.train:
        output_lines.append(f"  - {train_file}")
    output_lines.append(f"\nTotal unique training SMILES: {len(training_set):,}")
    
    # Summary table
    output_lines.append("\n" + "=" * 80)
    output_lines.append("SUMMARY TABLE")
    output_lines.append("=" * 80)
    output_lines.append(f"{'File':<40} {'Original':>12} {'Removed':>12} {'Kept':>12} {'Rate':>10}")
    output_lines.append("-" * 80)
    
    for result in results:
        if "error" in result:
            output_lines.append(f"{result['file']:<40} {'ERROR':>12}")
        elif result.get("removed_count", 0) == 0:
            output_lines.append(
                f"{result['file']:<40} {result['original_count']:>12,} "
                f"{'0':>12} {result['kept_count']:>12,} {'0.00%':>10}"
            )
        else:
            output_lines.append(
                f"{result['file']:<40} {result['original_count']:>12,} "
                f"{result['removed_count']:>12,} {result['kept_count']:>12,} "
                f"{result['overlap_rate']:>9.2%}"
            )
    output_lines.append("=" * 80)
    
    # Detailed results
    output_lines.append("\n" + "=" * 80)
    output_lines.append("DETAILED RESULTS")
    output_lines.append("=" * 80)
    
    for result in results:
        output_lines.append(f"\nFile: {result['file']}")
        if "error" in result:
            output_lines.append(f"  ERROR: {result['error']}")
        else:
            output_lines.append(f"  Original lines:        {result['original_count']:,}")
            output_lines.append(f"  Overlapping lines:     {result['removed_count']:,}")
            output_lines.append(f"  Kept lines:            {result['kept_count']:,}")
            if result['original_count'] > 0:
                output_lines.append(f"  Overlap rate:          {result['overlap_rate']:.4%}")
            if result.get('output_file'):
                output_lines.append(f"  Output file:           {result['output_file']}")
    
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

