#!/usr/bin/env python3
"""
Calculate overlap rate between test datasets (JSONL) and training sets (TXT).

For each test dataset, calculates:
  overlap_rate = (number of overlapping SMILES) / (total SMILES in test set)

SMILES are canonicalized before comparison to ensure fair matching.

Example:
  python utils/calculate_overlap.py \
    --test data/test1.jsonl data/test2.jsonl \
    --train data/train1.txt data/train2.txt \
    --output overlap_results.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Set, Optional, Dict

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
    
    print(f"\nTotal unique training SMILES: {len(training_set)}")
    return training_set


def extract_test_smiles(jsonl_path: str, smiles_key: str = "smiles") -> List[str]:
    """Extract SMILES from a test JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        smiles_key: Key name for SMILES in each JSON object (default: "smiles")
    
    Returns:
        List of SMILES strings (may contain duplicates)
    """
    smiles_list = []
    file_name = Path(jsonl_path).name
    
    # Count total lines for progress bar
    total_lines = 0
    try:
        with open(jsonl_path, "rb") as f:
            total_lines = sum(1 for _ in f)
    except Exception:
        pass
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        iterator = enumerate(f, start=1)
        if total_lines > 0:
            iterator = tqdm(
                iterator,
                total=total_lines,
                desc=f"Reading {file_name}",
                unit="lines",
                leave=False
            )
        
        for line_num, line in iterator:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                if smiles_key in record:
                    smiles_value = record[smiles_key]
                    if isinstance(smiles_value, str) and smiles_value.strip():
                        smiles_list.append(smiles_value.strip())
                    elif smiles_value is not None:
                        smiles_list.append(str(smiles_value))
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
    
    return smiles_list


def calculate_overlap(test_smiles: List[str], training_set: Set[str]) -> Dict[str, float]:
    """Calculate overlap statistics between test and training SMILES.
    
    Args:
        test_smiles: List of SMILES from test set (may contain duplicates)
        training_set: Set of canonicalized training SMILES
    
    Returns:
        Dictionary with overlap statistics
    """
    # Canonicalize test SMILES
    print("Canonicalizing test SMILES...")
    canonicalized_test = []
    invalid_count = 0
    
    for smiles in tqdm(test_smiles, desc="Canonicalizing", unit="SMILES", leave=False):
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            canonicalized_test.append(canonical)
        else:
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"[WARNING] {invalid_count} invalid SMILES in test set")
    
    # Calculate overlap
    test_set = set(canonicalized_test)
    overlap_set = test_set & training_set
    
    total_test = len(canonicalized_test)
    unique_test = len(test_set)
    overlap_count = len(overlap_set)
    
    if total_test == 0:
        overlap_rate = 0.0
    else:
        overlap_rate = overlap_count / total_test
    
    return {
        "total_test_smiles": total_test,
        "unique_test_smiles": unique_test,
        "overlap_count": overlap_count,
        "overlap_rate": overlap_rate,
        "invalid_count": invalid_count
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate overlap rate between test datasets and training sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--test",
        nargs="+",
        required=True,
        help="Path(s) to test JSONL file(s)"
    )
    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Path(s) to training TXT file(s) (one SMILES per line)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output results file (default: print to stdout)"
    )
    parser.add_argument(
        "--smiles-key",
        default="smiles",
        help="Key name for SMILES in test JSON objects (default: 'smiles')"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for jsonl_path in args.test:
        if not Path(jsonl_path).exists():
            sys.stderr.write(f"[ERROR] Test file not found: {jsonl_path}\n")
            return 1
    
    for txt_path in args.train:
        if not Path(txt_path).exists():
            sys.stderr.write(f"[ERROR] Training file not found: {txt_path}\n")
            return 1
    
    # Load training SMILES
    training_set = load_training_smiles(args.train)
    
    if len(training_set) == 0:
        sys.stderr.write("[ERROR] No valid training SMILES loaded\n")
        return 1
    
    # Calculate overlap for each test dataset
    results = []
    
    print(f"\n{'='*80}")
    print("Calculating overlap rates...")
    print(f"{'='*80}\n")
    
    for test_file in args.test:
        print(f"\nProcessing: {test_file}")
        print("-" * 80)
        
        test_smiles = extract_test_smiles(test_file, args.smiles_key)
        stats = calculate_overlap(test_smiles, training_set)
        
        results.append({
            "test_file": test_file,
            **stats
        })
        
        print(f"\nResults for {Path(test_file).name}:")
        print(f"  Total test SMILES:        {stats['total_test_smiles']:,}")
        print(f"  Unique test SMILES:       {stats['unique_test_smiles']:,}")
        print(f"  Overlap count:            {stats['overlap_count']:,}")
        print(f"  Overlap rate:             {stats['overlap_rate']:.4%}")
        if stats['invalid_count'] > 0:
            print(f"  Invalid SMILES:           {stats['invalid_count']:,}")
    
    # Print summary table to console
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Test Dataset':<60} {'Total':>12} {'Overlap':>12} {'Rate':>10}")
    print("-" * 80)
    for result in results:
        test_name = result['test_file']
        if len(test_name) > 58:
            test_name = "..." + test_name[-(58-3):]
        print(
            f"{test_name:<60} {result['total_test_smiles']:>12,} "
            f"{result['overlap_count']:>12,} {result['overlap_rate']:>9.2%}"
        )
    print("=" * 80)
    
    # Write or print results
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("OVERLAP CALCULATION RESULTS")
    output_lines.append("=" * 80)
    output_lines.append(f"\nTraining sets: {len(args.train)} file(s)")
    for train_file in args.train:
        output_lines.append(f"  - {train_file}")
    output_lines.append(f"\nTotal unique training SMILES: {len(training_set):,}")
    
    # Summary table
    output_lines.append("\n" + "=" * 80)
    output_lines.append("SUMMARY TABLE")
    output_lines.append("=" * 80)
    output_lines.append(f"{'Test Dataset':<60} {'Total':>12} {'Overlap':>12} {'Rate':>10}")
    output_lines.append("-" * 80)
    for result in results:
        test_name = result['test_file']
        if len(test_name) > 58:
            test_name = "..." + test_name[-(58-3):]
        output_lines.append(
            f"{test_name:<60} {result['total_test_smiles']:>12,} "
            f"{result['overlap_count']:>12,} {result['overlap_rate']:>9.2%}"
        )
    output_lines.append("=" * 80)
    
    # Detailed results
    output_lines.append("\n" + "=" * 80)
    output_lines.append("DETAILED RESULTS")
    output_lines.append("=" * 80)
    
    for result in results:
        output_lines.append(f"\nTest file: {result['test_file']}")
        output_lines.append(f"  Total test SMILES:        {result['total_test_smiles']:,}")
        output_lines.append(f"  Unique test SMILES:       {result['unique_test_smiles']:,}")
        output_lines.append(f"  Overlap count:            {result['overlap_count']:,}")
        output_lines.append(f"  Overlap rate:             {result['overlap_rate']:.4%}")
        if result['invalid_count'] > 0:
            output_lines.append(f"  Invalid SMILES:           {result['invalid_count']:,}")
    
    output_lines.append("\n" + "=" * 80)
    
    output_text = "\n".join(output_lines)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"\n{'='*80}")
        print(f"Results written to: {args.output}")
    else:
        print("\n" + output_text)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

