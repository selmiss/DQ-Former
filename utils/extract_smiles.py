#!/usr/bin/env python3
"""
Extract SMILES strings from one or more JSONL files.

Each line in each JSONL file should be a JSON object with a "smiles" key.
All SMILES strings are extracted and written to a single output text file,
one SMILES per line.

Example:
  python utils/extract_smiles.py \
    --input file1.jsonl file2.jsonl file3.jsonl \
    --output smiles.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

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
        "[WARNING] RDKit not installed. Canonicalization will be skipped.\n"
        "Install it via conda: conda install -c conda-forge rdkit\n"
    )


def count_lines(file_path: str) -> int:
    """Count total lines in a file efficiently."""
    try:
        with open(file_path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def canonicalize_smiles(smiles: str, skip_invalid: bool = False) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit.
    
    Args:
        smiles: Input SMILES string
        skip_invalid: If True, return None for invalid SMILES. If False, return original.
    
    Returns:
        Canonicalized SMILES string, or None if invalid and skip_invalid=True,
        or original SMILES if invalid and skip_invalid=False
    """
    if not RDKIT_AVAILABLE:
        return smiles
    
    if not isinstance(smiles, str) or not smiles.strip():
        return None if skip_invalid else smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None if skip_invalid else smiles
        canonical = Chem.MolToSmiles(mol, canonical=True)
        return canonical
    except Exception:
        return None if skip_invalid else smiles


def extract_smiles_from_jsonl(jsonl_path: str, smiles_key: str = "smiles", show_progress: bool = True) -> List[str]:
    """Extract all SMILES strings from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file
        smiles_key: Key name for SMILES in each JSON object (default: "smiles")
        show_progress: Whether to show progress bar (default: True)
    
    Returns:
        List of SMILES strings extracted from the file
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If a line contains invalid JSON
        KeyError: If a line doesn't contain the smiles_key
    """
    smiles_list = []
    
    # Count total lines for progress bar
    total_lines = count_lines(jsonl_path) if show_progress else 0
    file_name = Path(jsonl_path).name
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        iterator = enumerate(f, start=1)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=total_lines,
                desc=f"Reading {file_name}",
                unit="lines",
                leave=False
            )
        
        for line_num, line in iterator:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                record = json.loads(line)
                if smiles_key in record:
                    smiles_value = record[smiles_key]
                    if isinstance(smiles_value, str) and smiles_value.strip():
                        smiles_list.append(smiles_value.strip())
                    elif smiles_value is not None:
                        # Non-string value, convert to string
                        smiles_list.append(str(smiles_value))
                else:
                    sys.stderr.write(
                        f"[WARNING] Line {line_num} in {jsonl_path} missing key '{smiles_key}', skipping\n"
                    )
            except json.JSONDecodeError as e:
                sys.stderr.write(
                    f"[WARNING] Line {line_num} in {jsonl_path} contains invalid JSON: {e}, skipping\n"
                )
            except Exception as e:
                sys.stderr.write(
                    f"[WARNING] Line {line_num} in {jsonl_path} error: {e}, skipping\n"
                )
    
    return smiles_list


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract SMILES strings from one or more JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Path(s) to input JSONL file(s)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output text file (one SMILES per line)"
    )
    parser.add_argument(
        "--smiles-key",
        default="smiles",
        help="Key name for SMILES in JSON objects (default: 'smiles')"
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid SMILES during canonicalization (default: keep original)"
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for jsonl_path in args.input:
        if not Path(jsonl_path).exists():
            sys.stderr.write(f"[ERROR] Input file not found: {jsonl_path}\n")
            return 1
    
    # Extract SMILES from all files
    all_smiles = []
    total_files = len(args.input)
    
    # Process files with overall progress
    file_iterator = args.input
    if total_files > 1:
        file_iterator = tqdm(args.input, desc="Processing files", unit="file")
    
    for jsonl_path in file_iterator:
        try:
            smiles = extract_smiles_from_jsonl(jsonl_path, args.smiles_key, show_progress=True)
            all_smiles.extend(smiles)
            if total_files == 1:
                print(f"Extracted {len(smiles)} SMILES from {jsonl_path}")
        except FileNotFoundError:
            sys.stderr.write(f"[ERROR] File not found: {jsonl_path}\n")
            return 1
        except Exception as e:
            sys.stderr.write(f"[ERROR] Error processing {jsonl_path}: {e}\n")
            return 1
    
    # Canonicalize all SMILES
    canonicalized_smiles = []
    invalid_count = 0
    
    if RDKIT_AVAILABLE:
        print(f"\nCanonicalizing {len(all_smiles)} SMILES...")
        for smiles in tqdm(all_smiles, desc="Canonicalizing", unit="SMILES"):
            canonical = canonicalize_smiles(smiles, skip_invalid=args.skip_invalid)
            if canonical is not None:
                canonicalized_smiles.append(canonical)
            else:
                invalid_count += 1
                if not args.skip_invalid:
                    # If not skipping, keep original
                    canonicalized_smiles.append(smiles)
    else:
        canonicalized_smiles = all_smiles
        print(f"\n[WARNING] RDKit not available. Skipping canonicalization.")
    
    if invalid_count > 0:
        print(f"[WARNING] {invalid_count} invalid SMILES encountered during canonicalization")
    
    # Write all SMILES to output file
    try:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nWriting {len(canonicalized_smiles)} SMILES to output file...")
        with open(output_path, "w", encoding="utf-8") as f:
            for smiles in tqdm(canonicalized_smiles, desc="Writing output", unit="SMILES"):
                f.write(smiles + "\n")
        
        print(f"\nExtraction complete!")
        print(f"Total SMILES extracted: {len(all_smiles)}")
        if RDKIT_AVAILABLE:
            print(f"Canonicalized SMILES: {len(canonicalized_smiles)}")
            if invalid_count > 0:
                print(f"Invalid SMILES: {invalid_count}")
        print(f"Output written to: {args.output}")
        print(f"Processed {total_files} file(s)")
        
    except Exception as e:
        sys.stderr.write(f"[ERROR] Failed to write output file: {e}\n")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

