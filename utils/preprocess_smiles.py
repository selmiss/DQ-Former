#!/usr/bin/env python3
"""
Validate and preprocess a JSON dataset containing a list of dict records
with a "smiles" field. Invalid SMILES entries are removed. The script logs
the original indices of invalid entries and writes a cleaned JSON file.

Example:
  python scripts/preprocess_smiles.py \
    --input /path/to/data.json \
    --output /path/to/data.clean.json \
    --log /path/to/data.invalid.log

Requirements:
  - rdkit: Install via conda (recommended):
      conda install -c conda-forge rdkit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _safe_import_rdkit() -> Any:
    try:
        from rdkit import Chem  # type: ignore
        return Chem
    except Exception as exc:  # pragma: no cover - environment specific
        sys.stderr.write(
            "[ERROR] Failed to import RDKit. Install it via conda:\n"
            "        conda install -c conda-forge rdkit\n"
            f"Details: {exc}\n"
        )
        sys.exit(1)


@dataclass
class PreprocessResult:
    cleaned_records: List[Dict[str, Any]]
    invalid_indices: List[int]
    invalid_examples: List[Tuple[int, Any]]


def is_valid_smiles(chem_module: Any, smiles: Any) -> bool:
    """Return True if the given SMILES string parses into a molecule.

    The input may not be a string; in that case, it is considered invalid.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    try:
        mol = chem_module.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def preprocess_records(records: List[Dict[str, Any]], smiles_key: str = "smiles") -> PreprocessResult:
    chem = _safe_import_rdkit()

    cleaned: List[Dict[str, Any]] = []
    invalid_indices: List[int] = []
    invalid_examples: List[Tuple[int, Any]] = []

    for idx, record in enumerate(records):
        smiles_value = record.get(smiles_key)
        if is_valid_smiles(chem, smiles_value):
            cleaned.append(record)
        else:
            invalid_indices.append(idx)
            invalid_examples.append((idx, smiles_value))

    return PreprocessResult(cleaned_records=cleaned, invalid_indices=invalid_indices, invalid_examples=invalid_examples)


def _derive_default_paths(input_path: str) -> Tuple[str, str]:
    base_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    name, _ext = os.path.splitext(base_name)
    out_path = os.path.join(base_dir, f"{name}.clean.json")
    log_path = os.path.join(base_dir, f"{name}.invalid.log")
    return out_path, log_path


def _load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records (dicts).")
    # Best-effort validation that elements are dict-like
    for i, el in enumerate(data):
        if not isinstance(el, dict):
            raise ValueError(f"Element at index {i} is not an object/dict.")
    return data  # type: ignore[return-value]


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_log(path: str, input_path: str, result: PreprocessResult, total: int) -> None:
    timestamp = datetime.utcnow().isoformat() + "Z"
    lines: List[str] = []
    lines.append(f"timestamp: {timestamp}")
    lines.append(f"input_file: {input_path}")
    lines.append(f"total_records: {total}")
    lines.append(f"valid_records: {len(result.cleaned_records)}")
    lines.append(f"invalid_records: {len(result.invalid_indices)}")
    lines.append("invalid_indices:")
    if result.invalid_indices:
        lines.append(",".join(str(i) for i in result.invalid_indices))
    else:
        lines.append("<none>")
    lines.append("")
    lines.append("invalid_examples (index\tsmiles):")
    if result.invalid_examples:
        for idx, smi in result.invalid_examples[:100]:  # cap examples to keep logs readable
            lines.append(f"{idx}\t{smi}")
        if len(result.invalid_examples) > 100:
            lines.append(f"... and {len(result.invalid_examples) - 100} more")
    else:
        lines.append("<none>")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SMILES, filter invalid entries, and save cleaned JSON plus a log.")
    parser.add_argument("--input", required=True, help="Path to input JSON file (list of dicts with 'smiles' key)")
    parser.add_argument("--output", default=None, help="Path to write cleaned JSON (default: <input>.clean.json)")
    parser.add_argument("--log", dest="log_path", default=None, help="Path to write log file (default: <input>.invalid.log)")
    parser.add_argument("--smiles-key", default="smiles", help="JSON key containing SMILES (default: smiles)")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    input_path = os.path.abspath(args.input)
    if args.output is None or args.log_path is None:
        derived_output, derived_log = _derive_default_paths(input_path)
        output_path = os.path.abspath(args.output or derived_output)
        log_path = os.path.abspath(args.log_path or derived_log)
    else:
        output_path = os.path.abspath(args.output)
        log_path = os.path.abspath(args.log_path)

    try:
        records = _load_json_list(input_path)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to read input JSON: {exc}\n")
        return 2

    result = preprocess_records(records, smiles_key=args.smiles_key)

    # Write outputs
    try:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _write_json(output_path, result.cleaned_records)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to write cleaned JSON: {exc}\n")
        return 3

    try:
        _write_log(log_path, input_path, result, total=len(records))
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to write log file: {exc}\n")
        return 4

    # Print summary to stdout
    print("Preprocessing complete.")
    print(f"Input:    {input_path}")
    print(f"Output:   {output_path}")
    print(f"Log:      {log_path}")
    print(f"Total:    {len(records)}")
    print(f"Valid:    {len(result.cleaned_records)}")
    print(f"Invalid:  {len(result.invalid_indices)}")
    if result.invalid_indices:
        print("Invalid indices:")
        print(",".join(str(i) for i in result.invalid_indices))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))


