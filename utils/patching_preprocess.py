#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process a JSON list of dicts, each with:
  {"smiles": "<SMILES with implicit H (no explicit hydrogens)>"}

For each record, add:
  "brics_ids": [int, ...]  # length == number of heavy atoms (RDKit order)

Behavior:
- Uses RDKit BRICS to break molecules.
- Fragment IDs follow the **node (atom) order** (no re-sorting).
- If *anything* goes wrong (invalid SMILES, BRICS failure, etc.), it **falls back**
  to fixed-length segments (default length = 3) and STILL writes brics_ids.
- Shows a tqdm progress bar.
"""

import json
import argparse
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds


# ---------------------------
# Core: your BRICS splitter
# ---------------------------

def bond_break(mol):
    """
    Split molecule with BRICS, return cluster_idx as a LongTensor
    where each row is the list of atom indices for one fragment.
    (We keep your original approach and minimal tweaks.)
    """
    results = np.array(sorted(list(FindBRICSBonds(mol))), dtype=np.int64)

    if results.size == 0:
        cluster_idx = []
        # Collect atom indices per connected component directly from original mol
        Chem.rdmolops.GetMolFrags(mol, asMols=True, frags=cluster_idx)
    else:
        bond_to_break = results[:, 0, :].tolist()
        with Chem.RWMol(mol) as rwmol:
            for i in bond_to_break:
                rwmol.RemoveBond(*i)
        rwmol = rwmol.GetMol()
        cluster_idx = []
        # Keep sanitizeFrags=False for robustness (matches your working code)
        Chem.rdmolops.GetMolFrags(
            rwmol, asMols=True, sanitizeFrags=False, frags=cluster_idx
        )

    return cluster_idx


def brics_ids_from_smiles(smiles: str) -> List[int]:
    """
    Convert BRICS fragments to a per-atom fragment-ID vector.
    IDs follow **node order** (atom index order) — no remapping/sorting.
    """
    mol = AllChem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    N = mol.GetNumAtoms()  # heavy atoms only (implicit H)
    cluster_idx = bond_break(mol)

    return cluster_idx


# ---------------------------
# Fallback: fixed-length split
# ---------------------------

def fixed_length_ids(n_atoms: int, seg_len: int = 3) -> List[int]:
    """
    Produce fragment IDs by slicing atoms into contiguous blocks of seg_len.
    Example (n=8, seg_len=3) -> [0,0,0,1,1,1,2,2]
    """
    if n_atoms <= 0:
        return []
    seg_len = max(1, int(seg_len))
    ids = []
    cur = 0
    gid = 0
    while cur < n_atoms:
        run = min(seg_len, n_atoms - cur)
        ids.extend([gid] * run)
        gid += 1
        cur += run
    return ids


# ---------------------------
# I/O pipeline
# ---------------------------

def process_file(in_path: str, out_path: str, fallback_len: int = 3) -> None:
    """
    Read JSON or JSONL file, compute brics_ids per record, write output in same format.
    Detects format based on file extension: .json vs .jsonl
    Never raises on a per-record basis: uses fixed-length fallback if anything fails.
    """
    # Detect format based on file extension
    is_jsonl = in_path.lower().endswith('.jsonl')
    unwrap = False
    
    with open(in_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            # Parse JSONL format (one JSON object per line)
            data = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        else:
            # Parse JSON format
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    # If it's not a list, still try to handle gracefully by wrapping and unwrapping.
                    data = [data]
                    unwrap = True
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format: {e}")

    out = []
    for rec in tqdm(data, desc="BRICS fragmenting"):
        # Be permissive: if smiles missing or malformed, fallback
        smiles = rec.get("smiles") if isinstance(rec, dict) else None
        rec2 = dict(rec) if isinstance(rec, dict) else {"data": rec}

        # try:
        if not smiles or not isinstance(smiles, str):
            raise ValueError("Missing or invalid 'smiles' field")

        # Try BRICS first
        ids = brics_ids_from_smiles(smiles)

        # except Exception:
        #     print(f"Error processing {smiles}")
        #     # On any error: fallback to fixed-length IDs using number of heavy atoms
        #     try:
        #         mol = AllChem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        #         n_heavy = mol.GetNumAtoms() if mol is not None else 0
        #     except Exception:
        #         n_heavy = 0
        #     ids = fixed_length_ids(n_heavy, seg_len=fallback_len)

        rec2["brics_gids"] = ids
        out.append(rec2)

    # Write output in the same format as input
    with open(out_path, "w", encoding="utf-8") as f:
        if is_jsonl:
            # Write as JSONL (one JSON object per line)
            for record in out:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        else:
            # Write as JSON
            to_write = out[0] if unwrap and len(out) == 1 else out
            json.dump(to_write, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="BRICS-based subgroup IDs with robust fallback.")
    ap.add_argument("-i", "--input", required=True, help="Path to input JSON (list of dicts with 'smiles').")
    ap.add_argument("-o", "--output", required=True, help="Path to output JSON.")
    ap.add_argument("--fallback-len", type=int, default=3, help="Fixed-length segment size on error (default: 3).")
    args = ap.parse_args()

    process_file(args.input, args.output, fallback_len=args.fallback_len)


if __name__ == "__main__":
    main()
