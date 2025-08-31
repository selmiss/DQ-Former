import os
import json
import argparse
from typing import Dict, List

import sys
from tqdm import tqdm

# Ensure project root is on path to import generate_conformer
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.generate_from_csv import generate_conformer  # noqa: E402
from utils.patching_preprocess import brics_ids_from_smiles
from trainer.entropy_model.entropy_process import group_ids_by_entropy

try:
    import selfies as sf  # type: ignore
except Exception:
    sf = None  # type: ignore


def normalize_split(name: str) -> str:
    key = str(name).strip().lower()
    if key in ("val", "valid", "validation"):
        return "valid"
    if key in ("train",):
        return "train"
    return "test"


def selfies_to_smiles(selfies_str: str) -> str:
    if sf is None:
        raise ImportError("selfies is required to decode SELFIES → SMILES.")
    smiles = sf.decoder(selfies_str)
    if smiles is None or len(str(smiles).strip()) == 0:
        raise ValueError("Failed to decode SELFIES to SMILES")
    return smiles


def build_records(items: List[Dict]) -> Dict[str, List[Dict]]:
    splits: Dict[str, List[Dict]] = {"train": [], "valid": [], "test": []}
    cid = 0
    sampled_items, train_items = [], []
    for item in tqdm(items, desc="Sampling items"):
        meta = item.get("metadata", {}) or {}
        split = normalize_split(meta.get("split", "train"))
        if item['metadata']['split'] == 'test' or split == 'valid':
            sampled_items.append(item)
        if item['metadata']['split'] == 'train':
            train_items.append(item)
    # Sample 3000 items from train set if there are more than 3000 items
    if len(train_items) > 3000:
        import random
        random.seed(42)  # For reproducibility
        sampled_train = random.sample(train_items, len(train_items)//4)
    else:
        sampled_train = train_items
    
    # Add sampled train items to sampled_items list
    sampled_items.extend(sampled_train)

    for item in tqdm(sampled_items, desc="Processing molecules"):

        instruction = str(item.get("instruction", "")).strip()
        selfies_str = str(item.get("input", "")).strip()
        answer = str(item.get("output", "")).strip()
        meta = item.get("metadata", {}) or {}
        split = normalize_split(meta.get("split", "train"))
        

        # SELFIES → SMILES
        try:
            smiles = selfies_to_smiles(selfies_str)
        except Exception:
            continue

        atoms, coords = generate_conformer(smiles)
        brics_ids = brics_ids_from_smiles(smiles)
        entropy_ids, _ = group_ids_by_entropy([smiles], ckpt_dir="checkpoints/entropy_model/checkpoint-4929", vocab_path="trainer/entropy_model/vocab.txt")
        if atoms is None or coords is None or brics_ids is None or entropy_ids is None:
            print(f"generate_conformer failed for {smiles}, skip it")
            continue

        record_text = {
            "system": "",
            "conversations": [
                {"user": f"{instruction}\nMolecule <mol>.", "assistant": answer}
            ],
            "cid": cid,
            "smiles": smiles,
        }

        record_mol = {
            "atoms": atoms,
            "coordinates": [coords.tolist()],
            "brics_gids": brics_ids[0],
            "entropy_gids": entropy_ids[0],
            "cid": cid,
            "smiles": smiles,
        }

        splits[split].append({"text": record_text, "mol": record_mol})
        cid += 1

    return splits


def write_outputs(splits: Dict[str, List[Dict]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for split_name, bundle in splits.items():
        text = [b["text"] for b in bundle]
        mol = [b["mol"] for b in bundle]
        with open(os.path.join(out_dir, f"{split_name}.json"), "w", encoding="utf-8") as f:
            json.dump(text, f, ensure_ascii=False)
        with open(os.path.join(out_dir, f"{split_name}_mol.json"), "w", encoding="utf-8") as f:
            json.dump(mol, f, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess mol_gen instructions JSON → split JSONs")
    parser.add_argument(
        "--input_json",
        default=os.path.join(
            "data", "Molecule-oriented_Instructions", "molecular_description_generation.json"
        ),
        help="Path to the input JSON list file",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("data", "mol_en", "mol_gen_quarter"),
        help="Directory to write split outputs",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of dicts")

    splits = build_records(items)
    write_outputs(splits, args.output_dir)


if __name__ == "__main__":
    main()


