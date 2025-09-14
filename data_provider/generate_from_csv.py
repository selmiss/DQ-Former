import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    Chem = None
    AllChem = None

try:
    # OpenBabel via Pybel is optional and used as a fallback
    from openbabel import pybel  # type: ignore
except Exception:
    pybel = None  # type: ignore


@dataclass
class GenerationConfig:
    smiles_column_name: str
    target_column_name: str
    answer_map: Dict[str, str]
    dataset_name: str
    split_column_name: Optional[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    max_rows: Optional[int]
    prompts: Dict[str, Dict[str, str]]


def read_csv_rows(
    csv_path: str,
    smiles_column_name: str,
    target_column_name: str,
    max_rows: Optional[int] = None,
) -> List[Dict[str, str]]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [
            col
            for col in [smiles_column_name, target_column_name]
            if col not in reader.fieldnames
        ]
        if missing:
            raise KeyError(
                f"CSV is missing required columns: {missing}. Found columns: {reader.fieldnames}"
            )
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
        return rows


def read_csv_dir(
    csv_dir: str,
    smiles_column_name: str,
    target_column_name: str,
    max_rows: Optional[int] = None,
) -> List[Tuple[Dict[str, str], str]]:
    if not os.path.isdir(csv_dir):
        raise NotADirectoryError(f"csv_dir not found or not a directory: {csv_dir}")

    # Prefer common names
    candidates = {
        "train": ["train.csv"],
        "val": ["valid.csv", "validation.csv", "val.csv"],
        "test": ["test.csv"],
    }
    split_to_path: Dict[str, str] = {}
    for split, names in candidates.items():
        for name in names:
            path = os.path.join(csv_dir, name)
            if os.path.isfile(path):
                split_to_path[split] = path
                break

    missing = [s for s in ["train", "val", "test"] if s not in split_to_path]
    if missing:
        raise FileNotFoundError(
            f"Could not locate CSV files for splits: {missing} in directory {csv_dir}. "
            f"Looked for names: train.csv, valid.csv/validation.csv/val.csv, test.csv"
        )

    # Read in order train -> val -> test
    rows_with_split: List[Tuple[Dict[str, str], str]] = []
    total_count = 0
    for split in ["train", "val", "test"]:
        rows = read_csv_rows(
            split_to_path[split], smiles_column_name, target_column_name, None
        )
        for row in rows:
            rows_with_split.append((row, split))
            total_count += 1
            if max_rows is not None and total_count >= max_rows:
                return rows_with_split
    return rows_with_split


def get_default_prompts(dataset_name: str) -> Dict[str, Dict[str, str]]:
    system_common = (
        f"You are a molecular property prediction assistant for the {dataset_name} task. "
        "Your final answer should be formatted exactly as the provided answer mapping string."
    )
    return {
        "default": {
            "system": system_common,
            "user": "Predict the property for the molecule.\nMolecule <mol>.",
        },
        "rationale": {
            "system": system_common,
            "user": "Predict the property for the molecule and provide a brief rationale.\nMolecule <mol>.",
        },
        "task_info": {
            "system": system_common,
            "user": "Predict the property for the molecule given the task context.\nMolecule <mol>.",
        },
    }


def load_prompts_from_file(path: Optional[str], dataset_name: str) -> Dict[str, Dict[str, str]]:
    if path is None:
        return get_default_prompts(dataset_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts JSON not found: {path}")
    with open(path, "r") as f:
        prompts = json.load(f)
    required = {"default", "rationale", "task_info"}
    missing = required - set(prompts.keys())
    if missing:
        raise KeyError(f"Prompts file missing sections: {sorted(missing)}")
    for section in required:
        if not (isinstance(prompts[section], dict) and {"system", "user"} <= prompts[section].keys()):
            raise KeyError(f"Prompts['{section}'] must contain 'system' and 'user' keys")
    return prompts


def parse_answer_map(mapping: str) -> Dict[str, str]:
    try:
        parsed = json.loads(mapping)
        if not isinstance(parsed, dict):
            raise ValueError
        # Normalize keys to strings for robust matching
        normalized: Dict[str, str] = {str(k): str(v) for k, v in parsed.items()}
        return normalized
    except Exception as exc:
        raise ValueError(
            "--answer_map must be a JSON object string, e.g. '{""1"":""Positive"",""0"":""Negative""}'"
        ) from exc


def generate_conformer_with_rdkit(smiles: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    if Chem is None or AllChem is None:
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=1,
            numThreads=0,
            pruneRmsThresh=1.0,
            maxAttempts=2000,
            useRandomCoords=False,
        )
        try:
            # Use all available CPU cores for optimization
            num_threads = int(os.environ.get('NUM_WORKERS', '12'))  # 0 means use all cores
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads)
            # AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=10, numThreads=10)

        except Exception:
            print("MMFFOptimizeMoleculeConfs {} failed".format(smiles))
            pass
        mol = Chem.RemoveHs(mol)

        if mol.GetNumConformers() == 0:
            print("GetNumConformers == 0 {}".format(smiles))
            return None, None
        if num_atoms != mol.GetNumAtoms():
            print("num_atoms != mol.GetNumAtoms() {}".format(smiles))
            return None, None

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coordinates = np.array(mol.GetConformer().GetPositions(), dtype=float)
        return atoms, coordinates
    except Exception:
        return None, None


def generate_conformer_with_openbabel(smiles: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    if pybel is None:
        return None, None
    try:
        mol = pybel.readstring("smi", smiles)
        mol.make3D(forcefield="mmff94", steps=10000)
        mol.OBMol.DeleteHydrogens()
        atomic_nums = [atom.atomicnum for atom in mol.atoms]
        if len(atomic_nums) == 0:
            return None, None
        pt = Chem.GetPeriodicTable() if Chem is not None else None
        if pt is not None:
            atoms = [pt.GetElementSymbol(n) for n in atomic_nums]
        else:
            atoms = [str(int(n)) for n in atomic_nums]
        coordinates = np.array([atom.coords for atom in mol.atoms], dtype=float)
        return atoms, coordinates
    except Exception:
        return None, None


def generate_conformer(smiles: str) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
    atoms, coordinates = generate_conformer_with_rdkit(smiles)
    if atoms is None or coordinates is None:
        return None, None
        atoms, coordinates = generate_conformer_with_openbabel(smiles)
    return atoms, coordinates


def build_splits(
    num_records: int,
    split_column: Optional[List[str]] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    if split_column is not None:
        split_map: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
        for idx, tag in enumerate(split_column):
            key = str(tag).strip().lower()
            if key in ("train", "trn", "training"):
                split_map["train"].append(idx)
            elif key in ("val", "valid", "validation"):
                split_map["val"].append(idx)
            elif key in ("test", "tst"):
                split_map["test"].append(idx)
        return split_map

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    n_train = int(num_records * train_ratio)
    n_val = int(num_records * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def row_to_answer(raw_value: str, answer_map: Dict[str, str]) -> Optional[str]:
    key = str(raw_value).strip()
    if key in answer_map:
        return answer_map[key]
    # Try numeric normalization
    if key.isdigit() and key in answer_map:
        return answer_map[key]
    if key in ("1.0", "0.0") and key in answer_map:
        return answer_map[key]
    # Case-insensitive direct match
    lower_map = {k.lower(): v for k, v in answer_map.items()}
    if key.lower() in lower_map:
        return lower_map[key.lower()]
    return None


def write_meta_json(
    output_path: str,
    prompts: Dict[str, Dict[str, str]],
    split: Dict[str, List[int]],
    dataset_name: str,
) -> None:
    meta = {
        "dataset": dataset_name,
        "prompts": prompts,
        "split": split,
    }
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)


def write_data_jsonl(records: List[Dict], output_path: str) -> None:
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate(
    csv_path: Optional[str],
    csv_dir: Optional[str],
    output_dir: str,
    config: GenerationConfig,
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)

    data_records: List[Dict] = []
    split: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    if csv_dir is not None:
        rows_with_split = read_csv_dir(
            csv_dir,
            smiles_column_name=config.smiles_column_name,
            target_column_name=config.target_column_name,
            max_rows=config.max_rows,
        )
        for row, declared_split in tqdm(rows_with_split, desc="Processing molecules from CSV directory"):
            smiles = str(row[config.smiles_column_name]).strip()
            target_value = row[config.target_column_name]
            answer = row_to_answer(str(target_value), config.answer_map)
            if answer is None:
                continue
            atoms, coordinates = generate_conformer(smiles)
            if atoms is None or coordinates is None:
                continue
            record = {
                "smiles": smiles,
                "answer": answer,
                "atoms": atoms,
                "coordinates": [coordinates.tolist()],
            }
            data_records.append(record)
            split_key = declared_split if declared_split in split else "train"
            split[split_key].append(len(data_records) - 1)
    else:
        # Single CSV path flow
        assert csv_path is not None
        rows = read_csv_rows(
            csv_path,
            smiles_column_name=config.smiles_column_name,
            target_column_name=config.target_column_name,
            max_rows=config.max_rows,
        )

        split_column: Optional[List[str]] = None
        if config.split_column_name is not None and len(rows) > 0:
            sample_row = rows[0]
            if config.split_column_name in sample_row:
                split_column = [row[config.split_column_name] for row in rows]

        for row in tqdm(rows, desc="Processing molecules from single CSV"):
            smiles = str(row[config.smiles_column_name]).strip()
            target_value = row[config.target_column_name]
            answer = row_to_answer(str(target_value), config.answer_map)
            if answer is None:
                continue
            atoms, coordinates = generate_conformer(smiles)
            if atoms is None or coordinates is None:
                continue
            record = {
                "smiles": smiles,
                "answer": answer,
                "atoms": atoms,
                "coordinates": [coordinates.tolist()],
            }
            data_records.append(record)

        split = build_splits(
            num_records=len(data_records),
            split_column=split_column,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.random_seed,
        )

    meta_path = os.path.join(output_dir, f"{config.dataset_name}_meta.json")
    data_jsonl_path = os.path.join(output_dir, f"{config.dataset_name}.jsonl")

    write_meta_json(meta_path, config.prompts, split, config.dataset_name)
    write_data_jsonl(data_records, data_jsonl_path)

    return meta_path, data_jsonl_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zeroshot meta JSON and data JSONL from a CSV with SMILES and targets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Path to input CSV file")
    group.add_argument("--csv_dir", help="Directory containing train/valid(or val)/test CSV files")
    parser.add_argument("--smiles_col", required=True, help="Column name for SMILES")
    parser.add_argument("--target_col", required=True, help="Column name for target labels")
    parser.add_argument("--answer_map", required=True, help="JSON mapping from raw target values to final answer strings, e.g. '{\"1\":\"High permeability\",\"0\":\"Low-to-moderate permeability\"}'")
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs")
    parser.add_argument("--dataset_name", default="dataset", help="Name used in output file names and meta")
    parser.add_argument("--prompts_json", default=None, help="Optional path to a prompts JSON with sections: default/rationale/task_info")
    parser.add_argument("--split_col", default=None, help="Optional column name specifying split tags (train/val/test)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train ratio if auto-splitting")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio if auto-splitting")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test ratio if auto-splitting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for auto-splitting")
    parser.add_argument("--max_rows", type=int, default=None, help="Optionally limit rows for quick tests")

    args = parser.parse_args()

    prompts = load_prompts_from_file(args.prompts_json, args.dataset_name)
    answer_map = parse_answer_map(args.answer_map)

    config = GenerationConfig(
        smiles_column_name=args.smiles_col,
        target_column_name=args.target_col,
        answer_map=answer_map,
        dataset_name=args.dataset_name,
        split_column_name=args.split_col,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
        max_rows=args.max_rows,
        prompts=prompts,
    )

    meta_path, data_jsonl_path = generate(
        csv_path=args.csv,
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(f"Wrote meta JSON to: {meta_path}")
    print(f"Wrote data JSONL to: {data_jsonl_path}")


if __name__ == "__main__":
    main()


