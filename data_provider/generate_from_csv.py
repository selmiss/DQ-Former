"""
CSV to JSONL Data Generator for Molecular Property Prediction

This module converts molecular datasets from CSV format to JSONL format with 3D conformers.

INPUT DATA FORMAT:
==================
The script accepts two types of input:

1. Single CSV file (--csv):
   - Must contain columns for SMILES strings and target labels
   - Optional split column to specify train/val/test assignments
   - Example CSV structure:
     smiles,target,split
     CCO,1,train
     c1ccccc1,0,test

2. CSV directory (--csv_dir):
   - Must contain separate files: train.csv, valid.csv (or val.csv), test.csv
   - Each file must have SMILES and target columns
   - Split is determined by filename

COMMAND LINE ARGUMENTS:
========================
Required:
  --csv or --csv_dir: Input CSV file or directory path
  --smiles_col: Name of column containing SMILES strings
  --target_col: Name of column containing target labels
  --answer_map: JSON mapping of target values to answer strings
                Example: '{"1":"High permeability","0":"Low permeability"}'
  --output_dir: Directory for output files

Optional:
  --dataset_name: Name prefix for output files (default: "dataset")
  --prompts_json: Path to custom prompts JSON file
  --split_col: Column name for train/val/test split tags
  --train_ratio: Train split ratio (default: 0.8)
  --val_ratio: Validation split ratio (default: 0.1)
  --test_ratio: Test split ratio (default: 0.1)
  --seed: Random seed for splitting (default: 42)
  --max_rows: Maximum rows to process (for testing)

OUTPUT FORMAT:
==============
1. Meta JSON file ({dataset_name}_meta.json):
   - Contains dataset name, prompts, and split indices
   
2. Data JSONL file ({dataset_name}.jsonl):
   - One JSON object per line with fields:
     * smiles: SMILES string
     * answer: Mapped target value
     * atoms: List of atom symbols
     * coordinates: List of 3D coordinates (Nx3 array)
"""

import argparse
import csv
import glob
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
    """
    Configuration for dataset generation from CSV.
    
    Attributes:
        smiles_column_name: Name of CSV column containing SMILES strings
        target_column_name: Name of CSV column containing target labels
        answer_map: Dictionary mapping raw target values to human-readable answers
                   Example: {"1": "Active", "0": "Inactive"}
        dataset_name: Name identifier for the dataset (used in output filenames)
        split_column_name: Optional CSV column specifying train/val/test split
        train_ratio: Fraction of data for training (0.0-1.0)
        val_ratio: Fraction of data for validation (0.0-1.0)
        test_ratio: Fraction of data for testing (0.0-1.0)
        random_seed: Seed for reproducible random splitting
        max_rows: Optional limit on number of rows to process
        prompts: Dictionary of prompt templates for different contexts
                Format: {"default": {"system": "...", "user": "..."},
                        "rationale": {"system": "...", "user": "..."},
                        "task_info": {"system": "...", "user": "..."}}
    """
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
    """
    Read rows from a single CSV file.
    
    Args:
        csv_path: Path to the CSV file
        smiles_column_name: Name of the column containing SMILES strings
        target_column_name: Name of the column containing target values
        max_rows: Optional limit on number of rows to read
        
    Returns:
        List of dictionaries, where each dict represents a CSV row
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If required columns are missing from the CSV
    """
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
    """
    Read rows from a directory containing separate train/val/test CSV files.
    
    Expected directory structure (any subset of splits is acceptable):
        csv_dir/
            train.csv (or train_*.csv)
            valid.csv (or validation.csv, val.csv, valid_*.csv)
            test.csv (or test_*.csv)
    
    Also supports nested 'raw' subdirectory structure:
        csv_dir/
            raw/
                train_dataset.csv
                valid_dataset.csv
                test_dataset.csv
    
    Args:
        csv_dir: Path to directory containing split CSV files
        smiles_column_name: Name of the column containing SMILES strings
        target_column_name: Name of the column containing target values
        max_rows: Optional limit on total number of rows to read across all files
        
    Returns:
        List of tuples (row_dict, split_name) where:
            - row_dict: Dictionary representing a CSV row
            - split_name: One of "train", "val", or "test"
            
    Raises:
        NotADirectoryError: If csv_dir doesn't exist or isn't a directory
        FileNotFoundError: If no valid split files are found
        
    Note:
        At least one split file (train, val, or test) must be present.
        Missing splits will be skipped gracefully.
    """
    if not os.path.isdir(csv_dir):
        raise NotADirectoryError(f"csv_dir not found or not a directory: {csv_dir}")

    # Check for 'raw' subdirectory (for molnet structure)
    raw_dir = os.path.join(csv_dir, "raw")
    search_dir = raw_dir if os.path.isdir(raw_dir) else csv_dir

    # Prefer common names (exact matches first, then patterns)
    candidates = {
        "train": ["train.csv"],
        "val": ["valid.csv", "validation.csv", "val.csv"],
        "test": ["test.csv"],
    }
    split_to_path: Dict[str, str] = {}
    
    # First try exact name matches
    for split, names in candidates.items():
        for name in names:
            path = os.path.join(search_dir, name)
            if os.path.isfile(path):
                split_to_path[split] = path
                break
    
    # If exact matches not found, try pattern matching (e.g., train_bace_1.csv)
    if len(split_to_path) < 3:
        for split in ["train", "val", "test"]:
            if split in split_to_path:
                continue
            # Look for files matching patterns like train_*.csv, valid_*.csv, test_*.csv
            patterns = []
            if split == "train":
                patterns = [os.path.join(search_dir, "train_*.csv")]
            elif split == "val":
                patterns = [
                    os.path.join(search_dir, "valid_*.csv"),
                    os.path.join(search_dir, "validation_*.csv"),
                    os.path.join(search_dir, "val_*.csv"),
                ]
            elif split == "test":
                patterns = [os.path.join(search_dir, "test_*.csv")]
            
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    # Take the first match
                    split_to_path[split] = matches[0]
                    break

    # Check if at least one split file was found
    if not split_to_path:
        raise FileNotFoundError(
            f"Could not locate any CSV files for splits (train/val/test) in directory {csv_dir} or {search_dir}. "
            f"Looked for names: train.csv, valid.csv/validation.csv/val.csv, test.csv, "
            f"and patterns: train_*.csv, valid_*.csv, test_*.csv"
        )

    # Read available splits in order train -> val -> test
    rows_with_split: List[Tuple[Dict[str, str], str]] = []
    total_count = 0
    for split in ["train", "val", "test"]:
        if split not in split_to_path:
            continue  # Skip missing splits
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
    """
    Generate default prompt templates for molecular property prediction.
    
    Args:
        dataset_name: Name of the dataset for context in prompts
        
    Returns:
        Dictionary with three prompt types:
            - "default": Basic property prediction prompt
            - "rationale": Prompt requesting prediction with explanation
            - "task_info": Prompt with task context
        Each prompt type contains "system" and "user" messages.
        The placeholder <mol> is replaced with actual molecule data during inference.
    """
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
    """
    Load custom prompt templates from a JSON file, or use defaults.
    
    Expected JSON format:
    {
        "default": {"system": "...", "user": "..."},
        "rationale": {"system": "...", "user": "..."},
        "task_info": {"system": "...", "user": "..."}
    }
    
    Args:
        path: Path to prompts JSON file, or None to use defaults
        dataset_name: Dataset name for default prompts if path is None
        
    Returns:
        Dictionary of prompt templates
        
    Raises:
        FileNotFoundError: If specified path doesn't exist
        KeyError: If required sections or keys are missing from the JSON
    """
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
    """
    Parse answer mapping from JSON string.
    
    Args:
        mapping: JSON string mapping raw target values to answer strings
                Example: '{"1":"Active","0":"Inactive"}'
                
    Returns:
        Dictionary with normalized string keys and string values
        
    Raises:
        ValueError: If mapping is not valid JSON or not a dictionary
    """
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
    """
    Generate 3D molecular conformer from SMILES string using RDKit.
    
    This function:
    1. Parses SMILES string into molecular structure
    2. Adds hydrogens for embedding
    3. Generates 3D conformer using distance geometry
    4. Optimizes geometry using MMFF force field
    5. Removes hydrogens to get heavy atoms only
    6. Extracts atom symbols and 3D coordinates
    
    Args:
        smiles: SMILES string representing the molecule
        
    Returns:
        Tuple of (atoms, coordinates):
            - atoms: List of atomic symbols (e.g., ['C', 'O', 'N'])
            - coordinates: NumPy array of shape (N, 3) with 3D coordinates
        Returns (None, None) if conformer generation fails
        
    Note:
        Uses MMFF force field with multiple threads for optimization.
        Number of threads controlled by NUM_WORKERS environment variable (default: 12).
    """
    if Chem is None or AllChem is None:
        return None, None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        
        # Use configurable number of threads (default 16 to avoid overwhelming shared servers)
        num_threads = int(os.environ.get('RDKIT_NUM_THREADS', '16'))
        
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=1,
            numThreads=num_threads,
            pruneRmsThresh=1.0,
            maxAttempts=2000,
            useRandomCoords=False,
        )
        try:
            # Use same number of threads for optimization
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
    """
    Generate 3D molecular conformer from SMILES string using OpenBabel (fallback).
    
    This function serves as a fallback when RDKit fails or is unavailable.
    
    Args:
        smiles: SMILES string representing the molecule
        
    Returns:
        Tuple of (atoms, coordinates):
            - atoms: List of atomic symbols (e.g., ['C', 'O', 'N'])
            - coordinates: NumPy array of shape (N, 3) with 3D coordinates
        Returns (None, None) if conformer generation fails
        
    Note:
        Uses MMFF94 force field with 10000 optimization steps.
        Atomic symbols are obtained from RDKit's periodic table if available.
    """
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
    """
    Generate 3D molecular conformer, trying RDKit first, then OpenBabel if needed.
    
    Args:
        smiles: SMILES string representing the molecule
        
    Returns:
        Tuple of (atoms, coordinates) from successful generator, or (None, None) if both fail
        
    Note:
        This is the main conformer generation entry point.
        Currently only tries RDKit due to early return when it fails.
    """
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
    """
    Build train/validation/test splits either from column values or by random splitting.
    
    Args:
        num_records: Total number of records to split
        split_column: Optional list of split tags (e.g., ['train', 'test', 'train', ...])
                     If provided, splits are determined by these tags.
                     Recognized tags: 'train'/'trn'/'training', 'val'/'valid'/'validation', 'test'/'tst'
        train_ratio: Fraction for training set (used only if split_column is None)
        val_ratio: Fraction for validation set (used only if split_column is None)
        test_ratio: Fraction for test set (used only if split_column is None)
        seed: Random seed for reproducible splitting (used only if split_column is None)
        
    Returns:
        Dictionary mapping split names to lists of record indices:
        {"train": [0, 2, 5, ...], "val": [1, 7, ...], "test": [3, 4, ...]}
        
    Raises:
        ValueError: If ratios don't sum to 1.0 (when doing random splitting)
    """
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
    """
    Map a raw target value to its corresponding answer string.
    
    Tries multiple matching strategies:
    1. Direct string match
    2. Numeric normalization (for digit strings)
    3. Float normalization (for "1.0", "0.0")
    4. Case-insensitive match
    
    Args:
        raw_value: Raw target value from CSV (e.g., "1", "0", "Active")
        answer_map: Dictionary mapping raw values to answer strings
        
    Returns:
        Mapped answer string if found, None otherwise
        
    Example:
        answer_map = {"1": "Active", "0": "Inactive"}
        row_to_answer("1", answer_map) -> "Active"
        row_to_answer("1.0", answer_map) -> "Active"
        row_to_answer("active", answer_map) -> None (no match)
    """
    key = str(raw_value).strip()
    if key in answer_map:
        return answer_map[key]
    # Try numeric normalization
    if key.isdigit() and key in answer_map:
        return answer_map[key]
    # Try to convert float strings (e.g., "1.0" -> "1", "0.0" -> "0")
    try:
        float_val = float(key)
        if float_val.is_integer():
            int_key = str(int(float_val))
            if int_key in answer_map:
                return answer_map[int_key]
    except (ValueError, OverflowError):
        pass
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
    """
    Write metadata JSON file containing dataset information, prompts, and split indices.
    
    Args:
        output_path: Path where the JSON file will be written
        prompts: Dictionary of prompt templates
        split: Dictionary mapping split names to record indices
        dataset_name: Name of the dataset
        
    Output format:
        {
            "dataset": "dataset_name",
            "prompts": {"default": {...}, "rationale": {...}, "task_info": {...}},
            "split": {"train": [0, 2, ...], "val": [1, ...], "test": [3, ...]}
        }
    """
    meta = {
        "dataset": dataset_name,
        "prompts": prompts,
        "split": split,
    }
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)


def write_data_jsonl(records: List[Dict], output_path: str) -> None:
    """
    Write data records to JSONL file (one JSON object per line).
    
    Args:
        records: List of record dictionaries, each containing:
                 - smiles: SMILES string
                 - answer: Mapped target value
                 - atoms: List of atom symbols
                 - coordinates: List of 3D coordinates
        output_path: Path where the JSONL file will be written
        
    Output format (one line per record):
        {"smiles": "CCO", "answer": "Active", "atoms": ["C","C","O"], "coordinates": [[[...]]]}`
    """
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def generate(
    csv_path: Optional[str],
    csv_dir: Optional[str],
    output_dir: str,
    config: GenerationConfig,
) -> Tuple[str, str]:
    """
    Main generation function: convert CSV data to JSONL format with 3D conformers.
    
    Processing steps:
    1. Read CSV data from either single file or directory
    2. For each row:
       a. Extract SMILES and target value
       b. Map target value using answer_map
       c. Generate 3D conformer (atoms + coordinates)
       d. Create record with all information
    3. Build or use provided train/val/test splits
    4. Write metadata JSON and data JSONL files
    
    Args:
        csv_path: Path to single CSV file (mutually exclusive with csv_dir)
        csv_dir: Path to directory with train/val/test CSV files (mutually exclusive with csv_path)
        output_dir: Directory where output files will be written
        config: GenerationConfig object with all configuration parameters
        
    Returns:
        Tuple of (meta_json_path, data_jsonl_path)
        
    Notes:
        - Molecules that fail conformer generation are skipped
        - Rows with unmappable target values are skipped
        - Progress is shown with tqdm progress bars
    """
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
    """
    Command-line interface for CSV to JSONL conversion.
    
    Parses command-line arguments, validates configuration, and calls generate() function.
    See module docstring at top of file for detailed usage information.
    """
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


