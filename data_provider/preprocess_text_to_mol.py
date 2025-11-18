#!/usr/bin/env python3
"""
Preprocess text-to-text or text-to-molecule datasets (no graph input).

This script handles datasets where the INPUT is plain text and the OUTPUT 
is either plain text or a molecule (SELFIES). No molecular graph processing is needed.

Example tasks:
  1. Description-guided molecule design (text-to-molecule):
     Input: "The molecule is a natural product found in..."  (TEXT)
     Output: "[C][C@H1][C@@H1]..."  (SELFIES)
  
  2. Open question (text-to-text):
     Instruction: "Answer the following question about molecules."
     Input: "What is the main mechanism of drug metabolism?"
     Output: "The main mechanism is enzymatic transformation by cytochrome P450..."  (TEXT)

Usage:
    python preprocess_text_to_mol.py \
        --input_json data/Biomolecular_Text_Instructions/open_question.json \
        --output_dir data/mol_instructions_processed \
        --task_name open_question
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.precess_mol_instructions import normalize_split


def process_single_item(item: Dict, cid: int, task_name: str) -> Optional[Dict]:
    """
    Process a single text-to-text or text-to-molecule item.
    
    Args:
        item: Original data item with instruction, input (text), output (text/SELFIES), metadata
        cid: Unique compound ID
        task_name: Name of the task
    
    Returns:
        Processed record in simplified format (no graph_data)
    """
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()  # Plain text input
    output_text = str(item.get("output", "")).strip()  # Text answer or SELFIES molecule
    metadata = item.get("metadata", {}) or {}
    split = normalize_split(metadata.get("split", "train"))
    
    # Validate inputs
    if not output_text:
        print(f"  Warning: Missing output for cid {cid}")
        return None
    
    # For open question tasks, input might be empty (only instruction)
    # Create user prompt by combining instruction and input
    if input_text:
        user_prompt = f"{instruction}\n{input_text}".strip()
    else:
        user_prompt = instruction.strip()
    
    if not user_prompt:
        print(f"  Warning: Missing instruction and input for cid {cid}")
        return None
    
    # Create record in simplified format (NO graph_data)
    record = {
        "cid": str(cid),
        "system": "",
        "conversations": [
            {
                "user": user_prompt,
                "assistant": output_text  # Output text (answer or SELFIES)
            }
        ],
        "category": task_name,
    }
    
    return record


def process_dataset(
    input_json: str,
    task_name: str,
    max_train_samples: Optional[int] = None,
    skip_failures: bool = True
) -> Dict[str, List[Dict]]:
    """
    Process a full text-to-text or text-to-molecule dataset.
    
    Args:
        input_json: Path to input JSON file
        task_name: Name of the task (e.g., 'open_question', 'description_guided_molecule_design')
        max_train_samples: Maximum number of training samples (None = all)
        skip_failures: If True, skip failed items; if False, raise error
    
    Returns:
        Dict mapping split names to lists of processed records
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_json)}")
    print(f"Task: {task_name}")
    print(f"{'='*60}\n")
    
    # Load input data
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total items in dataset: {len(data)}")
    
    # Group by split
    splits_data = defaultdict(list)
    for item in data:
        split = normalize_split(item.get("metadata", {}).get("split", "train"))
        splits_data[split].append(item)
    
    print("Split distribution:")
    for split, items in splits_data.items():
        print(f"  {split}: {len(items)} items")
    
    # Sample training data if requested
    if max_train_samples and "train" in splits_data:
        if len(splits_data["train"]) > max_train_samples:
            import random
            random.seed(42)
            splits_data["train"] = random.sample(splits_data["train"], max_train_samples)
            print(f"\nSampled {max_train_samples} training items")
    
    # Process each split
    results = {}
    for split, items in splits_data.items():
        print(f"\nProcessing {split} split...")
        processed_records = []
        failed_count = 0
        
        with tqdm(items, desc=f"  {split}") as pbar:
            for idx, item in enumerate(pbar):
                cid = idx  # Use index as CID
                
                try:
                    record = process_single_item(item, cid, task_name)
                    if record:
                        processed_records.append(record)
                    else:
                        failed_count += 1
                        if not skip_failures:
                            raise ValueError(f"Failed to process item {cid}")
                except Exception as e:
                    failed_count += 1
                    if not skip_failures:
                        raise
                    pbar.set_postfix({"failed": failed_count})
        
        print(f"  {split}: {len(processed_records)} successful, {failed_count} failed")
        results[split] = processed_records
    
    return results


def write_jsonl(records: List[Dict], output_path: str):
    """Write records to JSONL file (one JSON object per line)."""
    with open(output_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess text-to-text or text-to-molecule instruction datasets (no graph input)'
    )
    parser.add_argument(
        '--input_json',
        type=str,
        required=True,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed JSONL files'
    )
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        help='Name of the task (e.g., open_question, description_guided_molecule_design)'
    )
    parser.add_argument(
        '--max_train_samples',
        type=int,
        default=None,
        help='Maximum number of training samples to process (default: all)'
    )
    parser.add_argument(
        '--skip_failures',
        action='store_true',
        default=True,
        help='Skip failed items instead of raising errors'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("Text-Based Instructions Preprocessor")
    print("(No graph input - text to text/molecule)")
    print("="*60)
    print(f"Input: {args.input_json}")
    print(f"Output: {args.output_dir}")
    print(f"Max train samples: {args.max_train_samples or 'all'}")
    print(f"Skip failures: {args.skip_failures}")
    
    # Process dataset
    splits = process_dataset(
        input_json=args.input_json,
        task_name=args.task_name,
        max_train_samples=args.max_train_samples,
        skip_failures=args.skip_failures
    )
    
    # Write outputs
    print(f"\n{'='*60}")
    print(f"Writing output to: {args.output_dir}")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    total_records = 0
    for split, records in splits.items():
        output_path = os.path.join(args.output_dir, f"{split}.jsonl")
        write_jsonl(records, output_path)
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   {split}: {len(records):5d} records -> {output_path}")
        print(f"           Size: {file_size_mb:.2f} MB")
        total_records += len(records)
    
    print("="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total records processed: {total_records}")
    for split, records in splits.items():
        print(f"  {split}: {len(records)} records")
    print("="*60)


if __name__ == "__main__":
    main()

