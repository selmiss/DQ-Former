#!/usr/bin/env python3
"""
Script to extract 200 random samples from a JSONL file.
"""
import json
import random
import argparse
from pathlib import Path


def count_lines(file_path):
    """Count the total number of lines in a file."""
    print(f"Counting lines in {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    print(f"Total lines: {count}")
    return count


def extract_random_samples(input_file, output_file, num_samples=200, seed=42):
    """
    Extract random samples from a JSONL file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        num_samples: Number of random samples to extract
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Count total lines
    total_lines = count_lines(input_file)
    
    if num_samples > total_lines:
        print(f"Warning: Requested {num_samples} samples but file only has {total_lines} lines.")
        print(f"Extracting all {total_lines} lines instead.")
        num_samples = total_lines
    
    # Generate random line indices (0-based)
    selected_indices = sorted(random.sample(range(total_lines), num_samples))
    print(f"Selected {len(selected_indices)} random indices")
    
    # Extract selected lines
    print(f"Extracting samples from {input_file}...")
    selected_lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx in selected_indices:
                selected_lines.append(line.strip())
                if len(selected_lines) % 50 == 0:
                    print(f"  Extracted {len(selected_lines)}/{num_samples} samples...")
    
    # Write to output file
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in selected_lines:
            f.write(line + '\n')
    
    print(f"Successfully extracted {len(selected_lines)} samples to {output_file}")
    
    # Validate output
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                json.loads(line)  # Validate JSON format
        print(f"Output file validated: all {len(selected_lines)} lines are valid JSON")
    except json.JSONDecodeError as e:
        print(f"Warning: JSON validation error at line {i+1}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random samples from a JSONL file")
    parser.add_argument(
        "--input",
        type=str,
        default="../../data/finetune/detailed_structural_descriptions-preprocessed.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hallu_fg.jsonl",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of random samples to extract"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    extract_random_samples(
        input_file=str(input_path),
        output_file=str(output_path),
        num_samples=args.num_samples,
        seed=args.seed
    )

