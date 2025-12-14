#!/usr/bin/env python3
"""
Script to convert all result txt files in a directory to a CSV table.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

# Define model mappings
MODEL_MAPPING = {
    'llama2': 'Llama2',
    'llama3_1': 'Llama3.1',
    'molinstructions_2': 'Mol-Instructions-Llama2',
    'molinstructions_3.1': 'Mol-Instructions-Llama3.1',
    'molllama_2': 'Mol-Llama2',
    'molllama': 'Mol-Llama3.1',
    'edt_former_s2_large': 'EDT-Former',
    'edt_former_s2_large_budget_32_0': 'EDT-Former (32_0)',
    'edt_former_s2_large_budget_16_16': 'EDT-Former (16_16)',
    'edt_former_s2_large_budget_0_32': 'EDT-Former (0_32)',
}

# Define prompt type order and mappings
PROMPT_ORDER = [
    'default', 'default_variant_1', 'default_variant_2', 'default_variant_3',
    'rationale', 'rationale_variant_1', 'rationale_variant_2',
    'task_info', 'task_info_variant_1', 'task_info_variant_2',
    'binary_instruction', 'checklist_instruction', 'confidence_instruction'
]

# Friendly column names
COLUMN_NAMES = {
    'default': 'default-0',
    'default_variant_1': 'default-1',
    'default_variant_2': 'default-2',
    'default_variant_3': 'default-3',   
    'rationale': 'rationale-0',
    'rationale_variant_1': 'rationale-1',
    'rationale_variant_2': 'rationale-2',
    'task_info': 'task-info-0',
    'task_info_variant_1': 'task-info-1',
    'task_info_variant_2': 'task-info-2',
    'binary_instruction': 'binary',
    'checklist_instruction': 'checklist',
    'confidence_instruction': 'confidence'
}


def parse_acc_file(filepath):
    """Parse an accuracy file and extract metrics."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Extract metrics using regex
        acc_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
        non_rate_match = re.search(r'Non-rate:\s*([\d.]+)%', content)
        f1_match = re.search(r'F1 Score:\s*([\d.]+)%', content)
        precision_match = re.search(r'Precision:\s*([\d.]+)%', content)
        
        if acc_match:
            metrics['accuracy'] = float(acc_match.group(1))
        if non_rate_match:
            metrics['non_rate'] = float(non_rate_match.group(1))
        if f1_match:
            metrics['f1_score'] = float(f1_match.group(1))
        if precision_match:
            metrics['precision'] = float(precision_match.group(1))
            
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return metrics


def extract_model_and_prompt(filename):
    """Extract model name and prompt type from filename."""
    # Remove 'acc_' prefix and '.txt' suffix
    name = filename.replace('acc_', '').replace('.txt', '')
    
    # Try to match different patterns
    for model_key in sorted(MODEL_MAPPING.keys(), key=len, reverse=True):
        if name.startswith(model_key + '_'):
            prompt_type = name[len(model_key) + 1:]
            
            # Handle 'clean_' prefix in prompt type (for clean dataset variants)
            if prompt_type.startswith('clean_'):
                prompt_type = prompt_type[6:]  # Remove 'clean_' prefix
            
            return model_key, prompt_type
    
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Convert result txt files to CSV table'
    )
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing the result txt files'
    )
    parser.add_argument(
        '--output_file', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--dataset_name', '-d',
        type=str,
        default=None,
        help='Dataset name to include in the output filename (optional)'
    )
    
    args = parser.parse_args()
    
    # Directory containing the result files
    results_dir = Path(args.input_dir)
    if not results_dir.exists():
        print(f"Error: Input directory does not exist: {results_dir}")
        return
    
    # Collect all data
    data = defaultdict(dict)
    
    # Process all acc_*.txt files
    for filename in os.listdir(results_dir):
        if filename.startswith('acc_') and filename.endswith('.txt'):
            model_key, prompt_type = extract_model_and_prompt(filename)
            
            if model_key and prompt_type:
                filepath = results_dir / filename
                metrics = parse_acc_file(filepath)
                
                if metrics and 'accuracy' in metrics:
                    model_name = MODEL_MAPPING.get(model_key, model_key)
                    data[model_name][prompt_type] = metrics['accuracy']
                    
                    # Print for debugging
                    print(f"Processed: {filename} -> {model_name} / {prompt_type} = {metrics['accuracy']}%")
    
    # Sort models for consistent ordering
    all_models = sorted(data.keys())
    
    # Create output directory if it doesn't exist
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV
    with open(output_file, 'w', newline='') as csvfile:
        # Create header
        header = ['Model'] + [COLUMN_NAMES.get(p, p) for p in PROMPT_ORDER]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write data for each model
        for model in all_models:
            row = [model]
            for prompt in PROMPT_ORDER:
                accuracy = data[model].get(prompt, '')
                if accuracy != '':
                    row.append(f"{accuracy:.2f}")
                else:
                    row.append('')
            writer.writerow(row)
    
    print(f"\nCSV file created: {output_file}")
    print(f"Total models: {len(all_models)}")
    print(f"Models: {', '.join(all_models)}")


if __name__ == '__main__':
    main()

