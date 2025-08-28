import json
from pathlib import Path
import shutil
import os

# Path to your JSON file
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules_brics.json")
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules_brics_entropy_gids.json")
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules-test_brics.json")
path = Path("data/zeroshot/pampa/data.jsonl")

# Backup the original file
backup_path = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, backup_path)

# Detect format based on file extension
is_jsonl = str(path).lower().endswith('.jsonl')

# Load data
if is_jsonl:
    # Parse JSONL format (one JSON object per line)
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num}: {e}")
else:
    # Parse JSON format
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("Top-level JSON is not a list.")

# Transform data
changed = 0
for item in data:
    if isinstance(item, dict) and "brics_ids" in item:
        item["brics_gids"] = item.pop("brics_ids")
        changed += 1

# Write back in the same format
if is_jsonl:
    # Write as JSONL (one JSON object per line)
    with path.open("w", encoding="utf-8") as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
            f.write('\n')
else:
    # Write as JSON
    with path.open("w", encoding="utf-8") as f:
        # Write minified to avoid changing indentation style unpredictably
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

print(f"Renamed key in {changed} item(s). Backup saved to: {backup_path}")