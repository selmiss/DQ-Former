import json
from pathlib import Path
import shutil
import os

# Path to your JSON file
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules_brics.json")
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules_brics_entropy_gids.json")
# path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules-test_brics.json")
path = Path("data/Mol-LLaMA-Instruct/pubchem-molecules-test_brics_entropy_gids.json")

# Backup the original file
backup_path = path.with_suffix(path.suffix + ".bak")
shutil.copy2(path, backup_path)

# Load, transform, and write back
with path.open("r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise ValueError("Top-level JSON is not a list.")

changed = 0
for item in data:
    if isinstance(item, dict) and "brics_ids" in item:
        item["brics_gids"] = item.pop("brics_ids")
        changed += 1

with path.open("w", encoding="utf-8") as f:
    # Write minified to avoid changing indentation style unpredictably
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

print(f"Renamed key in {changed} item(s). Backup saved to: {backup_path}")