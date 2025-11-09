"""
Convert a combined JSON file into separate metadata JSON and data JSONL files.

This script takes a single JSON file containing both metadata and a 'data_list' field,
then splits it into:
  1. A metadata JSON file containing all fields except 'data_list'
  2. A JSONL file where each line is a JSON object from the 'data_list' array

This is useful for processing zeroshot evaluation data where metadata and data records
need to be stored separately.
"""
import argparse
import json
from typing import Any, Dict


def load_combined_json(input_path: str) -> Dict[str, Any]:
    with open(input_path, "r") as f:
        return json.load(f)


def write_meta_json(meta: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)


def write_data_jsonl(data_list: Any, output_path: str) -> None:
    with open(output_path, "w") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert(input_path: str, meta_out: str, jsonl_out: str) -> None:
    combined = load_combined_json(input_path)

    if "data_list" not in combined:
        raise KeyError("Input combined JSON must contain key 'data_list'.")

    data_list = combined["data_list"]

    # Everything except data_list is considered metadata
    meta = {k: v for k, v in combined.items() if k != "data_list"}

    write_meta_json(meta, meta_out)
    write_data_jsonl(data_list, jsonl_out)

    print(f"Wrote meta JSON to: {meta_out}")
    print(f"Wrote data JSONL with {len(data_list)} records to: {jsonl_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert combined zeroshot JSON into meta JSON and data JSONL")
    parser.add_argument("--input", required=True, help="Path to combined JSON file (contains metadata and data_list)")
    parser.add_argument("--meta_out", required=True, help="Path to write metadata JSON")
    parser.add_argument("--jsonl_out", required=True, help="Path to write data JSONL")
    args = parser.parse_args()

    convert(args.input, args.meta_out, args.jsonl_out)


if __name__ == "__main__":
    main()

# python {BASE_DIR}/zeroshot/convert_to_jsonl.py --input {DATA_DIR}/pampa/data/pampa.json --meta_out {DATA_DIR}/zeroshot/pampa/pampa_meta.json --jsonl_out {DATA_DIR}/zeroshot/pampa/pampa.jsonl
