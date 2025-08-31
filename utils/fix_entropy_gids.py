import argparse
import json
import os
from pathlib import Path
from typing import Any


def fix_entropy_gids_in_obj(obj: Any) -> None:
    """Recursively fix nested entropy_gids fields from [[...]] to [...].

    This mutates the input object in-place.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "entropy_gids" and isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
                obj[key] = value[0]
            else:
                fix_entropy_gids_in_obj(value)
    elif isinstance(obj, list):
        for item in obj:
            fix_entropy_gids_in_obj(item)


def process_json_file(in_path: str, out_path: str) -> None:
    text = Path(in_path).read_text(encoding="utf-8")
    data = json.loads(text)
    fix_entropy_gids_in_obj(data)
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_jsonl_file(in_path: str, out_path: str) -> None:
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            rec = json.loads(line)
            fix_entropy_gids_in_obj(rec)
            json.dump(rec, fout, ensure_ascii=False)
            fout.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten nested entropy_gids fields from [[...]] to [...]")
    ap.add_argument("input", help="Input file path (.json or .jsonl)")
    ap.add_argument("--output", "-o", help="Output file path. If omitted, overwrites input in-place.")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or in_path

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    lower = in_path.lower()
    if lower.endswith(".jsonl"):
        process_jsonl_file(in_path, out_path)
    elif lower.endswith(".json"):
        process_json_file(in_path, out_path)
    else:
        raise ValueError("Unsupported file type. Expect .json or .jsonl")

    print(f"Wrote fixed file -> {out_path}")


if __name__ == "__main__":
    main()


