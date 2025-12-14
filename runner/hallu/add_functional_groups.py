#!/usr/bin/env python3
"""
Augment hallu datasets with GPT-5 generated functional groups.

The script reads a JSONL file (default: ../../data/hallu/hallu_fg.jsonl),
extracts the assistant-provided structural description from each entry,
invokes the GPT-5 API to enumerate functional groups, and writes the
updated JSONL back to disk (overwriting by default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI  # type: ignore

    _USE_RESPONSES_API = True
except ImportError:  # pragma: no cover - fallback for legacy openai package
    _USE_RESPONSES_API = False
    try:
        import openai  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The openai package is required. Install it via `pip install openai`."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add GPT-5 generated functional_groups entries to hallu JSONL data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "hallu" / "hallu_fg.jsonl",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1-mini",
        help="GPT-5 model identifier to use for extraction.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process at most this many records (useful for smoke tests).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the GPT-5 call (omit to use model default).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait between retries when the API fails.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries per record.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate functional_groups even if already present.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
    return records


def dump_records(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_assistant_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    assistant_chunks = [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant" and msg.get("content")
    ]
    combined = "\n\n".join(chunk.strip() for chunk in assistant_chunks if chunk.strip())
    return combined or None


def build_prompt(structural_description: str) -> str:
    return (
        "You are analyzing a molecule description written by a chemist.\n"
        "Read the structural analysis and respond with a comma-separated list of the\n"
        "functional groups that are explicitly present. Use generic names (e.g., "
        "alkene, alkyne, tertiary alcohol, aldehyde, amide, aromatic ring). "
        "Avoid speculative language and omit duplicates.\n\n"
        "Structural description:\n"
        f"{structural_description.strip()}\n\n"
        "Functional groups:"
    )


def init_client() -> Any:
    if _USE_RESPONSES_API:
        return OpenAI()
    # Legacy ChatCompletion client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set."
        )
    openai.api_key = api_key
    return openai


def call_gpt5(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    retry_delay: float,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if _USE_RESPONSES_API:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "input": [
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "Return only comma-separated functional groups.",
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt}],
                        },
                    ],
                }
                if temperature is not None:
                    request_kwargs["temperature"] = temperature
                response = client.responses.create(**request_kwargs)
                text = (response.output_text or "").strip()
            else:
                response = client.ChatCompletion.create(
                    model=model,
                    temperature=temperature if temperature is not None else 0.0,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only comma-separated functional groups.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
                text = response.choices[0].message["content"].strip()

            if not text:
                raise ValueError("Received empty response from GPT-5.")
            return text
        except Exception as exc:  # pragma: no cover - network failures
            last_error = exc
            print(
                f"[WARN] GPT-5 call failed on attempt {attempt}/{max_retries}: {exc}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)
    raise RuntimeError("GPT-5 call failed after retries") from last_error


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input
    client = init_client()

    records = load_records(args.input)
    total = len(records)
    limit = args.max_records or total
    updated = 0

    for idx, record in enumerate(records[:limit], start=1):
        if (
            "functional_groups" in record
            and not args.overwrite_existing
            and record["functional_groups"]
        ):
            continue

        structural_description = extract_assistant_text(record.get("messages", []))
        if not structural_description:
            print(
                f"[WARN] Record {idx} missing assistant description; skipping.",
                file=sys.stderr,
            )
            continue

        prompt = build_prompt(structural_description)
        generated = call_gpt5(
            client=client,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )
        record["functional_groups"] = generated
        updated += 1
        print(f"[INFO] Processed record {idx}/{limit}: {generated}")

    dump_records(records, output_path)
    print(
        f"[DONE] Updated {updated} of {limit} processed records. "
        f"Written to {output_path}."
    )


if __name__ == "__main__":
    main()

