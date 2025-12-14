#!/usr/bin/env bash
# Helper script that runs add_functional_groups.py with repo defaults.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/data/hallu/hallu_fg.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${INPUT_JSONL}}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[ERROR] OPENAI_API_KEY is not set. Please export it before running." >&2
  exit 1
fi

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[ERROR] Input JSONL not found at ${INPUT_JSONL}" >&2
  exit 1
fi

echo "[INFO] Using input: ${INPUT_JSONL}"
echo "[INFO] Writing output: ${OUTPUT_JSONL}"
echo "[INFO] Invoking model: ${MODEL_ID:-gpt-5-mini}"

CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/add_functional_groups.py"
  "--input" "${INPUT_JSONL}"
  "--output" "${OUTPUT_JSONL}"
  "--model" "${MODEL_ID:-gpt-5-mini}"
)

exec "${CMD[@]}" "$@"

