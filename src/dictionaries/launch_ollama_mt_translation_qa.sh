#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_EXE="${OLLAMA_EXE:-ollama.exe}"
MODEL="${1:-qwen2.5:7b}"
THINK_MODE="${THINK_MODE:-false}"
BATCH_SIZE="${BATCH_SIZE:-3}"
PILOT_BATCHES="${PILOT_BATCHES:-3}"
FULL_APPLY="${FULL_APPLY:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-240}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SAFE_MODEL="$(printf '%s' "$MODEL" | tr ':/' '__')"
PILOT_RUN_NAME="pilot_${SAFE_MODEL}_${TIMESTAMP}"
FULL_RUN_NAME="full_${SAFE_MODEL}_${TIMESTAMP}"

if [[ ! -x "$OLLAMA_EXE" ]]; then
  echo "Ollama executable not found: $OLLAMA_EXE" >&2
  exit 1
fi

if ! "$OLLAMA_EXE" list | grep -q "^${MODEL}[[:space:]]"; then
  "$OLLAMA_EXE" pull "$MODEL"
fi

python3 "$SCRIPT_DIR/run_ollama_mt_translation_qa.py" \
  --model "$MODEL" \
  --think-mode "$THINK_MODE" \
  --batch-size "$BATCH_SIZE" \
  --max-batches "$PILOT_BATCHES" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --run-name "$PILOT_RUN_NAME"

python3 - "$SCRIPT_DIR/translation_qa_20260416/ollama_auto_review/$PILOT_RUN_NAME/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("successful_batch_count", 0) <= 0:
    raise SystemExit("Pilot produced no successful batches")
if summary.get("failed_batch_count", 0) > 0:
    raise SystemExit("Pilot had batch failures; refusing to start full run")
PY

if [[ "$FULL_APPLY" == "1" ]]; then
  python3 "$SCRIPT_DIR/run_ollama_mt_translation_qa.py" \
    --model "$MODEL" \
    --think-mode "$THINK_MODE" \
    --batch-size "$BATCH_SIZE" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --run-name "$FULL_RUN_NAME" \
    --apply \
    --rebuild
else
  python3 "$SCRIPT_DIR/run_ollama_mt_translation_qa.py" \
    --model "$MODEL" \
    --think-mode "$THINK_MODE" \
    --batch-size "$BATCH_SIZE" \
    --timeout-seconds "$TIMEOUT_SECONDS" \
    --run-name "$FULL_RUN_NAME"
fi
