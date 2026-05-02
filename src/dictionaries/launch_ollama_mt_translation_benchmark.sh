#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: launch_ollama_mt_translation_benchmark.sh <model> [run_name] [batch_size] [timeout_seconds] [benchmark_path]}"
RUN_NAME="${2:-benchmark_${MODEL//[:\/]/_}}"
BATCH_SIZE="${3:-1}"
TIMEOUT_SECONDS="${4:-300}"
BENCHMARK_PATH="${5:-$SCRIPT_DIR/translation_qa_20260416/mt_translation_gold_benchmark.tsv}"

if [[ ! -f "$BENCHMARK_PATH" ]]; then
  python3 "$SCRIPT_DIR/build_mt_translation_gold_benchmark.py" --output-path "$BENCHMARK_PATH"
fi

python3 "$SCRIPT_DIR/run_ollama_mt_translation_qa.py" \
  --audit-path "$BENCHMARK_PATH" \
  --run-name "$RUN_NAME" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --skip-refresh-audit

python3 "$SCRIPT_DIR/score_ollama_mt_translation_qa_benchmark.py" \
  --benchmark-path "$BENCHMARK_PATH" \
  --run-dir "$SCRIPT_DIR/translation_qa_20260416/ollama_auto_review/$RUN_NAME"
