#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_EXE="ollama.exe"
MODEL="${1:?usage: wait_then_launch_ollama_mt_translation_benchmark.sh <model> [run_name] [batch_size] [timeout_seconds] [poll_seconds] [benchmark_path]}"
RUN_NAME="${2:-benchmark_${MODEL//[:\/]/_}}"
BATCH_SIZE="${3:-1}"
TIMEOUT_SECONDS="${4:-300}"
POLL_SECONDS="${5:-60}"
BENCHMARK_PATH="${6:-$SCRIPT_DIR/translation_qa_20260416/mt_translation_gold_benchmark.tsv}"

until "$OLLAMA_EXE" list | grep -Fq "$MODEL"; do
  sleep "$POLL_SECONDS"
done

exec "$SCRIPT_DIR/launch_ollama_mt_translation_benchmark.sh" "$MODEL" "$RUN_NAME" "$BATCH_SIZE" "$TIMEOUT_SECONDS" "$BENCHMARK_PATH"
