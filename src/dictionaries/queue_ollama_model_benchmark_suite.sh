#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_EXE="ollama.exe"
BENCHMARK_PATH="${1:?usage: queue_ollama_model_benchmark_suite.sh <benchmark_path> <model> [model ...]}"
shift

if [[ ! -f "$BENCHMARK_PATH" ]]; then
  echo "benchmark not found: $BENCHMARK_PATH" >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  echo "no models provided" >&2
  exit 1
fi

for MODEL in "$@"; do
  RUN_NAME="benchmark_${MODEL//[:\/]/_}_normal_b3"
  if ! "$OLLAMA_EXE" list | grep -Fq "$MODEL"; then
    "$OLLAMA_EXE" pull "$MODEL"
  fi
  "$SCRIPT_DIR/launch_ollama_mt_translation_benchmark.sh" "$MODEL" "$RUN_NAME" 3 300 "$BENCHMARK_PATH"
done
