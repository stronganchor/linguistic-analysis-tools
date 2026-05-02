#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OLLAMA_EXE="${OLLAMA_EXE:-ollama.exe}"
MODEL="${1:-qwen2.5:7b}"
POLL_SECONDS="${POLL_SECONDS:-30}"

if [[ ! -x "$OLLAMA_EXE" ]]; then
  echo "Ollama executable not found: $OLLAMA_EXE" >&2
  exit 1
fi

until "$OLLAMA_EXE" list | grep -q "^${MODEL}[[:space:]]"; do
  sleep "$POLL_SECONDS"
done

"$SCRIPT_DIR/launch_ollama_mt_translation_qa.sh" "$MODEL"
