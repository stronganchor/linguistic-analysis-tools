#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-vibevoice-ak-lora}"
SCRIPT="$Z_WORKSPACE/Speech to Text IPA/run_vibevoice_asr_ak_lora_train_wsl.sh"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash '$SCRIPT'"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
