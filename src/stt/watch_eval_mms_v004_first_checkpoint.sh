#!/usr/bin/env bash
set -euo pipefail

cd "$Z_WORKSPACE/Speech to Text IPA"

STT_MODELS_DIR="${Z_STT_MODELS_DIR:-$Z_MODELS_DIR/speech_to_text_ipa}"
MODEL_DIR="${STT_MODELS_DIR}/alternatives/mms-1b-all-ipa-warmstart-v004-20260408-134225"
MANIFEST="training_data/genc-palu-recording_ipa-20260408-134225/manifest_ipa.tsv"
OUTPUT_DIR="evals/mms-v004-first-checkpoint-on-20260408-134225-eval-cpu-greedy"
LOG_PATH="evals/mms-v004-first-checkpoint-on-20260408-134225-eval-cpu-greedy.log"

while true; do
    CKPT="$(find "$MODEL_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort | tail -n 1 || true)"
    if [[ -n "$CKPT" ]]; then
        nice -n 10 $LINUX_STT_ENV/.venv/bin/python3 evaluate_hf_phone_model.py \
            --manifest "$MANIFEST" \
            --model "$CKPT" \
            --output-dir "$OUTPUT_DIR" \
            --device cpu \
            --decoder greedy \
            > "$LOG_PATH" 2>&1
        break
    fi
    sleep 300
done
