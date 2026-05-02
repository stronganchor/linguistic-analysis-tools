#!/usr/bin/env bash
set -euo pipefail

WIN_ROOT="$Z_WORKSPACE/Speech to Text IPA"
PYTHON_EXE="$Z_WORKSPACE/audio_corpus/venv311/Scripts/python.exe"
RUN_ID="${1:-$(date +%Y%m%d-%H%M%S)}"

MANIFEST_PATH="$WIN_ROOT/training_data/genc-palu-recording_text-20260409-071931-simple-v002-whisper-medium/manifest_text.tsv"
MODEL_SOURCE="$WIN_ROOT/models/high-accuracy/whisper-medium-ipa-nostress-genc-palu-warmstart-v005-20260410-065107/checkpoint-1254"
PROCESSOR_SOURCE="$WIN_ROOT/models/high-accuracy/whisper-medium-ipa-nostress-genc-palu-warmstart-v004-20260406"
OUTPUT_DIR="$WIN_ROOT/models/high-accuracy/whisper-medium-orth-simple-genc-palu-warmstart-v001-$RUN_ID"

SCRIPT_WIN=$(wslpath -w "$WIN_ROOT/train_whisper_ipa.py")
MANIFEST_WIN=$(wslpath -w "$MANIFEST_PATH")
MODEL_WIN=$(wslpath -w "$MODEL_SOURCE")
PROCESSOR_WIN=$(wslpath -w "$PROCESSOR_SOURCE")
OUTPUT_WIN=$(wslpath -w "$OUTPUT_DIR")

mkdir -p "$OUTPUT_DIR"

exec "$PYTHON_EXE" "$SCRIPT_WIN" \
  --manifest "$MANIFEST_WIN" \
  --model-source "$MODEL_WIN" \
  --processor-source "$PROCESSOR_WIN" \
  --target-field target_text \
  --target-mode plain \
  --output-dir "$OUTPUT_WIN" \
  --num-train-epochs 20 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --gradient-checkpointing \
  --learning-rate 5e-6 \
  --warmup-ratio 0.1 \
  --optim adafactor \
  --generation-num-beams 4 \
  --torch-dtype bfloat16
