#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ipa.py" \
  --manifest "$Z_WORKSPACE/Speech to Text IPA/training_data/manifest_ipa.tsv" \
  --base-model "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-ipa-v001-checkpoint235-export" \
  --output-dir "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-ipa-dental-continue-v001" \
  --run-name "mms-1b-all-ipa-dental-continue-v001" \
  --epochs 8 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 8 \
  --learning-rate 5e-6 \
  --warmup-ratio 0.05 \
  --weight-decay 0.01 \
  --patience 3 \
  --dataloader-workers 0 \
  --freeze-feature-encoder \
  --gradient-checkpointing \
  --strip-dental-stops \
  --redentalize-output \
  --bf16
