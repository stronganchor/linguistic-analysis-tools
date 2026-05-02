#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ipa.py" \
  --manifest "$Z_WORKSPACE/Speech to Text IPA/training_data/manifest_ipa.tsv" \
  --base-model "facebook/mms-1b-all" \
  --target-lang "zza" \
  --output-dir "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-zza-ipa-v001" \
  --run-name "mms-1b-all-zza-ipa-v001" \
  --epochs 15 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --patience 4 \
  --dataloader-workers 0 \
  --freeze-feature-encoder \
  --gradient-checkpointing \
  --bf16
