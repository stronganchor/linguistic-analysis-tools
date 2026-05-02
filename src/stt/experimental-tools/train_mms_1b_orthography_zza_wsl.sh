#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ctc_text.py" \
  --manifest "$Z_WORKSPACE/Speech to Text IPA/training_data/genc-palu-recording_text-20260409-071931-simple-v001/manifest_text.tsv" \
  --base-model "facebook/mms-1b-all" \
  --target-lang "zza" \
  --output-dir "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-orthography-zza-simple-v001-20260409-071931" \
  --run-name "mms-1b-all-orthography-zza-simple-v001-20260409-071931" \
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
