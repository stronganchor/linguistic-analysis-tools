#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

ROOT="$Z_WORKSPACE/Speech to Text IPA"
IPA_MANIFEST="${ROOT}/training_data/genc-palu-recording_ipa-20260407-124628/manifest_ipa.tsv"
STAGE1_MODEL="${ROOT}/models/alternatives/mms-300m-zza-cv25-orthography-v001"
OUTPUT_DIR="${ROOT}/models/alternatives/mms-300m-zza-cv25-to-ipa-v001"

python3 "${ROOT}/train_wav2vec2_ipa.py" \
  --manifest "${IPA_MANIFEST}" \
  --base-model "${STAGE1_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "mms-300m-zza-cv25-to-ipa-v001" \
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
