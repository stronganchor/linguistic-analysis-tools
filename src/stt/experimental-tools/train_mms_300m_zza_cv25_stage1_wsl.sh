#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

ROOT="$Z_WORKSPACE/Speech to Text IPA"
ARCHIVE="${ROOT}/external-data/common-voice/common-voice-scripted-speech-25-0-zaza-4baca5e5.tar.gz"
DATASET_DIR="${ROOT}/training_data/common-voice-zza-25-orthography"
OUTPUT_DIR="${ROOT}/models/alternatives/mms-300m-zza-cv25-orthography-v001"

python3 "${ROOT}/prepare_common_voice_zaza_dataset.py" \
  --input "${ARCHIVE}" \
  --output-dir "${DATASET_DIR}" \
  --lowercase-targets

python3 "${ROOT}/train_wav2vec2_ctc_text.py" \
  --manifest "${DATASET_DIR}/manifest_text.tsv" \
  --target-column "target_text" \
  --base-model "${ROOT}/models/base/facebook-mms-300m" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "mms-300m-zza-cv25-orthography-v001" \
  --epochs 15 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --grad-accum 8 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --weight-decay 0.01 \
  --patience 4 \
  --dataloader-workers 0 \
  --freeze-feature-encoder \
  --gradient-checkpointing \
  --lowercase-targets \
  --bf16
