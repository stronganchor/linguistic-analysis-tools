#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/remap_wav2vec2_ctc_vocab.py" \
  --source-model "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-ipa-v001-checkpoint235-export" \
  --manifest "$Z_WORKSPACE/Speech to Text IPA/training_data/manifest_ipa.tsv" \
  --output-dir "$Z_MODELS_DIR/speech_to_text_ipa/alternatives/mms-1b-all-ipa-dental-remap-export-v001" \
  --strip-dental-stops \
  --redentalize-output
