#!/usr/bin/env bash
set -euo pipefail

MAX_SAMPLES="${1:-25}"

PROJECT_ROOT="$Z_WORKSPACE"
VENV_PY="$LINUX_STT_ENV/.venv-vibevoice/bin/python"
VIBEVOICE_SRC="$LINUX_STT_ENV/vendor/VibeVoice"
BASE_MODEL="$Z_MODELS_DIR/speech_to_text_ipa/base/microsoft-vibevoice-asr"
LORA_PATH="$Z_MODELS_DIR/speech_to_text_ipa/experiments/vibevoice-asr-ak-orthography-lora-v001"
EVAL_DIR="${PROJECT_ROOT}/Speech to Text IPA/training_data/vibevoice_asr_ak_orthography_v001/eval"
OUT_DIR="$Z_MODELS_DIR/speech_to_text_ipa/evaluations/vibevoice-asr-ak-orthography-lora-v001"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${VIBEVOICE_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CC="${PROJECT_ROOT}/Speech to Text IPA/zig-cc-wrapper.sh"

if [[ "${MAX_SAMPLES}" == "0" ]]; then
  RUN_LABEL="eval_all"
else
  RUN_LABEL="eval_first${MAX_SAMPLES}"
fi

"${VENV_PY}" "${PROJECT_ROOT}/Speech to Text IPA/evaluate_vibevoice_asr_lora.py" \
  --base-model "${BASE_MODEL}" \
  --lora-path "${LORA_PATH}" \
  --eval-dir "${EVAL_DIR}" \
  --output-jsonl "${OUT_DIR}/${RUN_LABEL}_predictions.jsonl" \
  --summary-json "${OUT_DIR}/${RUN_LABEL}_summary.json" \
  --max-samples "${MAX_SAMPLES}" \
  --max-new-tokens 192
