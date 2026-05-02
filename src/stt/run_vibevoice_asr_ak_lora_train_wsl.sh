#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$Z_WORKSPACE"
STT_DIR="$WORKSPACE/Speech to Text IPA"
VENV="$LINUX_STT_ENV/.venv-vibevoice"
VIBEVOICE_SRC="$LINUX_STT_ENV/vendor/VibeVoice"
MODEL_DIR="$Z_MODELS_DIR/speech_to_text_ipa/base/microsoft-vibevoice-asr"
DATA_DIR="$STT_DIR/training_data/vibevoice_asr_ak_ipa_v001/train"
OUTPUT_DIR="$Z_MODELS_DIR/speech_to_text_ipa/experiments/vibevoice-asr-ak-ipa-lora-v001"
LOG_DIR="$Z_MODELS_DIR/speech_to_text_ipa/logs"
LOG_FILE="$LOG_DIR/vibevoice-asr-ak-ipa-lora-v001-$(date +%Y%m%d-%H%M%S).log"

missing=0
for path in "$VENV/bin/python" "$VIBEVOICE_SRC/vibevoice" "$STT_DIR/train_vibevoice_asr_lora_local.py" "$DATA_DIR"; do
  if [ ! -e "$path" ]; then
    echo "Missing required path: $path" >&2
    missing=1
  fi
done

for idx in 1 2 3 4 5 6 7 8; do
  shard="$MODEL_DIR/model-$(printf '%05d' "$idx")-of-00008.safetensors"
  if [ ! -s "$shard" ]; then
    echo "Model download is not complete; missing shard: $shard" >&2
    missing=1
  fi
done

if [ ! -s "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "Model download is not complete; missing model.safetensors.index.json" >&2
  missing=1
fi

if [ "$missing" -ne 0 ]; then
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$WORKSPACE"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CC="$STT_DIR/zig-cc-wrapper.sh"
export PYTHONPATH="$VIBEVOICE_SRC:$STT_DIR:${PYTHONPATH:-}"

"$VENV/bin/python" "$STT_DIR/train_vibevoice_asr_lora_local.py" \
  --model_path "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --device auto \
  --attn_implementation sdpa \
  --load_in_4bit true \
  --use_customized_context true \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --num_train_epochs 8 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 3 \
  --gradient_checkpointing true \
  --bf16 true \
  --report_to none \
  2>&1 | tee "$LOG_FILE"
