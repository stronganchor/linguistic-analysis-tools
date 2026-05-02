#!/usr/bin/env bash
set -euo pipefail

WIN_ROOT="$Z_WORKSPACE/Speech to Text IPA"
WSL_ROOT="$HOME/zazaki-stt-linux/workspace"
VENV_ROOT="$HOME/zazaki-stt-linux/.venv"
MODEL_NAME="whisper-large-v3-ipa-nostress-genc-palu-lora-wsl-v001"
OUTPUT_DIR="$WSL_ROOT/outputs/$MODEL_NAME"
HF_CACHE_LARGE="$HOME/.cache/huggingface/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"
WIN_HF_CACHE_LARGE="$HF_HOME/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1"

MODEL_SOURCE="openai/whisper-large-v3"
if [[ -f "$HF_CACHE_LARGE/model.safetensors" ]]; then
  MODEL_SOURCE="$HF_CACHE_LARGE"
elif [[ -f "$WIN_HF_CACHE_LARGE/model.safetensors" ]]; then
  MODEL_SOURCE="$WIN_HF_CACHE_LARGE"
fi

if [[ ! -x "$VENV_ROOT/bin/python" ]]; then
  echo "Missing WSL venv at $VENV_ROOT"
  echo "Set up the Linux environment first."
  exit 1
fi

mkdir -p "$WSL_ROOT/audio" "$WSL_ROOT/training_data" "$WSL_ROOT/models" "$WSL_ROOT/outputs"

cp "$WIN_ROOT/train_whisper_ipa.py" "$WIN_ROOT/prepare_lltools_ipa_dataset.py" "$WSL_ROOT/"
rsync -a \
  "$WIN_ROOT/extracted_bundles_20260403/ll-tools-stt-training-genc-palu-recording_ipa-20260403-090111/audio/" \
  "$WSL_ROOT/audio/"

python3 - <<'PY'
import csv
from pathlib import Path

win_root = Path("$Z_WORKSPACE/Speech to Text IPA")
wsl_root = Path.home() / "zazaki-stt-linux/workspace"
src = win_root / "training_data/manifest_ipa.tsv"
dest = wsl_root / "training_data/manifest_ipa_linux.tsv"

with src.open("r", encoding="utf-8", newline="") as in_handle, dest.open("w", encoding="utf-8", newline="") as out_handle:
    reader = csv.DictReader(in_handle, delimiter="\t")
    writer = csv.DictWriter(out_handle, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    for row in reader:
        rel = str(row.get("audio_relpath") or "").replace("audio/", "", 1)
        audio_path = (wsl_root / "audio" / rel).resolve()
        audio_str = str(audio_path)
        row["audio"] = audio_str
        row["audio_windows"] = audio_str
        row["audio_posix"] = audio_str
        writer.writerow(row)

print(dest)
PY

if [[ ! -d "$WSL_ROOT/models/whisper-medium-ipa-nostress-genc-palu-warmstart-best-v003" ]]; then
  rsync -a \
    "$WIN_ROOT/models/high-accuracy/whisper-medium-ipa-nostress-genc-palu-warmstart-best-v003/" \
    "$WSL_ROOT/models/whisper-medium-ipa-nostress-genc-palu-warmstart-best-v003/"
fi

source "$VENV_ROOT/bin/activate"
cd "$WSL_ROOT"

python train_whisper_ipa.py \
  --manifest "$WSL_ROOT/training_data/manifest_ipa_linux.tsv" \
  --model-source "$MODEL_SOURCE" \
  --output-dir "$OUTPUT_DIR" \
  --num-train-epochs 10 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-4 \
  --warmup-ratio 0.1 \
  --optim "adamw_torch" \
  --generation-num-beams 1 \
  --max-grad-norm 0.5 \
  --torch-dtype "bfloat16" \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-target-modules "q_proj,k_proj,v_proj,out_proj,fc1,fc2" \
  --lora-modules-to-save "embed_tokens,proj_out"
