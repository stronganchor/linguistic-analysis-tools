#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_rows(manifest_path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split") != "eval":
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def resolve_audio_path(row: dict) -> Path:
    raw_value = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw_value = str(raw_value).strip()
    if not raw_value:
        raise RuntimeError("Missing audio path in manifest row")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail

    return Path(raw_value).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode sample eval rows from a CTC text model.")
    parser.add_argument("--processor", required=True, help="Directory containing the saved processor/tokenizer")
    parser.add_argument("--model", required=True, help="Directory containing the model weights")
    parser.add_argument("--manifest", required=True, help="Path to manifest_text.tsv")
    parser.add_argument("--limit", type=int, default=10, help="Number of eval rows to decode")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.processor)
    model = Wav2Vec2ForCTC.from_pretrained(args.model)
    model.eval()
    model.to(args.device)

    target_sr = int(processor.feature_extractor.sampling_rate)
    rows = load_rows(Path(args.manifest), args.limit)

    for idx, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row)
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        inputs = processor(
            waveform.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(args.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0]

        print(
            json.dumps(
                {
                    "index": idx,
                    "word_title": row.get("word_title"),
                    "recording_type": row.get("recording_type"),
                    "audio": row.get("audio"),
                    "target_text": row.get("target_text"),
                    "predicted_text": pred_text,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
