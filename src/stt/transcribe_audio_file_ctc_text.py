#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from prepare_lltools_ipa_dataset import normalize_plain_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def normalize_simple_text(value: str) -> str:
    text = normalize_plain_text(value)
    text = text.lower()
    text = text.replace("|", " ")
    for mark in [".", "?", "!", ",", ":", ";", "(", ")"]:
        text = text.replace(mark, " ")
    text = text.replace("â", "e")
    text = " ".join(text.split())
    return text


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Transcribe an audio file with a local Wav2Vec2/MMS CTC text model.")
    parser.add_argument("--audio", required=True, help="Audio file path.")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Model checkpoint directory containing config/model weights.",
    )
    parser.add_argument(
        "--processor-dir",
        default="",
        help="Optional processor/tokenizer directory if separate from model-dir.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--chunk-sec", type=float, default=12.0, help="Chunk size in seconds.")
    parser.add_argument("--batch-size", type=int, default=2, help="Chunk batch size.")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output file prefix. Defaults to Speech to Text IPA/transcription_outputs/<audio-stem>_<model-stem>.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "transcription_outputs"),
        help="Directory for output files when output-prefix is not absolute.",
    )
    return parser.parse_args()


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    waveform, sr = torchaudio.load(str(path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def make_output_prefix(args: argparse.Namespace, audio_path: Path, model_dir: Path) -> Path:
    if args.output_prefix:
        prefix = Path(args.output_prefix)
        if prefix.is_absolute():
            return prefix
        return Path(args.output_dir).resolve() / prefix
    safe_audio = audio_path.stem.replace(" ", "_")
    safe_model = model_dir.name.replace(" ", "_")
    return Path(args.output_dir).resolve() / f"{safe_audio}_{safe_model}_cpu"


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    processor_dir = Path(args.processor_dir).expanduser().resolve() if args.processor_dir else model_dir
    output_prefix = make_output_prefix(args, audio_path, model_dir)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    target_sr = int(processor.feature_extractor.sampling_rate)
    waveform, sr = load_audio(audio_path, target_sr)
    duration_sec = waveform.numel() / sr
    chunk_samples = max(1, int(args.chunk_sec * sr))
    total_chunks = math.ceil(waveform.numel() / chunk_samples)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for start_index in range(0, total_chunks, args.batch_size):
            chunk_batch = []
            batch_meta = []
            for chunk_index in range(start_index, min(start_index + args.batch_size, total_chunks)):
                start_sample = chunk_index * chunk_samples
                end_sample = min(waveform.numel(), (chunk_index + 1) * chunk_samples)
                chunk = waveform[start_sample:end_sample].numpy()
                chunk_batch.append(chunk)
                batch_meta.append(
                    {
                        "segment_index": chunk_index + 1,
                        "start_sec": start_sample / sr,
                        "end_sec": end_sample / sr,
                        "duration_sec": (end_sample - start_sample) / sr,
                    }
                )

            inputs = processor.feature_extractor(
                chunk_batch,
                sampling_rate=target_sr,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs["input_values"].to(device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            decoded = [normalize_simple_text(text) for text in processor.batch_decode(pred_ids)]

            for meta, pred in zip(batch_meta, decoded):
                rows.append(
                    {
                        **meta,
                        "predicted_text": pred,
                    }
                )
            print(f"Decoded {min(start_index + args.batch_size, total_chunks)}/{total_chunks} chunk(s)", flush=True)

    rows.sort(key=lambda row: int(row["segment_index"]))
    joined_text = " ".join(str(row["predicted_text"]) for row in rows if str(row["predicted_text"]).strip())

    tsv_path = output_prefix.with_name(output_prefix.name + "_segments.tsv")
    txt_path = output_prefix.with_name(output_prefix.name + "_joined.txt")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["segment_index", "start_sec", "end_sec", "duration_sec", "predicted_text"])
        for row in rows:
            writer.writerow(
                [
                    row["segment_index"],
                    f"{float(row['start_sec']):.3f}",
                    f"{float(row['end_sec']):.3f}",
                    f"{float(row['duration_sec']):.3f}",
                    row["predicted_text"],
                ]
            )

    txt_path.write_text(joined_text + ("\n" if joined_text else ""), encoding="utf-8")

    summary = {
        "audio_path": str(audio_path),
        "model_dir": str(model_dir),
        "processor_dir": str(processor_dir),
        "segments": len(rows),
        "duration_sec": duration_sec,
        "chunk_sec": args.chunk_sec,
        "device": str(device),
        "outputs": {
            "tsv": str(tsv_path),
            "txt": str(txt_path),
        },
        "preview": rows[:5],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
