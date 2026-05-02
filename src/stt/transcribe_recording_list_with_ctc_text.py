#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

try:
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    torchaudio = None

from prepare_lltools_ipa_dataset import normalize_plain_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg"}


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
    parser = argparse.ArgumentParser(
        description="Transcribe a TSV recording list with a local Wav2Vec2/MMS CTC text model."
    )
    parser.add_argument("--input-tsv", required=True, help="TSV with a recording_id column.")
    parser.add_argument("--model-dir", required=True, help="Local CTC text model checkpoint directory.")
    parser.add_argument(
        "--processor-dir",
        default="",
        help="Optional separate processor/tokenizer directory. Defaults to --model-dir.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--chunk-sec", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--search-root", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def default_search_roots(script_dir: Path) -> list[Path]:
    return [
        script_dir / "extracted_bundles",
        script_dir / "extracted_bundles_20260403",
        script_dir / "extracted_bundles_20260406",
        script_dir / "extracted_bundles_20260407_124628",
        script_dir / "extracted_bundles_20260408_134225",
        script_dir / "extracted_bundles_20260408_134225_blacklist_preview",
        script_dir / "extracted_bundles_full_audio_20260408",
        script_dir / "extracted_bundles_text",
        script_dir / "extracted_bundles_whisper_20260410",
    ]


def make_output_prefix(input_tsv: Path, output_prefix: str) -> Path:
    if output_prefix:
        prefix = Path(output_prefix)
        if prefix.is_absolute():
            return prefix
        return input_tsv.parent / prefix
    return input_tsv.with_suffix("")


def collect_audio_index(roots: list[Path]) -> dict[str, list[Path]]:
    rec_audio: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(\d+)-")
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return rec_audio

    if os.name == "nt":
        for root in existing_roots:
            for audio_dir_name in ("audio", "media"):
                for audio_dir in root.rglob(audio_dir_name):
                    if not audio_dir.is_dir():
                        continue
                    for path in audio_dir.iterdir():
                        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                            continue
                        match = pattern.match(path.name)
                        if match:
                            rec_audio[match.group(1)].append(path.resolve())
        return rec_audio

    cmd = [
        "find",
        *[str(root) for root in existing_roots],
        "-type",
        "f",
        "(",
        "-iname",
        "*.wav",
        "-o",
        "-iname",
        "*.mp3",
        "-o",
        "-iname",
        "*.m4a",
        "-o",
        "-iname",
        "*.aac",
        "-o",
        "-iname",
        "*.ogg",
        ")",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for raw_path in completed.stdout.splitlines():
        path = Path(raw_path.strip())
        if not path:
            continue
        match = pattern.match(path.name)
        if match:
            rec_audio[match.group(1)].append(path.resolve())
    return rec_audio


def choose_best_audio_path(paths: list[Path]) -> Path:
    def score(path: Path) -> tuple[int, int, str]:
        suffix_rank = {".wav": 0, ".mp3": 1, ".m4a": 2, ".aac": 3, ".ogg": 4}.get(path.suffix.lower(), 9)
        text_bonus = 0 if "extracted_bundles_text" in str(path) else 1
        return (text_bonus, suffix_rank, str(path))

    return sorted(paths, key=score)[0]


def windows_to_posix_path(path_str: str) -> Path:
    windows_path = PureWindowsPath(path_str)
    drive = windows_path.drive.rstrip(":").lower()
    path = Path("/mnt") / drive
    for part in windows_path.parts[1:]:
        if part not in {"\\", "/"}:
            path /= part
    return path


def resolve_existing_path(raw_value: str) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    direct_path = Path(value).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    if re.match(r"^[A-Za-z]:[\\/]", value):
        posix_path = windows_to_posix_path(value)
        if posix_path.exists():
            return posix_path.resolve()
    return None


def resolve_audio_path(row: dict[str, str], rec_audio: dict[str, list[Path]]) -> Path | None:
    for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"):
        resolved = resolve_existing_path(str(row.get(key) or ""))
        if resolved is not None:
            return resolved

    rid = str(row.get("recording_id") or "").strip()
    if not rid:
        return None
    audio_paths = rec_audio.get(rid) or []
    if not audio_paths:
        return None
    return choose_best_audio_path(audio_paths)


def load_rows(input_tsv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row.get("recording_id"):
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def load_audio(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    if torchaudio is not None:
        waveform, sr = torchaudio.load(str(path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        return waveform, sr

    audio_array, sr = librosa.load(str(path), sr=target_sr, mono=True)
    waveform = torch.from_numpy(audio_array.copy())
    return waveform, target_sr


def transcribe_audio_file(
    *,
    audio_path: Path,
    processor: Wav2Vec2Processor,
    model: Wav2Vec2ForCTC,
    device: torch.device,
    chunk_sec: float,
    batch_size: int,
) -> str:
    target_sr = int(processor.feature_extractor.sampling_rate)
    waveform, sr = load_audio(audio_path, target_sr)
    chunk_samples = max(1, int(chunk_sec * sr))
    total_chunks = math.ceil(waveform.numel() / chunk_samples)

    predicted_chunks: list[str] = []
    with torch.no_grad():
        for start_index in range(0, total_chunks, batch_size):
            batch_waveforms = []
            for chunk_index in range(start_index, min(start_index + batch_size, total_chunks)):
                start_sample = chunk_index * chunk_samples
                end_sample = min(waveform.numel(), (chunk_index + 1) * chunk_samples)
                batch_waveforms.append(waveform[start_sample:end_sample].numpy())

            inputs = processor.feature_extractor(
                batch_waveforms,
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
            predicted_chunks.extend(decoded)

    return " ".join(chunk for chunk in predicted_chunks if chunk.strip()).strip()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    processor_dir = Path(args.processor_dir).expanduser().resolve() if args.processor_dir else model_dir
    output_prefix = make_output_prefix(input_tsv, args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    roots = [Path(p).expanduser().resolve() for p in args.search_root] if args.search_root else default_search_roots(script_dir)
    rows = load_rows(input_tsv, args.limit)
    needs_search_index = any(
        not any(str(row.get(key) or "").strip() for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"))
        for row in rows
    )
    rec_audio = collect_audio_index(roots) if needs_search_index else {}

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    device = torch.device(args.device)
    model.to(device).eval()

    results: list[dict[str, str]] = []
    missing_audio: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        rid = row["recording_id"].strip()
        audio_path = resolve_audio_path(row, rec_audio)
        if audio_path is None:
            missing_audio.append(row)
            continue
        predicted_text = transcribe_audio_file(
            audio_path=audio_path,
            processor=processor,
            model=model,
            device=device,
            chunk_sec=args.chunk_sec,
            batch_size=args.batch_size,
        )
        result = dict(row)
        result["local_audio_path"] = str(audio_path)
        result["predicted_text"] = predicted_text
        results.append(result)
        print(f"[{index}/{len(rows)}] {rid} -> {predicted_text}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["recording_id"]
    for extra in ("local_audio_path", "predicted_text"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    tsv_path = output_prefix.with_name(output_prefix.name + "_ctc_text_predictions.tsv")
    jsonl_path = output_prefix.with_name(output_prefix.name + "_ctc_text_predictions.jsonl")
    missing_audio_path = output_prefix.with_name(output_prefix.name + "_missing_audio.tsv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing_fields = list(rows[0].keys()) if rows else ["recording_id"]
    with missing_audio_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(missing_audio)

    summary = {
        "input_tsv": str(input_tsv),
        "model_dir": str(model_dir),
        "processor_dir": str(processor_dir),
        "device": str(device),
        "chunk_sec": args.chunk_sec,
        "batch_size": args.batch_size,
        "input_rows": len(rows),
        "transcribed_rows": len(results),
        "missing_audio_rows": len(missing_audio),
        "outputs": {
            "tsv": str(tsv_path),
            "jsonl": str(jsonl_path),
            "missing_audio_tsv": str(missing_audio_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
