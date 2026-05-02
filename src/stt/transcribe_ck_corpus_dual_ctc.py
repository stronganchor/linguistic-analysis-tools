#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor

from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text, to_windows_path
from review_lltools_ipa_comparison import ipa_to_canonical, orthography_to_canonical
from serve_local_stt_mms import (
    TARGET_SR,
    build_pyctc_labels,
    choose_torch_dtype,
    clean_beam_decoded_text,
    load_audio_file,
    load_local_model_config,
    resolve_feature_dtype,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
AUDIO_WAV16K_DIR = ROOT_DIR / "audio_corpus" / "audio_wav16k"
PHASE1A_SPLIT_TSV = SCRIPT_DIR / "transcription_outputs" / "phase1a_processed_multi_ipa_cpu_20260410-133537" / "phase_split.tsv"
PHASE1B_AUDIO = ROOT_DIR / "Z class with C" / "Quizlet recordings" / "processed" / "phase1b_processed.mp3"


def default_ipa_model_dir() -> str:
    return str(SCRIPT_DIR / "models" / "alternatives" / "mms-1b-all-ipa-ck-tier1-ak-anchor-v001-20260410")


def default_orth_model_dir() -> str:
    return str(SCRIPT_DIR / "models" / "alternatives" / "mms-1b-all-orthography-ck-tier1-ak-anchor-v001-20260411")


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def choose_device(value: str) -> str:
    requested = (value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_source_key(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def slugify(value: str) -> str:
    slug = normalize_source_key(value).replace(" ", "_")
    return slug or "audio"


def repo_relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_local_path(value: str | Path) -> Path:
    raw = str(value).strip()
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        return (Path("/mnt") / drive / tail).resolve()
    return Path(raw).expanduser().resolve()


def normalize_simple_text(value: str) -> str:
    text = normalize_plain_text(value)
    text = text.lower()
    text = text.replace("|", " ")
    for mark in [".", "?", "!", ",", ":", ";", "(", ")", '"']:
        text = text.replace(mark, " ")
    text = text.replace("â", "e")
    return " ".join(text.split())


def load_audio_for_split(audio_path: Path) -> tuple[np.ndarray, int]:
    waveform, sr = load_audio_file(audio_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR
    return waveform.numpy(), sr


def load_clip_audio(audio_path: Path, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(str(resolve_local_path(audio_path)))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.numpy()


def split_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    merge_gap_sec: float,
    pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
) -> list[tuple[int, int]]:
    raw_intervals = librosa.effects.split(audio, top_db=top_db)
    if len(raw_intervals) == 0:
        return []

    merged: list[tuple[int, int]] = []
    current_start, current_end = raw_intervals[0]
    for start, end in raw_intervals[1:]:
        gap_sec = (start - current_end) / sr
        if gap_sec <= merge_gap_sec:
            current_end = end
            continue
        merged.append((int(current_start), int(current_end)))
        current_start, current_end = start, end
    merged.append((int(current_start), int(current_end)))

    pad_samples = int(round(pad_sec * sr))
    min_clip_samples = int(round(min_clip_sec * sr))
    max_clip_samples = max(min_clip_samples, int(round(max_clip_sec * sr)))
    clipped: list[tuple[int, int]] = []
    total_samples = len(audio)
    for start, end in merged:
        padded_start = max(0, start - pad_samples)
        padded_end = min(total_samples, end + pad_samples)
        if padded_end - padded_start < min_clip_samples:
            continue
        cursor = padded_start
        while cursor < padded_end:
            chunk_end = min(padded_end, cursor + max_clip_samples)
            if chunk_end - cursor >= min_clip_samples:
                clipped.append((cursor, chunk_end))
            cursor = chunk_end
    return clipped


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def collect_unique_full_recordings() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    wav_paths = sorted(
        path for path in AUDIO_WAV16K_DIR.iterdir() if path.is_file() and path.suffix.lower() in {".wav", ".mp3", ".m4a"}
    )
    wav_keys = {normalize_source_key(path.stem) for path in wav_paths}

    for path in wav_paths:
        rows.append(
            {
                "source_id": f"full_{slugify(path.stem)}",
                "source_title": path.stem,
                "source_group": "audio_wav16k",
                "source_kind": "full_recording",
                "source_audio_posix": path.resolve().as_posix(),
                "source_audio_windows": to_windows_path(path),
                "source_audio_relpath": repo_relpath(path),
                "split_method": "librosa_split",
            }
        )

    extras: list[Path] = []
    raw_dirs = [
        ROOT_DIR / "Z class with C" / "Phase 2C recordings",
        ROOT_DIR / "Z class with C" / "Phase 4",
        ROOT_DIR / "Z class with C" / "World stories",
    ]
    seen_extra_keys: set[str] = set()
    for directory in raw_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".mp3", ".m4a"}:
                continue
            key = normalize_source_key(path.stem)
            if directory.name != "World stories" and key in wav_keys:
                continue
            if key in seen_extra_keys:
                continue
            seen_extra_keys.add(key)
            extras.append(path)

    for path in extras:
        if path.parent.name == "World stories":
            source_group = "world_stories_extra"
        elif path.parent.name == "Phase 4":
            source_group = "phase4_extra"
        else:
            source_group = "phase2c_extra"
        rows.append(
            {
                "source_id": f"full_{slugify(path.stem)}",
                "source_title": path.stem,
                "source_group": source_group,
                "source_kind": "full_recording",
                "source_audio_posix": path.resolve().as_posix(),
                "source_audio_windows": to_windows_path(path),
                "source_audio_relpath": repo_relpath(path),
                "split_method": "librosa_split",
            }
        )

    return rows


def build_phase1a_segments() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_tsv(PHASE1A_SPLIT_TSV)
    if not rows:
        raise RuntimeError(f"No rows in {PHASE1A_SPLIT_TSV}")
    first = rows[0]
    source_audio_path = resolve_local_path(first["source_audio_posix"] or first["source_audio_windows"])
    source_row = {
        "source_id": "phase1a_processed",
        "source_title": "phase1a processed",
        "source_group": "quizlet_phase1a",
        "source_kind": "phase_processed_existing_split",
        "source_audio_posix": source_audio_path.as_posix(),
        "source_audio_windows": to_windows_path(source_audio_path),
        "source_audio_relpath": repo_relpath(source_audio_path),
        "split_method": "existing_phase1a_split",
    }
    segment_rows: list[dict[str, Any]] = []
    for row in rows:
        clip_id = int(row["clip_id"])
        clip_path = resolve_local_path(row["audio_posix"] or row["audio_windows"])
        segment_rows.append(
            {
                "segment_id": f"phase1a_{clip_id:04d}",
                "source_id": source_row["source_id"],
                "source_title": source_row["source_title"],
                "source_group": source_row["source_group"],
                "source_kind": source_row["source_kind"],
                "segment_index": clip_id,
                "clip_audio_posix": clip_path.as_posix(),
                "clip_audio_windows": to_windows_path(clip_path),
                "clip_audio_relpath": repo_relpath(clip_path),
                "source_audio_posix": source_row["source_audio_posix"],
                "source_audio_windows": source_row["source_audio_windows"],
                "source_audio_relpath": source_row["source_audio_relpath"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "duration_sec": float(row["duration_sec"]),
                "split_method": source_row["split_method"],
                "split_params": "",
            }
        )
    return source_row, segment_rows


def split_source_to_segments(
    *,
    source_row: dict[str, Any],
    run_dir: Path,
    top_db: float,
    merge_gap_sec: float,
    pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
) -> list[dict[str, Any]]:
    source_audio = resolve_local_path(source_row["source_audio_posix"])
    audio, sr = load_audio_for_split(source_audio)
    intervals = split_intervals(
        audio=audio,
        sr=sr,
        top_db=top_db,
        merge_gap_sec=merge_gap_sec,
        pad_sec=pad_sec,
        min_clip_sec=min_clip_sec,
        max_clip_sec=max_clip_sec,
    )
    clips_dir = run_dir / "clips" / source_row["source_group"] / slugify(source_row["source_title"])
    clips_dir.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(intervals, start=1):
        clip_name = f"{slugify(source_row['source_title'])}_clip_{index:04d}.wav"
        clip_path = clips_dir / clip_name
        clip_audio = torch.from_numpy(audio[start:end]).unsqueeze(0)
        torchaudio.save(str(clip_path), clip_audio, sr, encoding="PCM_S", bits_per_sample=16)
        segment_rows.append(
            {
                "segment_id": f"{source_row['source_id']}_{index:04d}",
                "source_id": source_row["source_id"],
                "source_title": source_row["source_title"],
                "source_group": source_row["source_group"],
                "source_kind": source_row["source_kind"],
                "segment_index": index,
                "clip_audio_posix": clip_path.resolve().as_posix(),
                "clip_audio_windows": to_windows_path(clip_path),
                "clip_audio_relpath": clip_path.resolve().relative_to(run_dir).as_posix(),
                "source_audio_posix": source_row["source_audio_posix"],
                "source_audio_windows": source_row["source_audio_windows"],
                "source_audio_relpath": source_row["source_audio_relpath"],
                "start_sec": start / sr,
                "end_sec": end / sr,
                "duration_sec": (end - start) / sr,
                "split_method": source_row["split_method"],
                "split_params": (
                    f"top_db={top_db},merge_gap_sec={merge_gap_sec},pad_sec={pad_sec},min_clip_sec={min_clip_sec},max_clip_sec={max_clip_sec}"
                ),
            }
        )
    return segment_rows


def decode_ctc_ids(
    predicted_ids: torch.Tensor,
    id_to_token: dict[int, str],
    blank_id: int,
    word_delimiter_token: str,
) -> str:
    pieces: list[str] = []
    prev = None
    for item in predicted_ids.tolist():
        if item == prev:
            continue
        prev = item
        if item == blank_id:
            continue
        token = id_to_token.get(item, "")
        if not token or token.startswith("<"):
            continue
        if token == word_delimiter_token:
            pieces.append(" ")
        else:
            pieces.append(token)
    return normalize_ipa_text("".join(pieces))


def load_ipa_bundle(model_dir: Path, device_name: str, decoder_mode: str, beam_width: int) -> dict[str, Any]:
    torch_dtype = choose_torch_dtype(device_name, "auto")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), **model_kwargs).to(device_name).eval()

    model_config = load_local_model_config(model_dir)
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    tokenizer_config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    word_delimiter_token = tokenizer_config.get("word_delimiter_token", "|")
    resolved_decode_mode = decoder_mode
    if resolved_decode_mode == "auto":
        resolved_decode_mode = str(model_config.get("preferred_decoder") or "beam").strip().lower()

    decoder = None
    if resolved_decode_mode == "beam":
        try:
            from pyctcdecode import build_ctcdecoder
        except ModuleNotFoundError:
            print("pyctcdecode not installed; falling back to greedy IPA decode.", flush=True)
            resolved_decode_mode = "greedy"
        else:
            decoder = build_ctcdecoder(
                build_pyctc_labels(
                    vocab,
                    word_delimiter_token,
                    vocab_size=int(model.config.vocab_size),
                    added_tokens_decoder=tokenizer_config.get("added_tokens_decoder"),
                )
            )

    return {
        "feature_extractor": feature_extractor,
        "model": model,
        "device": torch.device(device_name),
        "feature_dtype": resolve_feature_dtype(model),
        "id_to_token": {token_id: token for token, token_id in vocab.items()},
        "blank_id": int(model.config.pad_token_id),
        "word_delimiter_token": word_delimiter_token,
        "decoder_mode": resolved_decode_mode,
        "decoder": decoder,
        "beam_width": int(model_config.get("beam_width") or beam_width),
        "redentalize_output": bool(model_config.get("redentalize_output")),
    }


def load_text_bundle(model_dir: Path, device_name: str) -> dict[str, Any]:
    torch_dtype = choose_torch_dtype(device_name, "auto")
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model_kwargs: dict[str, Any] = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), **model_kwargs).to(device_name).eval()
    return {
        "processor": processor,
        "model": model,
        "device": torch.device(device_name),
        "feature_dtype": resolve_feature_dtype(model),
    }


def transcribe_ipa_segments(
    segment_rows: list[dict[str, Any]],
    model_dir: Path,
    device_name: str,
    decoder_mode: str,
    beam_width: int,
    batch_size: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    bundle = load_ipa_bundle(model_dir, device_name, decoder_mode, beam_width)
    prediction_rows: list[dict[str, Any]] = []
    target_sr = int(bundle["feature_extractor"].sampling_rate)

    with torch.no_grad():
        for start in range(0, len(segment_rows), batch_size):
            batch_rows = segment_rows[start : start + batch_size]
            audio_arrays = [load_clip_audio(Path(str(row["clip_audio_posix"])), target_sr) for row in batch_rows]
            inputs = bundle["feature_extractor"](
                audio_arrays,
                sampling_rate=target_sr,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs["input_values"].to(bundle["device"])
            if bundle.get("feature_dtype") is not None:
                input_values = input_values.to(dtype=bundle["feature_dtype"])
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(bundle["device"])

            logits = bundle["model"](input_values=input_values, attention_mask=attention_mask).logits
            for offset, row in enumerate(batch_rows):
                if bundle.get("decoder") is not None:
                    prediction = normalize_ipa_text(
                        clean_beam_decoded_text(
                            bundle["decoder"].decode(
                                logits[offset].float().cpu().numpy(),
                                beam_width=bundle["beam_width"],
                            )
                        ),
                        redentalize_stops=bundle["redentalize_output"],
                    )
                else:
                    prediction = normalize_ipa_text(
                        decode_ctc_ids(
                            predicted_ids=torch.argmax(logits[offset], dim=-1).cpu(),
                            id_to_token=bundle["id_to_token"],
                            blank_id=bundle["blank_id"],
                            word_delimiter_token=bundle["word_delimiter_token"],
                        ),
                        redentalize_stops=bundle["redentalize_output"],
                    )
                prediction_rows.append(
                    {
                        "segment_id": row["segment_id"],
                        "predicted_ipa": prediction,
                        "predicted_ipa_canonical": ipa_to_canonical(prediction) if prediction else "",
                    }
                )

            if ((start // batch_size) + 1) % 25 == 0 or (start + batch_size) >= len(segment_rows):
                write_tsv(output_path, prediction_rows, ["segment_id", "predicted_ipa", "predicted_ipa_canonical"])
            print(f"IPA decoded {min(start + batch_size, len(segment_rows))}/{len(segment_rows)}", flush=True)

    return prediction_rows


def transcribe_text_segments(
    segment_rows: list[dict[str, Any]],
    model_dir: Path,
    device_name: str,
    batch_size: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    bundle = load_text_bundle(model_dir, device_name)
    processor = bundle["processor"]
    target_sr = int(processor.feature_extractor.sampling_rate)
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(segment_rows), batch_size):
            batch_rows = segment_rows[start : start + batch_size]
            audio_arrays = [load_clip_audio(Path(str(row["clip_audio_posix"])), target_sr) for row in batch_rows]
            inputs = processor.feature_extractor(
                audio_arrays,
                sampling_rate=target_sr,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_values = inputs["input_values"].to(bundle["device"])
            if bundle.get("feature_dtype") is not None:
                input_values = input_values.to(dtype=bundle["feature_dtype"])
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(bundle["device"])

            logits = bundle["model"](input_values=input_values, attention_mask=attention_mask).logits
            pred_ids = torch.argmax(logits, dim=-1)
            raw_predictions = processor.batch_decode(pred_ids)

            for row, raw_prediction in zip(batch_rows, raw_predictions):
                raw_text = normalize_plain_text(raw_prediction.replace("|", " "))
                normalized_text = normalize_simple_text(raw_prediction)
                prediction_rows.append(
                    {
                        "segment_id": row["segment_id"],
                        "predicted_orthography_raw": raw_text,
                        "predicted_orthography": normalized_text,
                        "predicted_orthography_canonical": (
                            orthography_to_canonical(normalized_text) if normalized_text else ""
                        ),
                    }
                )

            if ((start // batch_size) + 1) % 25 == 0 or (start + batch_size) >= len(segment_rows):
                write_tsv(
                    output_path,
                    prediction_rows,
                    [
                        "segment_id",
                        "predicted_orthography_raw",
                        "predicted_orthography",
                        "predicted_orthography_canonical",
                    ],
                )
            print(f"Orthography decoded {min(start + batch_size, len(segment_rows))}/{len(segment_rows)}", flush=True)

    return prediction_rows


def build_combined_rows(
    segment_rows: list[dict[str, Any]],
    ipa_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ipa_map = {str(row["segment_id"]): row for row in ipa_rows}
    text_map = {str(row["segment_id"]): row for row in text_rows}
    combined: list[dict[str, Any]] = []
    for row in segment_rows:
        segment_id = str(row["segment_id"])
        ipa_row = ipa_map.get(segment_id, {})
        text_row = text_map.get(segment_id, {})
        combined.append(
            {
                **row,
                "predicted_ipa": str(ipa_row.get("predicted_ipa") or ""),
                "predicted_ipa_canonical": str(ipa_row.get("predicted_ipa_canonical") or ""),
                "predicted_orthography_raw": str(text_row.get("predicted_orthography_raw") or ""),
                "predicted_orthography": str(text_row.get("predicted_orthography") or ""),
                "predicted_orthography_canonical": str(text_row.get("predicted_orthography_canonical") or ""),
            }
        )
    return combined


def build_joined_source_rows(combined_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined_rows:
        grouped[str(row["source_id"])].append(row)

    joined_rows: list[dict[str, Any]] = []
    for source_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["segment_index"]))
        first = rows[0]
        joined_rows.append(
            {
                "source_id": source_id,
                "source_title": first["source_title"],
                "source_group": first["source_group"],
                "source_kind": first["source_kind"],
                "source_audio_posix": first["source_audio_posix"],
                "source_audio_windows": first["source_audio_windows"],
                "source_audio_relpath": first["source_audio_relpath"],
                "segment_count": len(rows),
                "total_segment_sec": round(sum(float(row["duration_sec"]) for row in rows), 3),
                "joined_orthography": " ".join(row["predicted_orthography"] for row in rows if row["predicted_orthography"]),
                "joined_ipa": " ".join(row["predicted_ipa"] for row in rows if row["predicted_ipa"]),
            }
        )

    return sorted(joined_rows, key=lambda row: (str(row["source_group"]), str(row["source_title"])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe all unique C.K. audio into both IPA and orthography using the current "
            "C.K.-adapted MMS/Wav2Vec2 models, with segment-level outputs for later mining."
        )
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "transcription_outputs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--ipa-model-dir", default=default_ipa_model_dir())
    parser.add_argument("--orth-model-dir", default=default_orth_model_dir())
    parser.add_argument("--ipa-decoder", default="auto", choices=["auto", "greedy", "beam"])
    parser.add_argument("--ipa-beam-width", type=int, default=64)
    parser.add_argument("--ipa-batch-size", type=int, default=8)
    parser.add_argument("--orth-batch-size", type=int, default=8)
    parser.add_argument("--top-db", type=float, default=30.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.30)
    parser.add_argument("--pad-sec", type=float, default=0.08)
    parser.add_argument("--min-clip-sec", type=float, default=0.25)
    parser.add_argument("--max-clip-sec", type=float, default=20.0)
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name.strip() or f"ck_corpus_dual_ctc_{timestamp_slug()}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    device_name = choose_device(args.device)
    run_dir = make_run_dir(args)

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Device: {device_name}", flush=True)

    source_rows = collect_unique_full_recordings()
    phase1a_source_row, phase1a_segments = build_phase1a_segments()
    source_rows.append(phase1a_source_row)

    phase1b_source_row = {
        "source_id": "phase1b_processed",
        "source_title": "phase1b processed",
        "source_group": "quizlet_phase1b",
        "source_kind": "phase_processed_split",
        "source_audio_posix": PHASE1B_AUDIO.resolve().as_posix(),
        "source_audio_windows": to_windows_path(PHASE1B_AUDIO),
        "source_audio_relpath": repo_relpath(PHASE1B_AUDIO),
        "split_method": "librosa_split",
    }
    source_rows.append(phase1b_source_row)

    segment_rows: list[dict[str, Any]] = []
    segment_rows.extend(phase1a_segments)

    for source_row in source_rows:
        if source_row["source_id"] == "phase1a_processed":
            continue
        print(f"Splitting {source_row['source_title']} ({source_row['source_group']})", flush=True)
        segment_rows.extend(
            split_source_to_segments(
                source_row=source_row,
                run_dir=run_dir,
                top_db=args.top_db,
                merge_gap_sec=args.merge_gap_sec,
                pad_sec=args.pad_sec,
                min_clip_sec=args.min_clip_sec,
                max_clip_sec=args.max_clip_sec,
            )
        )

    segment_rows = sorted(segment_rows, key=lambda row: (str(row["source_group"]), str(row["source_title"]), int(row["segment_index"])))

    source_counter = Counter(row["source_group"] for row in source_rows)
    segment_counter = Counter(row["source_group"] for row in segment_rows)

    segment_inventory_path = run_dir / "ck_segment_inventory.tsv"
    source_inventory_path = run_dir / "ck_source_inventory.tsv"
    ipa_predictions_path = run_dir / "ck_ipa_predictions.tsv"
    orth_predictions_path = run_dir / "ck_orthography_predictions.tsv"
    combined_tsv_path = run_dir / "ck_dual_transcriptions.tsv"
    combined_jsonl_path = run_dir / "ck_dual_transcriptions.jsonl"
    joined_sources_path = run_dir / "ck_joined_by_source.tsv"
    summary_path = run_dir / "summary.json"
    readme_path = run_dir / "README.md"

    write_tsv(
        segment_inventory_path,
        segment_rows,
        [
            "segment_id",
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "segment_index",
            "clip_audio_posix",
            "clip_audio_windows",
            "clip_audio_relpath",
            "source_audio_posix",
            "source_audio_windows",
            "source_audio_relpath",
            "start_sec",
            "end_sec",
            "duration_sec",
            "split_method",
            "split_params",
        ],
    )

    source_summary_rows = []
    grouped_segments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in segment_rows:
        grouped_segments[str(row["source_id"])].append(row)
    for row in source_rows:
        items = grouped_segments.get(str(row["source_id"]), [])
        source_summary_rows.append(
            {
                **row,
                "segment_count": len(items),
                "total_segment_sec": round(sum(float(item["duration_sec"]) for item in items), 3),
            }
        )
    write_tsv(
        source_inventory_path,
        source_summary_rows,
        [
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "source_audio_posix",
            "source_audio_windows",
            "source_audio_relpath",
            "split_method",
            "segment_count",
            "total_segment_sec",
        ],
    )

    ipa_rows = transcribe_ipa_segments(
        segment_rows=segment_rows,
        model_dir=Path(args.ipa_model_dir).expanduser().resolve(),
        device_name=device_name,
        decoder_mode=args.ipa_decoder,
        beam_width=args.ipa_beam_width,
        batch_size=args.ipa_batch_size,
        output_path=ipa_predictions_path,
    )
    orth_rows = transcribe_text_segments(
        segment_rows=segment_rows,
        model_dir=Path(args.orth_model_dir).expanduser().resolve(),
        device_name=device_name,
        batch_size=args.orth_batch_size,
        output_path=orth_predictions_path,
    )

    combined_rows = build_combined_rows(segment_rows, ipa_rows, orth_rows)
    write_tsv(
        combined_tsv_path,
        combined_rows,
        [
            "segment_id",
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "segment_index",
            "clip_audio_posix",
            "clip_audio_windows",
            "clip_audio_relpath",
            "source_audio_posix",
            "source_audio_windows",
            "source_audio_relpath",
            "start_sec",
            "end_sec",
            "duration_sec",
            "split_method",
            "split_params",
            "predicted_ipa",
            "predicted_ipa_canonical",
            "predicted_orthography_raw",
            "predicted_orthography",
            "predicted_orthography_canonical",
        ],
    )
    write_jsonl(combined_jsonl_path, combined_rows)

    joined_source_rows = build_joined_source_rows(combined_rows)
    write_tsv(
        joined_sources_path,
        joined_source_rows,
        [
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "source_audio_posix",
            "source_audio_windows",
            "source_audio_relpath",
            "segment_count",
            "total_segment_sec",
            "joined_orthography",
            "joined_ipa",
        ],
    )

    summary = {
        "run_dir": str(run_dir),
        "device": device_name,
        "ipa_model_dir": str(Path(args.ipa_model_dir).expanduser().resolve()),
        "orth_model_dir": str(Path(args.orth_model_dir).expanduser().resolve()),
        "ipa_decoder_mode": args.ipa_decoder,
        "ipa_beam_width": args.ipa_beam_width,
        "source_count": len(source_rows),
        "segment_count": len(segment_rows),
        "source_group_counts": dict(source_counter),
        "segment_group_counts": dict(segment_counter),
        "total_segment_sec": round(sum(float(row["duration_sec"]) for row in segment_rows), 3),
        "outputs": {
            "source_inventory_tsv": str(source_inventory_path),
            "segment_inventory_tsv": str(segment_inventory_path),
            "ipa_predictions_tsv": str(ipa_predictions_path),
            "orthography_predictions_tsv": str(orth_predictions_path),
            "dual_tsv": str(combined_tsv_path),
            "dual_jsonl": str(combined_jsonl_path),
            "joined_by_source_tsv": str(joined_sources_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    readme_path.write_text(
        "\n".join(
            [
                "# C.K. Dual Transcription Run",
                "",
                "This folder contains a segment-level C.K. corpus transcription in both IPA and orthography.",
                "",
                "Main files:",
                "- `ck_source_inventory.tsv`: one row per unique source audio",
                "- `ck_segment_inventory.tsv`: one row per segment/clip",
                "- `ck_ipa_predictions.tsv`: IPA predictions keyed by `segment_id`",
                "- `ck_orthography_predictions.tsv`: orthography predictions keyed by `segment_id`",
                "- `ck_dual_transcriptions.tsv`: unified segment table with metadata + both predictions",
                "- `ck_dual_transcriptions.jsonl`: JSONL version of the unified table",
                "- `ck_joined_by_source.tsv`: per-source concatenated orthography and IPA views",
                "",
                "Notes:",
                "- `phase1a` reused the existing split clips",
                "- `phase1b` and full recordings were split by silence into WAV clips under `clips/`",
                "- orthography is in the simplified training target space of the current C.K.-adapted orthography model",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
