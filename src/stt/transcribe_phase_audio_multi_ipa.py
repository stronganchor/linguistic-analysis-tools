#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForCTC,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from prepare_lltools_ipa_dataset import normalize_ipa_text, to_windows_path
from review_lltools_ipa_comparison import ipa_to_canonical
from serve_local_stt_mms import (
    TARGET_SR,
    build_pyctc_labels,
    choose_torch_dtype,
    clean_beam_decoded_text,
    load_local_model_config,
    resolve_feature_dtype,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def default_mms_model_dir() -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir / "models" / "alternatives" / "mms-1b-all-ipa-warmstart-v003-20260407-124628")


def default_whisper_model_dir() -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir / "models" / "high-accuracy" / "whisper-medium-ipa-nostress-genc-palu-warmstart-v004-20260406")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Split a long phase-style recording into short clips, run MMS IPA and Whisper IPA "
            "on CPU, then write comparison and review artifacts."
        )
    )
    parser.add_argument("--audio", required=True, help="Input audio file.")
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "transcription_outputs"),
        help="Parent directory for the run folder.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional explicit run folder name. Defaults to <audio-stem>_multi_ipa_cpu_<timestamp>.",
    )
    parser.add_argument(
        "--split-tsv",
        default="",
        help="Optional existing split manifest TSV to reuse instead of re-splitting the source audio.",
    )
    parser.add_argument(
        "--mms-predictions-tsv",
        default="",
        help="Optional existing MMS predictions TSV to reuse instead of rerunning MMS.",
    )
    parser.add_argument(
        "--whisper-predictions-tsv",
        default="",
        help="Optional existing Whisper predictions TSV to reuse instead of rerunning Whisper.",
    )
    parser.add_argument("--device", default="cpu", help="Use cpu or cuda.")
    parser.add_argument("--top-db", type=float, default=30.0, help="Silence split threshold.")
    parser.add_argument("--merge-gap-sec", type=float, default=0.30, help="Merge gaps up to this many seconds.")
    parser.add_argument("--pad-sec", type=float, default=0.08, help="Padding added to both sides of each clip.")
    parser.add_argument("--min-clip-sec", type=float, default=0.25, help="Drop clips shorter than this.")
    parser.add_argument("--mms-model-dir", default=default_mms_model_dir())
    parser.add_argument(
        "--mms-decoder",
        choices=["auto", "greedy", "beam"],
        default="auto",
        help="CTC decode mode. Auto uses lltools_ctc_config.json when present.",
    )
    parser.add_argument("--mms-beam-width", type=int, default=64)
    parser.add_argument("--mms-batch-size", type=int, default=4)
    parser.add_argument("--whisper-model-dir", default=default_whisper_model_dir())
    parser.add_argument("--whisper-batch-size", type=int, default=4)
    parser.add_argument("--whisper-num-beams", type=int, default=4)
    parser.add_argument("--whisper-max-length", type=int, default=128)
    return parser.parse_args()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(args: argparse.Namespace, audio_path: Path) -> Path:
    if args.split_tsv:
        split_tsv = Path(args.split_tsv).expanduser().resolve()
        if not split_tsv.is_file():
            raise SystemExit(f"Missing split TSV: {split_tsv}")
        if not args.run_name.strip():
            return split_tsv.parent
    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"{audio_path.stem}_multi_ipa_cpu_{timestamp_slug()}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def choose_device(value: str) -> str:
    requested = (value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_audio_for_split(audio_path: Path) -> tuple[np.ndarray, int]:
    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    return audio, sr


def split_intervals(
    audio: np.ndarray,
    sr: int,
    top_db: float,
    merge_gap_sec: float,
    pad_sec: float,
    min_clip_sec: float,
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
    clipped: list[tuple[int, int]] = []
    total_samples = len(audio)
    for start, end in merged:
        padded_start = max(0, start - pad_samples)
        padded_end = min(total_samples, end + pad_samples)
        if padded_end - padded_start < min_clip_samples:
            continue
        clipped.append((padded_start, padded_end))
    return clipped


def save_split_clips(
    audio: np.ndarray,
    sr: int,
    intervals: list[tuple[int, int]],
    clips_dir: Path,
    source_audio: Path,
    run_dir: Path,
) -> list[dict[str, Any]]:
    clips_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stem = source_audio.stem.replace(" ", "_")
    for index, (start, end) in enumerate(intervals, start=1):
        clip_name = f"{stem}_clip_{index:04d}.wav"
        clip_path = clips_dir / clip_name
        clip_audio = torch.from_numpy(audio[start:end]).unsqueeze(0)
        torchaudio.save(str(clip_path), clip_audio, sr, encoding="PCM_S", bits_per_sample=16)
        rows.append(
            {
                "clip_id": index,
                "audio_posix": clip_path.resolve().as_posix(),
                "audio_windows": to_windows_path(clip_path),
                "audio_relpath": clip_path.resolve().relative_to(run_dir).as_posix(),
                "audio_filename": clip_path.name,
                "source_audio_posix": source_audio.resolve().as_posix(),
                "source_audio_windows": to_windows_path(source_audio),
                "start_sec": start / sr,
                "end_sec": end / sr,
                "duration_sec": (end - start) / sr,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_audio_array(audio_path: Path, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.numpy()


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


def load_mms_bundle(model_dir: Path, device_name: str, decoder_mode: str, beam_width: int) -> dict[str, Any]:
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
            print("pyctcdecode is unavailable; falling back to greedy decode for MMS.", flush=True)
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


def transcribe_mms_rows(
    clip_rows: list[dict[str, Any]],
    model_dir: Path,
    device_name: str,
    decoder_mode: str,
    beam_width: int,
    batch_size: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    bundle = load_mms_bundle(model_dir, device_name, decoder_mode, beam_width)
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(clip_rows), batch_size):
            batch_rows = clip_rows[start : start + batch_size]
            audio_arrays = [load_audio_array(Path(str(row["audio_posix"])), TARGET_SR) for row in batch_rows]
            inputs = bundle["feature_extractor"](
                audio_arrays,
                sampling_rate=TARGET_SR,
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
                prediction = ""
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
                        "clip_id": int(row["clip_id"]),
                        "audio_posix": row["audio_posix"],
                        "predicted_ipa": prediction,
                    }
                )

            if ((start // batch_size) + 1) % 25 == 0 or (start + batch_size) >= len(clip_rows):
                write_tsv(output_path, prediction_rows, ["clip_id", "audio_posix", "predicted_ipa"])
            print(
                f"MMS decoded {min(start + batch_size, len(clip_rows))}/{len(clip_rows)} clips",
                flush=True,
            )

    return prediction_rows


def load_whisper_bundle(model_dir: Path, device_name: str) -> dict[str, Any]:
    torch_dtype = choose_torch_dtype(device_name, "auto")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(model_dir))
    tokenizer = WhisperTokenizer.from_pretrained(str(model_dir))
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir), **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []

    model = model.to(device_name).eval()
    return {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": torch.device(device_name),
        "feature_dtype": resolve_feature_dtype(model),
    }


def transcribe_whisper_rows(
    clip_rows: list[dict[str, Any]],
    model_dir: Path,
    device_name: str,
    batch_size: int,
    num_beams: int,
    max_length: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    bundle = load_whisper_bundle(model_dir, device_name)
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(clip_rows), batch_size):
            batch_rows = clip_rows[start : start + batch_size]
            audio_arrays = [load_audio_array(Path(str(row["audio_posix"])), TARGET_SR) for row in batch_rows]
            batch = bundle["feature_extractor"](
                audio_arrays,
                sampling_rate=TARGET_SR,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = batch["input_features"].to(bundle["device"])
            if bundle.get("feature_dtype") is not None:
                input_features = input_features.to(dtype=bundle["feature_dtype"])

            generate_kwargs: dict[str, Any] = {
                "input_features": input_features,
                "max_length": max(32, max_length),
                "num_beams": max(1, num_beams),
            }
            if "attention_mask" in batch:
                generate_kwargs["attention_mask"] = batch["attention_mask"].to(bundle["device"])

            generated_ids = bundle["model"].generate(**generate_kwargs)
            predictions = bundle["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)

            for row, prediction in zip(batch_rows, predictions):
                prediction_rows.append(
                    {
                        "clip_id": int(row["clip_id"]),
                        "audio_posix": row["audio_posix"],
                        "predicted_ipa": normalize_ipa_text(prediction),
                    }
                )

            if ((start // batch_size) + 1) % 25 == 0 or (start + batch_size) >= len(clip_rows):
                write_tsv(output_path, prediction_rows, ["clip_id", "audio_posix", "predicted_ipa"])
            print(
                f"Whisper decoded {min(start + batch_size, len(clip_rows))}/{len(clip_rows)} clips",
                flush=True,
            )

    return prediction_rows


def choose_bucket_surface(model_to_surface: dict[str, str]) -> str:
    for model_name in ("new_mms", "whisper"):
        surface = model_to_surface.get(model_name, "")
        if surface:
            return surface
    for surface in model_to_surface.values():
        if surface:
            return surface
    return ""


def build_outputs(
    clip_rows: list[dict[str, Any]],
    mms_rows: list[dict[str, Any]],
    whisper_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    mms_map = {int(row["clip_id"]): normalize_ipa_text(str(row.get("predicted_ipa") or "")) for row in mms_rows}
    whisper_map = {int(row["clip_id"]): normalize_ipa_text(str(row.get("predicted_ipa") or "")) for row in whisper_rows}

    comparison_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stats = Counter()

    for clip_row in clip_rows:
        clip_id = int(clip_row["clip_id"])
        mms_ipa = mms_map.get(clip_id, "")
        whisper_ipa = whisper_map.get(clip_id, "")
        mms_canonical = ipa_to_canonical(mms_ipa) if mms_ipa else ""
        whisper_canonical = ipa_to_canonical(whisper_ipa) if whisper_ipa else ""

        exact_agree = bool(mms_ipa and whisper_ipa and mms_ipa == whisper_ipa)
        canonical_agree = bool(mms_canonical and whisper_canonical and mms_canonical == whisper_canonical)
        auto_accept_ipa = mms_ipa if exact_agree else ""

        stats["clips_total"] += 1
        if mms_ipa:
            stats["mms_nonempty"] += 1
        if whisper_ipa:
            stats["whisper_nonempty"] += 1
        if exact_agree:
            stats["exact_agree"] += 1
        if canonical_agree:
            stats["canonical_agree"] += 1
        if auto_accept_ipa:
            stats["auto_accept_exact"] += 1

        buckets_by_canonical: dict[str, dict[str, Any]] = {}
        for model_name, prediction in (("new_mms", mms_ipa), ("whisper", whisper_ipa)):
            if not prediction:
                continue
            canonical = ipa_to_canonical(prediction) or prediction
            bucket = buckets_by_canonical.setdefault(
                canonical,
                {
                    "canonical": canonical,
                    "models": [],
                    "model_to_surface": {},
                },
            )
            bucket["models"].append(model_name)
            bucket["model_to_surface"][model_name] = prediction

        candidate_buckets: list[dict[str, Any]] = []
        for bucket in sorted(
            buckets_by_canonical.values(),
            key=lambda item: (-len(item["models"]), item["canonical"]),
        ):
            candidate_buckets.append(
                {
                    "canonical": bucket["canonical"],
                    "chosen_ipa": choose_bucket_surface(bucket["model_to_surface"]),
                    "support_count": len(bucket["models"]),
                    "models": list(bucket["models"]),
                    "best_rule_count": 0,
                    "csv_support_any": False,
                    "csv_support_typed": False,
                    "supports_new_mms": "new_mms" in bucket["models"],
                    "supports_old_mms": False,
                    "supports_whisper": "whisper" in bucket["models"],
                }
            )

        comparison_rows.append(
            {
                "clip_id": clip_id,
                "audio_filename": clip_row["audio_filename"],
                "start_sec": f"{float(clip_row['start_sec']):.3f}",
                "end_sec": f"{float(clip_row['end_sec']):.3f}",
                "duration_sec": f"{float(clip_row['duration_sec']):.3f}",
                "new_mms_ipa": mms_ipa,
                "whisper_ipa": whisper_ipa,
                "new_mms_canonical": mms_canonical,
                "whisper_canonical": whisper_canonical,
                "exact_agree": "yes" if exact_agree else "no",
                "canonical_agree": "yes" if canonical_agree else "no",
                "auto_accept_ipa": auto_accept_ipa,
            }
        )

        prediction_rows.append(
            {
                "recording_id": clip_id,
                "word_id": clip_id,
                "word_title": clip_row["audio_filename"],
                "word_translation": "",
                "recording_type": "isolation_auto",
                "audio_posix": clip_row["audio_posix"],
                "audio_windows": clip_row["audio_windows"],
                "audio_relpath": clip_row["audio_relpath"],
                "audio_filename": clip_row["audio_filename"],
                "source_bundle": "phase1a_auto_split",
                "category_name": "Quizlet phase1a",
                "category_slug": "quizlet-phase1a",
                "speaker_user_id": 0,
                "speaker_name": "C.K.",
                "word_slug": "",
                "recording_slug": f"phase1a-clip-{clip_id:04d}",
                "start_sec": float(clip_row["start_sec"]),
                "end_sec": float(clip_row["end_sec"]),
                "duration_sec": float(clip_row["duration_sec"]),
            }
        )

        audit_rows.append(
            {
                "word_id": clip_id,
                "recording_id": clip_id,
                "word_title": clip_row["audio_filename"],
                "word_translation": "",
                "recording_type": "isolation_auto",
                "current_recording_ipa": "",
                "current_rule_count": 0,
                "current_canonical": "",
                "csv_candidates": [],
                "new_mms_recording_ipa": mms_ipa,
                "old_mms_recording_ipa": "",
                "whisper_recording_ipa": whisper_ipa,
                "candidate_buckets": candidate_buckets,
                "accepted": bool(auto_accept_ipa),
                "accepted_ipa": auto_accept_ipa,
                "exact_agree": exact_agree,
                "canonical_agree": canonical_agree,
            }
        )

    return comparison_rows, prediction_rows, audit_rows, dict(stats)


def build_auto_accept_rows(
    clip_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clip_map = {int(row["clip_id"]): row for row in clip_rows}
    accepted_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        accepted_ipa = str(row.get("auto_accept_ipa") or "").strip()
        if not accepted_ipa:
            continue
        clip_id = int(row["clip_id"])
        clip_row = clip_map[clip_id]
        accepted_rows.append(
            {
                "recording_id": clip_id,
                "audio_posix": clip_row["audio_posix"],
                "audio_windows": clip_row["audio_windows"],
                "audio_relpath": clip_row["audio_relpath"],
                "accepted_ipa": accepted_ipa,
                "consensus_type": "exact_two_model_match",
            }
        )
    return accepted_rows


def build_priority_review_rows(
    comparison_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clip_map = {int(row["clip_id"]): row for row in clip_rows}
    priority_rows: list[dict[str, Any]] = []
    for row in comparison_rows:
        if row["exact_agree"] == "yes":
            continue
        if row["canonical_agree"] != "yes":
            continue
        clip_id = int(row["clip_id"])
        clip_row = clip_map[clip_id]
        priority_rows.append(
            {
                "recording_id": clip_id,
                "audio_posix": clip_row["audio_posix"],
                "audio_windows": clip_row["audio_windows"],
                "audio_relpath": clip_row["audio_relpath"],
                "new_mms_ipa": row["new_mms_ipa"],
                "whisper_ipa": row["whisper_ipa"],
                "note": "canonical match but surface mismatch",
            }
        )
    return priority_rows


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Missing audio file: {audio_path}")

    device_name = choose_device(args.device)
    run_dir = make_run_dir(args, audio_path)
    clips_dir = run_dir / "clips"
    audio: np.ndarray | None = None
    sr = TARGET_SR
    split_tsv_path = Path(args.split_tsv).expanduser().resolve() if args.split_tsv else (run_dir / "phase_split.tsv")
    if args.split_tsv:
        print(f"Reusing split manifest: {split_tsv_path}", flush=True)
        clip_rows = read_tsv(split_tsv_path)
        if not clip_rows:
            raise SystemExit(f"No clip rows found in {split_tsv_path}")
    else:
        print(f"Loading and splitting: {audio_path}", flush=True)
        audio, sr = load_audio_for_split(audio_path)
        intervals = split_intervals(
            audio=audio,
            sr=sr,
            top_db=args.top_db,
            merge_gap_sec=args.merge_gap_sec,
            pad_sec=args.pad_sec,
            min_clip_sec=args.min_clip_sec,
        )
        if not intervals:
            raise SystemExit("No clips found after silence splitting.")

        clip_rows = save_split_clips(
            audio=audio,
            sr=sr,
            intervals=intervals,
            clips_dir=clips_dir,
            source_audio=audio_path,
            run_dir=run_dir,
        )
        write_tsv(
            split_tsv_path,
            clip_rows,
            [
                "clip_id",
                "audio_posix",
                "audio_windows",
                "audio_relpath",
                "audio_filename",
                "source_audio_posix",
                "source_audio_windows",
                "start_sec",
                "end_sec",
                "duration_sec",
            ],
        )
        print(f"Saved {len(clip_rows)} clips to {clips_dir}", flush=True)

    mms_path = run_dir / "mms_v003_predictions.tsv"
    whisper_path = run_dir / "whisper_medium_v004_predictions.tsv"
    if args.mms_predictions_tsv:
        mms_path = Path(args.mms_predictions_tsv).expanduser().resolve()
        print(f"Reusing MMS predictions: {mms_path}", flush=True)
        mms_rows = read_tsv(mms_path)
    else:
        mms_rows = transcribe_mms_rows(
            clip_rows=clip_rows,
            model_dir=Path(args.mms_model_dir).expanduser().resolve(),
            device_name=device_name,
            decoder_mode=args.mms_decoder,
            beam_width=args.mms_beam_width,
            batch_size=args.mms_batch_size,
            output_path=mms_path,
        )
    if args.whisper_predictions_tsv:
        whisper_path = Path(args.whisper_predictions_tsv).expanduser().resolve()
        print(f"Reusing Whisper predictions: {whisper_path}", flush=True)
        whisper_rows = read_tsv(whisper_path)
    else:
        whisper_rows = transcribe_whisper_rows(
            clip_rows=clip_rows,
            model_dir=Path(args.whisper_model_dir).expanduser().resolve(),
            device_name=device_name,
            batch_size=args.whisper_batch_size,
            num_beams=args.whisper_num_beams,
            max_length=args.whisper_max_length,
            output_path=whisper_path,
        )

    comparison_rows, prediction_rows, audit_rows, stats = build_outputs(clip_rows, mms_rows, whisper_rows)
    comparison_path = run_dir / "phase1a_mms_vs_whisper_compare.tsv"
    predictions_path = run_dir / "phase1a_review_predictions.jsonl"
    audit_path = run_dir / "phase1a_review_audit.jsonl"
    auto_accept_path = run_dir / "phase1a_high_confidence_exact.jsonl"
    priority_review_path = run_dir / "phase1a_canonical_agree_review_priority.jsonl"

    write_tsv(
        comparison_path,
        comparison_rows,
        [
            "clip_id",
            "audio_filename",
            "start_sec",
            "end_sec",
            "duration_sec",
            "new_mms_ipa",
            "whisper_ipa",
            "new_mms_canonical",
            "whisper_canonical",
            "exact_agree",
            "canonical_agree",
            "auto_accept_ipa",
        ],
    )
    write_jsonl(predictions_path, prediction_rows)
    write_jsonl(audit_path, audit_rows)
    write_jsonl(auto_accept_path, build_auto_accept_rows(clip_rows, comparison_rows))
    write_jsonl(priority_review_path, build_priority_review_rows(comparison_rows, clip_rows))

    if audio is None:
        waveform, source_sr = torchaudio.load(str(audio_path))
        duration_seconds = float(waveform.shape[-1] / source_sr)
    else:
        duration_seconds = float(len(audio) / sr)
    total_clip_audio_seconds = float(sum(float(row["duration_sec"]) for row in clip_rows))
    review_command = (
        f"python3 '{(Path(__file__).resolve().parent / 'serve_unresolved_ipa_review.py').as_posix()}' "
        f"--audit-path '{audit_path.as_posix()}' "
        f"--predictions-path '{predictions_path.as_posix()}' "
        f"--speaker-user-id 0 --open-browser"
    )
    summary = {
        "audio_path": audio_path.as_posix(),
        "audio_windows": to_windows_path(audio_path),
        "device": device_name,
        "run_dir": run_dir.as_posix(),
        "split": {
            "top_db": args.top_db,
            "merge_gap_sec": args.merge_gap_sec,
            "pad_sec": args.pad_sec,
            "min_clip_sec": args.min_clip_sec,
            "clips": len(clip_rows),
            "duration_seconds": duration_seconds,
            "total_clip_audio_seconds": total_clip_audio_seconds,
        },
        "models": {
            "mms_model_dir": str(Path(args.mms_model_dir).expanduser().resolve()),
            "mms_decoder": args.mms_decoder,
            "mms_beam_width": args.mms_beam_width,
            "whisper_model_dir": str(Path(args.whisper_model_dir).expanduser().resolve()),
            "whisper_num_beams": args.whisper_num_beams,
        },
        "agreement": {
            "clips_total": stats.get("clips_total", 0),
            "mms_nonempty": stats.get("mms_nonempty", 0),
            "whisper_nonempty": stats.get("whisper_nonempty", 0),
            "exact_agree": stats.get("exact_agree", 0),
            "canonical_agree": stats.get("canonical_agree", 0),
            "auto_accept_exact": stats.get("auto_accept_exact", 0),
        },
        "outputs": {
            "split_tsv": split_tsv_path.as_posix(),
            "mms_predictions_tsv": mms_path.as_posix(),
            "whisper_predictions_tsv": whisper_path.as_posix(),
            "comparison_tsv": comparison_path.as_posix(),
            "review_predictions_jsonl": predictions_path.as_posix(),
            "review_audit_jsonl": audit_path.as_posix(),
            "high_confidence_exact_jsonl": auto_accept_path.as_posix(),
            "canonical_agree_review_priority_jsonl": priority_review_path.as_posix(),
        },
        "review_command": review_command,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
