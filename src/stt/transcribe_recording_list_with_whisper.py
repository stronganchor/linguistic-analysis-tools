#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TARGET_SR = 16000
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


def normalize_target_text(value: str, mode: str) -> str:
    if mode == "ipa":
        return normalize_ipa_text(value)
    if mode == "simple_text":
        return normalize_simple_text(value)
    return normalize_plain_text(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a TSV recording list or direct-audio manifest with a local Whisper model."
    )
    parser.add_argument(
        "--input-tsv",
        required=True,
        help="TSV with recording_id and/or direct audio path columns such as audio_posix or audio_windows.",
    )
    parser.add_argument("--model-source", required=True, help="Local Whisper model/checkpoint directory.")
    parser.add_argument(
        "--processor-dir",
        default="",
        help="Optional directory for tokenizer/processor files when using a checkpoint subdirectory.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--target-mode",
        default="ipa",
        choices=["ipa", "plain", "simple_text"],
        help="Normalization mode for decoded outputs.",
    )
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


def choose_torch_dtype(device_name: str, precision: str):
    requested = (precision or "").strip().lower()
    if device_name != "cuda":
        return None
    if requested in ("", "auto", "float16", "fp16", "half"):
        return torch.float16
    if requested in ("bfloat16", "bf16"):
        return torch.bfloat16
    if requested in ("float32", "fp32"):
        return torch.float32
    raise SystemExit(f"Unsupported precision: {precision}")


def resolve_feature_dtype(model) -> torch.dtype | None:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


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


def load_rows(input_tsv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not has_audio_reference(row):
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


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


def has_audio_reference(row: dict[str, str]) -> bool:
    return any(
        str(row.get(key) or "").strip()
        for key in ("recording_id", "audio_posix", "audio_windows", "audio", "local_audio_path")
    )


def resolve_audio_path(row: dict[str, str], rec_audio: dict[str, list[Path]]) -> Path | None:
    for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"):
        resolved = resolve_existing_path(str(row.get(key) or ""))
        if resolved is not None:
            return resolved

    recording_id = str(row.get("recording_id") or "").strip()
    if not recording_id:
        return None
    audio_paths = rec_audio.get(recording_id) or []
    if not audio_paths:
        return None
    return choose_best_audio_path(audio_paths)


def row_label(row: dict[str, str], fallback_index: int) -> str:
    for key in ("recording_id", "clip_id", "source_id", "audio_filename"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(fallback_index)


def load_model_bundle(
    model_source: Path,
    processor_dir: Path | None,
    device_name: str,
    precision: str,
    num_beams: int,
    max_length: int,
) -> dict:
    torch_dtype = choose_torch_dtype(device_name, precision)
    processor_source = processor_dir or model_source
    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(processor_source))
    tokenizer = WhisperTokenizer.from_pretrained(str(processor_source))
    processor = WhisperProcessor.from_pretrained(str(processor_source))
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(str(model_source), **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.num_beams = max(1, num_beams)

    device = torch.device(device_name)
    model.to(device).eval()
    return {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "feature_dtype": resolve_feature_dtype(model),
        "num_beams": max(1, num_beams),
        "max_length": max(32, max_length),
    }


def transcribe_audio_file(bundle: dict, audio_path: Path, target_mode: str) -> str:
    audio_array, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    batch = bundle["feature_extractor"](
        [audio_array],
        sampling_rate=TARGET_SR,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = batch["input_features"].to(bundle["device"])
    if bundle.get("feature_dtype") is not None:
        input_features = input_features.to(dtype=bundle["feature_dtype"])

    generate_kwargs = {
        "input_features": input_features,
        "max_length": bundle["max_length"],
        "num_beams": bundle["num_beams"],
    }
    if "attention_mask" in batch:
        generate_kwargs["attention_mask"] = batch["attention_mask"].to(bundle["device"])

    with torch.no_grad():
        generated_ids = bundle["model"].generate(**generate_kwargs)
    prediction = bundle["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    return normalize_target_text(prediction, target_mode)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    model_source = Path(args.model_source).expanduser().resolve()
    processor_dir = Path(args.processor_dir).expanduser().resolve() if args.processor_dir else None
    output_prefix = make_output_prefix(input_tsv, args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_tsv, args.limit)
    needs_search_index = any(
        not any(str(row.get(key) or "").strip() for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"))
        for row in rows
    )
    roots = [Path(p).expanduser().resolve() for p in args.search_root] if args.search_root else default_search_roots(script_dir)
    rec_audio = collect_audio_index(roots) if needs_search_index else {}
    bundle = load_model_bundle(
        model_source,
        processor_dir,
        args.device,
        args.precision,
        args.num_beams,
        args.max_length,
    )

    results: list[dict[str, str]] = []
    missing_audio: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        label = row_label(row, index)
        audio_path = resolve_audio_path(row, rec_audio)
        if audio_path is None:
            missing_audio.append(row)
            continue
        predicted_text = transcribe_audio_file(bundle, audio_path, args.target_mode)
        result = dict(row)
        result["local_audio_path"] = str(audio_path)
        result["predicted_text"] = predicted_text
        results.append(result)
        print(f"[{index}/{len(rows)}] {label} -> {predicted_text}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["recording_id"]
    for extra in ("local_audio_path", "predicted_text"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    tsv_path = output_prefix.with_name(output_prefix.name + "_whisper_predictions.tsv")
    jsonl_path = output_prefix.with_name(output_prefix.name + "_whisper_predictions.jsonl")
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
        "model_source": str(model_source),
        "processor_dir": str(processor_dir) if processor_dir else "",
        "device": args.device,
        "precision": args.precision,
        "num_beams": max(1, args.num_beams),
        "max_length": max(32, args.max_length),
        "target_mode": args.target_mode,
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
