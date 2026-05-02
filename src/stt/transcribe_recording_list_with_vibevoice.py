#!/usr/bin/env python3
"""Transcribe a recording-list TSV with the local VibeVoice ASR LoRA."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VibeVoice ASR over a TSV with recording_id and local_audio_path columns."
    )
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--context-line", action="append", default=[])
    return parser.parse_args()


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


def resolve_audio_path(row: dict[str, str]) -> Path | None:
    for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"):
        resolved = resolve_existing_path(str(row.get(key) or ""))
        if resolved is not None:
            return resolved
    return None


def load_rows(input_tsv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not str(row.get("recording_id") or "").strip():
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def collect_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


def build_context(row: dict[str, str], base_context: list[str]) -> str:
    lines = list(base_context)
    word_title = str(row.get("word_title") or "").strip()
    word_translation = str(row.get("word_translation") or "").strip()
    recording_type = str(row.get("recording_type") or "").strip()
    speaker_name = str(row.get("speaker_name") or "").strip()
    speaker_user_id = str(row.get("speaker_user_id") or "").strip()

    if speaker_name or speaker_user_id:
        label = speaker_name or f"speaker_user_id {speaker_user_id}"
        lines.append(f"Speaker: {label}.")
    if word_title:
        lines.append(f"Parent word title: {word_title}.")
    if word_translation:
        lines.append(f"Known lexical hint: {word_translation}.")
    if recording_type:
        lines.append(f"Recording type: {recording_type}.")
    return "\n".join(line for line in lines if line.strip())


def load_model(
    base_model: str,
    lora_path: str,
    attn_implementation: str,
    load_in_4bit: bool,
) -> tuple[Any, VibeVoiceASRProcessor, torch.device]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    quantization_config = None
    device_map = "auto" if torch.cuda.is_available() else None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = "auto"

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    device = next(model.parameters()).device
    return model, processor, device


def transcribe(
    model: Any,
    processor: VibeVoiceASRProcessor,
    device: torch.device,
    audio_path: Path,
    context_info: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = processor(
        audio=str(audio_path),
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info or None,
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception:
        segments = []
    return {
        "raw_text": raw_text,
        "segments": segments,
        "prediction": collect_text(segments) if segments else raw_text.strip(),
    }


def main() -> int:
    args = parse_args()
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base_context = args.context_line or [
        "Central Zazaki practical orthography transcription.",
        "The speaker is from the Genç Central Zazaki project.",
        "Return only the spoken transcription, preserving Zazaki diacritics when heard.",
    ]
    rows = load_rows(input_tsv, args.limit)

    started = time.time()
    model, processor, device = load_model(
        base_model=args.base_model,
        lora_path=args.lora_path,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )

    results: list[dict[str, Any]] = []
    missing_audio: list[dict[str, str]] = []
    jsonl_path = output_prefix.with_name(output_prefix.name + "_vibevoice_predictions.jsonl")
    tsv_path = output_prefix.with_name(output_prefix.name + "_vibevoice_predictions.tsv")
    missing_audio_path = output_prefix.with_name(output_prefix.name + "_missing_audio.tsv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with jsonl_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            rid = str(row.get("recording_id") or "").strip()
            audio_path = resolve_audio_path(row)
            if audio_path is None:
                missing_audio.append(row)
                continue
            context_info = build_context(row, base_context)
            result = transcribe(
                model=model,
                processor=processor,
                device=device,
                audio_path=audio_path,
                context_info=context_info,
                max_new_tokens=args.max_new_tokens,
            )
            out_row = dict(row)
            out_row["local_audio_path"] = str(audio_path)
            out_row["vibevoice_prediction"] = result["prediction"]
            out_row["vibevoice_raw_text"] = result["raw_text"]
            out_row["vibevoice_segments"] = result["segments"]
            out_row["vibevoice_context"] = context_info
            results.append(out_row)
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{index}/{len(rows)}] {rid} -> {result['prediction']}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["recording_id"]
    for extra in ("local_audio_path", "vibevoice_prediction", "vibevoice_raw_text", "vibevoice_context"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    missing_fields = list(rows[0].keys()) if rows else ["recording_id"]
    with missing_audio_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(missing_audio)

    summary = {
        "input_tsv": str(input_tsv),
        "base_model": str(Path(args.base_model).expanduser().resolve()),
        "lora_path": str(Path(args.lora_path).expanduser().resolve()),
        "device": str(device),
        "load_in_4bit": args.load_in_4bit,
        "input_rows": len(rows),
        "transcribed_rows": len(results),
        "missing_audio_rows": len(missing_audio),
        "runtime_seconds": round(time.time() - started, 3),
        "outputs": {
            "tsv": str(tsv_path),
            "jsonl": str(jsonl_path),
            "missing_audio_tsv": str(missing_audio_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
