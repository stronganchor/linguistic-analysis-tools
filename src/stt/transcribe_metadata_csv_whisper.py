#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import librosa
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues
from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text, to_windows_path


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run a local Whisper IPA model on a metadata.csv bundle export and write JSONL predictions."
    )
    parser.add_argument("--model-source", required=True)
    parser.add_argument(
        "--tokenizer-source",
        default="",
        help="Optional separate tokenizer/processor source directory.",
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument(
        "--bundle-root",
        required=True,
        help="Root directory containing the audio/ folder referenced by metadata.csv.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--speaker-id", default="8")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--recording-ids-path",
        default="",
        help="Optional text file with one recording_id per line.",
    )
    parser.add_argument(
        "--only-missing-ipa",
        action="store_true",
        help="Only include rows whose current recording_ipa is blank.",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--text-field",
        default="recording_text",
        help="Expected text_field value in metadata.csv. Use empty string to disable filtering.",
    )
    return parser.parse_args()


def choose_device(value: str) -> str:
    requested = normalize_plain_text(value).lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_torch_dtype(device_name: str, precision: str):
    requested = normalize_plain_text(precision).lower()
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


def default_output_path(script_dir: Path, model_source: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_bits = normalize_plain_text(Path(model_source).name).replace(" ", "-") or "model"
    return script_dir / "transcription_outputs" / f"metadata-csv-{model_bits}-{timestamp}.jsonl"


def load_recording_id_filter(path: Path) -> set[str]:
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = normalize_plain_text(line)
            if value:
                allowed.add(value)
    return allowed


def load_rows(
    metadata_csv: Path,
    bundle_root: Path,
    speaker_id: str,
    *,
    expected_text_field: str,
    allowed_recording_ids: set[str] | None,
    only_missing_ipa: bool,
    max_examples: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expected = normalize_plain_text(expected_text_field)

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            if expected and normalize_plain_text(str(entry.get("text_field") or "")) != expected:
                continue

            entry_speaker = normalize_plain_text(str(entry.get("speaker_user_id") or ""))
            if speaker_id and entry_speaker != speaker_id:
                continue

            recording_id = normalize_plain_text(str(entry.get("recording_id") or ""))
            if allowed_recording_ids is not None and recording_id not in allowed_recording_ids:
                continue

            existing_ipa = normalize_ipa_text(str(entry.get("recording_ipa") or ""))
            if only_missing_ipa and existing_ipa:
                continue

            audio_relpath = normalize_plain_text(str(entry.get("audio") or ""))
            if not audio_relpath:
                continue

            audio_path = bundle_root / audio_relpath
            if not audio_path.is_file():
                continue

            rows.append(
                {
                    "word_id": normalize_plain_text(str(entry.get("word_id") or "")),
                    "word_title": normalize_plain_text(str(entry.get("word_title") or "")),
                    "word_translation": normalize_plain_text(str(entry.get("word_translation") or "")),
                    "recording_id": recording_id,
                    "recording_type": normalize_plain_text(str(entry.get("recording_types") or "")) or "unknown",
                    "speaker_user_id": entry_speaker,
                    "speaker_name": normalize_plain_text(str(entry.get("speaker_name") or "")),
                    "audio_relpath": audio_relpath,
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "recording_text": normalize_plain_text(str(entry.get("recording_text") or "")),
                    "recording_translation": normalize_plain_text(str(entry.get("recording_translation") or "")),
                    "existing_ipa": existing_ipa,
                    "recording_ipa": "",
                    "ipa_rule_issue_count": 0,
                    "ipa_rule_issue_types": [],
                    "ipa_rule_issue_details": [],
                    "ipa_rule_repaired": "",
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["word_title"]),
            str(row["recording_type"]),
            str(row["recording_id"]),
        )
    )
    if max_examples > 0:
        rows = rows[:max_examples]
    return rows


def load_model_bundle(
    model_source: str,
    device_name: str,
    precision: str,
    tokenizer_source_override: str = "",
) -> dict[str, object]:
    model_path = Path(model_source).expanduser().resolve()
    tokenizer_source = (
        Path(tokenizer_source_override).expanduser().resolve()
        if normalize_plain_text(tokenizer_source_override)
        else model_path
    )
    required_tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json")
    if tokenizer_source.is_dir() and not all((tokenizer_source / name).exists() for name in required_tokenizer_files):
        for parent in model_path.parents:
            if all((parent / name).exists() for name in required_tokenizer_files):
                tokenizer_source = parent
                break

    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_source)
    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_source)
    processor = WhisperProcessor.from_pretrained(tokenizer_source)
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(model_path, **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []

    device = torch.device(device_name)
    model.to(device)
    model.eval()

    return {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "feature_dtype": resolve_feature_dtype(model),
    }


def decode_rows(
    rows: list[dict[str, object]],
    bundle: dict[str, object],
    *,
    batch_size: int,
    num_beams: int,
    max_length: int,
) -> None:
    feature_extractor = bundle["feature_extractor"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    feature_dtype = bundle["feature_dtype"]

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            audio_arrays = []
            for row in chunk:
                audio_array, _ = librosa.load(str(row["audio_posix"]), sr=TARGET_SR, mono=True)
                audio_arrays.append(audio_array)

            batch = feature_extractor(
                audio_arrays,
                sampling_rate=TARGET_SR,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = batch["input_features"].to(device)
            if feature_dtype is not None:
                input_features = input_features.to(dtype=feature_dtype)

            generate_kwargs = {
                "input_features": input_features,
                "max_length": max(32, max_length),
                "num_beams": max(1, num_beams),
            }
            if "attention_mask" in batch:
                generate_kwargs["attention_mask"] = batch["attention_mask"].to(device)

            generated_ids = model.generate(**generate_kwargs)
            predicted_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for row, prediction in zip(chunk, predicted_text):
                row["recording_ipa"] = normalize_ipa_text(prediction)
                issue_summary = summarize_ipa_rule_issues(row["recording_ipa"])
                row["ipa_rule_issue_count"] = int(issue_summary["count"])
                row["ipa_rule_issue_types"] = issue_summary["types"]
                row["ipa_rule_issue_details"] = issue_summary["details"]
                row["ipa_rule_repaired"] = basic_repair_ipa_text(row["recording_ipa"])
                print(
                    f"[{rows.index(row) + 1}/{len(rows)}] "
                    f"{row['recording_id']} {row['audio_relpath']} -> {row['recording_ipa']}"
                )


def write_outputs(output_path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "total_predictions": len(rows),
        "model_source": args.model_source,
        "speaker_id": normalize_plain_text(args.speaker_id),
        "only_missing_ipa": bool(args.only_missing_ipa),
        "text_field": normalize_plain_text(args.text_field),
        "num_beams": int(args.num_beams),
        "by_recording_type": dict(Counter(str(row["recording_type"]) for row in rows)),
        "rows_with_existing_ipa": sum(1 for row in rows if str(row.get("existing_ipa") or "")),
        "rows_missing_existing_ipa": sum(1 for row in rows if not str(row.get("existing_ipa") or "")),
        "rows_with_ipa_rule_issues": sum(1 for row in rows if int(row.get("ipa_rule_issue_count") or 0) > 0),
        "ipa_rule_issue_type_counts": dict(
            Counter(
                issue_type
                for row in rows
                for issue_type in (row.get("ipa_rule_issue_types") or [])
                if issue_type
            )
        ),
    }

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote import-ready JSONL: {output_path}")
    print(f"Wrote summary:           {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    bundle_root = Path(args.bundle_root).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(script_dir, args.model_source).resolve()
    )

    allowed_recording_ids = None
    if args.recording_ids_path:
        allowed_recording_ids = load_recording_id_filter(Path(args.recording_ids_path).expanduser().resolve())

    rows = load_rows(
        metadata_csv=metadata_csv,
        bundle_root=bundle_root,
        speaker_id=normalize_plain_text(args.speaker_id),
        expected_text_field=args.text_field,
        allowed_recording_ids=allowed_recording_ids,
        only_missing_ipa=bool(args.only_missing_ipa),
        max_examples=max(0, int(args.max_examples)),
    )
    if not rows:
        raise SystemExit("No usable rows were found after filtering.")

    device_name = choose_device(args.device)
    print(f"Rows to decode: {len(rows)}")
    print(f"Using model:    {args.model_source}")
    print(f"Using device:   {device_name}")

    bundle = load_model_bundle(args.model_source, device_name, args.precision, args.tokenizer_source)
    decode_rows(
        rows,
        bundle,
        batch_size=max(1, args.batch_size),
        num_beams=max(1, args.num_beams),
        max_length=max(32, args.max_length),
    )
    write_outputs(output_path, rows, args)


if __name__ == "__main__":
    main()
