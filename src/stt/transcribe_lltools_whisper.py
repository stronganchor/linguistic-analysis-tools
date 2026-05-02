#!/usr/bin/env python3
import argparse
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
from prepare_lltools_ipa_dataset import (
    build_category_name_lookup,
    extract_bundle,
    first_meta_value,
    list_zip_paths,
    normalize_ipa_text,
    normalize_plain_text,
    resolve_bundle_content_root,
    to_windows_path,
)


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "zip files from zazacaogren"
    if not default_input_dir.is_dir():
        default_input_dir = script_dir

    parser = argparse.ArgumentParser(
        description="Run a local Whisper IPA model on LL Tools bundle audio and write import-ready JSONL."
    )
    parser.add_argument(
        "--model-source",
        required=True,
        help="Local Whisper model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Folder containing LL Tools full-bundle zip exports.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(script_dir / "extracted_bundles"),
        help="Where extracted bundle contents should be stored.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="JSONL file to write. Defaults to transcription_outputs/<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--zip",
        action="append",
        dest="zip_paths",
        default=[],
        help="Specific bundle zip(s) to use. Defaults to all .zip files in --input-dir.",
    )
    parser.add_argument(
        "--category-slug",
        default="",
        help="Optional category slug filter, for example yiyecekler-cerez.",
    )
    parser.add_argument(
        "--exclude-category-slug",
        action="append",
        dest="exclude_category_slugs",
        default=[],
        help="Optional category slug(s) to exclude. Repeat for multiple values.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Optional speaker_user_id filter. Use 0 to include all speakers.",
    )
    parser.add_argument(
        "--recording-type",
        action="append",
        dest="recording_types",
        default=[],
        help="Optional recording type filter. Repeat for multiple values.",
    )
    parser.add_argument(
        "--only-missing-ipa",
        action="store_true",
        help="Only transcribe rows whose bundle metadata currently lacks recording_ipa.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of audio clips to decode at once.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        help="Model precision for CUDA inference: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width for Whisper generation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Generation max_length for Whisper decoding.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Optional cap for quick tests.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract bundle zips even if an extracted folder already exists.",
    )
    return parser.parse_args()


def default_output_path(script_dir: Path, category_slug: str, model_source: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    category_bits = normalize_plain_text(category_slug).replace(" ", "-") or "all"
    model_bits = normalize_plain_text(Path(model_source).name).replace(" ", "-") or "model"
    return script_dir / "transcription_outputs" / f"{category_bits}-{model_bits}-{timestamp}.jsonl"


def maybe_int(value: str) -> int | str:
    text = normalize_plain_text(value)
    return int(text) if text.isdigit() else text


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


def iter_rows_for_bundle(
    bundle_name: str,
    extracted_dir: Path,
    bundle_data: dict,
    category_slug: str,
    excluded_category_slugs: set[str],
    speaker_id: int,
    allowed_recording_types: set[str],
    only_missing_ipa: bool,
) -> list[dict[str, object]]:
    category_name_lookup = build_category_name_lookup(bundle_data)
    rows: list[dict[str, object]] = []

    for word in bundle_data.get("words") or []:
        categories = [
            normalize_plain_text(str(item))
            for item in (word.get("categories") or [])
            if normalize_plain_text(str(item))
        ]
        if category_slug and category_slug not in categories:
            continue
        if excluded_category_slugs and any(category in excluded_category_slugs for category in categories):
            continue

        category_names = [category_name_lookup.get(slug, slug) for slug in categories]
        word_id = str(word.get("origin_id") or word.get("id") or "")
        word_slug = normalize_plain_text(str(word.get("slug") or ""))

        for entry in word.get("audio_entries") or []:
            recording_types = [
                normalize_plain_text(str(item))
                for item in (entry.get("recording_types") or [])
                if normalize_plain_text(str(item))
            ] or ["unknown"]
            primary_recording_type = recording_types[0]
            if allowed_recording_types and primary_recording_type not in allowed_recording_types:
                continue

            meta = entry.get("meta") or {}
            entry_speaker_raw = normalize_plain_text(first_meta_value(meta, "speaker_user_id"))
            entry_speaker_id = int(entry_speaker_raw) if entry_speaker_raw.isdigit() else 0
            if speaker_id > 0 and entry_speaker_id != speaker_id:
                continue

            existing_ipa = normalize_ipa_text(first_meta_value(meta, "recording_ipa"))
            if only_missing_ipa and existing_ipa:
                continue

            audio_file = entry.get("audio_file") or {}
            audio_relpath = normalize_plain_text(str(audio_file.get("file") or ""))
            if not audio_relpath:
                continue

            audio_path = extracted_dir / audio_relpath
            if not audio_path.is_file():
                continue

            rows.append(
                {
                    "word_id": maybe_int(word_id),
                    "word_title": str(word.get("title") or "").strip(),
                    "word_translation": str((meta.get("word_translation") or [""])[0] or "").strip(),
                    "recording_id": maybe_int(str(entry.get("origin_id") or entry.get("id") or "")),
                    "word_slug": word_slug,
                    "recording_slug": normalize_plain_text(str(entry.get("slug") or "")),
                    "recording_type": primary_recording_type,
                    "speaker_user_id": entry_speaker_id,
                    "source_bundle": bundle_name,
                    "category_slug": "|".join(categories),
                    "category_name": "|".join(category_names),
                    "audio_relpath": audio_relpath,
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "existing_ipa": existing_ipa,
                    "recording_ipa": "",
                    "ipa_rule_issue_count": 0,
                    "ipa_rule_issue_types": [],
                    "ipa_rule_issue_details": [],
                    "ipa_rule_repaired": "",
                }
            )

    return rows


def load_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = list_zip_paths(input_dir, args.zip_paths)
    if not zip_paths:
        raise SystemExit(f"No bundle zip files found in {input_dir}")

    category_slug = normalize_plain_text(args.category_slug)
    excluded_category_slugs = {
        normalize_plain_text(item)
        for item in args.exclude_category_slugs
        if normalize_plain_text(item)
    }
    allowed_recording_types = {
        normalize_plain_text(item)
        for item in args.recording_types
        if normalize_plain_text(item)
    }

    rows: list[dict[str, object]] = []
    for zip_path in zip_paths:
        extracted_bundle_dir = extract_bundle(zip_path, extract_dir, args.force_extract)
        bundle_content_dir = resolve_bundle_content_root(extracted_bundle_dir)
        data_json_path = bundle_content_dir / "data.json"
        if not data_json_path.is_file():
            raise SystemExit(f"Bundle missing data.json: {zip_path}")

        with data_json_path.open("r", encoding="utf-8") as handle:
            bundle_data = json.load(handle)

        rows.extend(
            iter_rows_for_bundle(
                bundle_name=zip_path.name,
                extracted_dir=bundle_content_dir,
                bundle_data=bundle_data,
                category_slug=category_slug,
                excluded_category_slugs=excluded_category_slugs,
                speaker_id=args.speaker_id,
                allowed_recording_types=allowed_recording_types,
                only_missing_ipa=args.only_missing_ipa,
            )
        )

    if args.max_examples > 0:
        rows = rows[: args.max_examples]

    if not rows:
        raise SystemExit("No usable rows were found after filtering.")

    rows.sort(
        key=lambda row: (
            str(row["source_bundle"]),
            str(row["category_slug"]),
            str(row["word_slug"]),
            str(row["recording_type"]),
            str(row["audio_relpath"]),
        )
    )
    return rows


def load_model_bundle(
    model_source: str,
    device_name: str,
    precision: str,
) -> dict[str, object]:
    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source)
    tokenizer = WhisperTokenizer.from_pretrained(model_source)
    processor = WhisperProcessor.from_pretrained(model_source)
    model_kwargs = {"low_cpu_mem_usage": True}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(model_source, **model_kwargs)

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
                    f"{row['audio_relpath']} -> {row['recording_ipa']}"
                )


def write_outputs(output_path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "total_predictions": len(rows),
        "category_slug": normalize_plain_text(args.category_slug),
        "excluded_category_slugs": [
            normalize_plain_text(item)
            for item in args.exclude_category_slugs
            if normalize_plain_text(item)
        ],
        "speaker_id": int(args.speaker_id),
        "only_missing_ipa": bool(args.only_missing_ipa),
        "model_source": args.model_source,
        "num_beams": int(args.num_beams),
        "by_bundle": dict(Counter(str(row["source_bundle"]) for row in rows)),
        "by_recording_type": dict(Counter(str(row["recording_type"]) for row in rows)),
        "by_speaker_user_id": dict(Counter(str(row["speaker_user_id"]) for row in rows)),
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
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(script_dir, args.category_slug, args.model_source).resolve()
    )

    rows = load_rows(args)
    device_name = choose_device(args.device)
    print(f"Rows to decode: {len(rows)}")
    print(f"Using model:    {args.model_source}")
    print(f"Using device:   {device_name}")

    bundle = load_model_bundle(
        model_source=args.model_source,
        device_name=device_name,
        precision=args.precision,
    )
    decode_rows(
        rows=rows,
        bundle=bundle,
        batch_size=max(1, args.batch_size),
        num_beams=max(1, args.num_beams),
        max_length=max(32, args.max_length),
    )
    write_outputs(output_path, rows, args)


if __name__ == "__main__":
    main()
