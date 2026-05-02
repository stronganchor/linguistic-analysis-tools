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

from prepare_lltools_ipa_dataset import (
    build_category_name_lookup,
    extract_bundle,
    first_meta_value,
    list_zip_paths,
    normalize_ipa_text,
    normalize_plain_text,
    to_windows_path,
)


TARGET_SR = 16000

OUTPUT_HEADERS = [
    "source_bundle",
    "word_id",
    "word_title",
    "word_slug",
    "audio_entry_id",
    "audio_attachment_id",
    "recording_type",
    "speaker_user_id",
    "categories",
    "category_names",
    "recording_text",
    "recording_translation",
    "existing_ipa",
    "predicted_ipa",
    "audio_relpath",
    "audio_windows",
    "audio_posix",
]


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
        description="Run a fine-tuned Whisper IPA model on LL Tools full-bundle zips."
    )
    parser.add_argument(
        "--model-source",
        required=True,
        help="Local model directory or Hugging Face model id.",
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
        help="TSV file to write. Defaults to transcription_outputs/<timestamp>.tsv.",
    )
    parser.add_argument(
        "--zip",
        action="append",
        dest="zip_paths",
        default=[],
        help="Specific bundle zip(s) to transcribe. Defaults to all .zip files in --input-dir.",
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
        help="Beam search width. Higher can improve accuracy at the cost of speed.",
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
        "--only-missing-ipa",
        action="store_true",
        help="Skip entries that already have recording_ipa in the bundle metadata.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract bundle zips even if an extracted folder already exists.",
    )
    return parser.parse_args()


def default_output_path(script_dir: Path, model_source: str) -> Path:
    safe_model_name = normalize_plain_text(Path(model_source).name or "model")
    safe_model_name = safe_model_name.replace(" ", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return script_dir / "transcription_outputs" / f"{safe_model_name}-{timestamp}.tsv"


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


def choose_device(value: str) -> str:
    requested = normalize_plain_text(value).lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def iter_prediction_rows(
    bundle_name: str,
    extracted_dir: Path,
    bundle_data: dict,
    speaker_id: int,
    allowed_recording_types: set[str],
    only_missing_ipa: bool,
) -> list[dict[str, str]]:
    category_name_lookup = build_category_name_lookup(bundle_data)
    rows: list[dict[str, str]] = []

    for word in bundle_data.get("words") or []:
        word_id = str(word.get("id") or "")
        word_slug = normalize_plain_text(str(word.get("slug") or ""))
        word_title = normalize_plain_text(str(word.get("title") or ""))
        categories = [normalize_plain_text(str(x)) for x in (word.get("categories") or []) if str(x).strip()]
        category_names = [category_name_lookup.get(slug, slug) for slug in categories]

        for entry in word.get("audio_entries") or []:
            meta = entry.get("meta") or {}
            recording_type_list = [
                normalize_plain_text(str(x))
                for x in (entry.get("recording_types") or [])
                if str(x).strip()
            ]
            if not recording_type_list:
                recording_type_list = ["unknown"]
            primary_recording_type = recording_type_list[0]
            if allowed_recording_types and primary_recording_type not in allowed_recording_types:
                continue

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
                    "source_bundle": bundle_name,
                    "word_id": word_id,
                    "word_title": word_title,
                    "word_slug": word_slug,
                    "audio_entry_id": str(entry.get("id") or ""),
                    "audio_attachment_id": str(audio_file.get("id") or ""),
                    "recording_type": primary_recording_type,
                    "speaker_user_id": str(entry_speaker_id),
                    "categories": "|".join(categories),
                    "category_names": "|".join(category_names),
                    "recording_text": normalize_plain_text(first_meta_value(meta, "recording_text")),
                    "recording_translation": normalize_plain_text(first_meta_value(meta, "recording_translation")),
                    "existing_ipa": existing_ipa,
                    "predicted_ipa": "",
                    "audio_relpath": audio_relpath,
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                }
            )

    return rows


def load_prediction_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    input_dir = Path(args.input_dir).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = list_zip_paths(input_dir, args.zip_paths)
    if not zip_paths:
        raise SystemExit(f"No bundle zip files found in {input_dir}")

    allowed_recording_types = {normalize_plain_text(x) for x in args.recording_types if normalize_plain_text(x)}
    all_rows: list[dict[str, str]] = []

    for zip_path in zip_paths:
        if not zip_path.is_file():
            raise SystemExit(f"Missing zip file: {zip_path}")

        extracted_bundle_dir = extract_bundle(zip_path, extract_dir, args.force_extract)
        data_json_path = extracted_bundle_dir / "data.json"
        if not data_json_path.is_file():
            raise SystemExit(f"Bundle missing data.json: {zip_path}")

        with data_json_path.open("r", encoding="utf-8") as handle:
            bundle_data = json.load(handle)

        rows = iter_prediction_rows(
            bundle_name=zip_path.name,
            extracted_dir=extracted_bundle_dir,
            bundle_data=bundle_data,
            speaker_id=args.speaker_id,
            allowed_recording_types=allowed_recording_types,
            only_missing_ipa=args.only_missing_ipa,
        )
        all_rows.extend(rows)

    if args.max_examples > 0:
        all_rows = all_rows[: args.max_examples]

    if not all_rows:
        raise SystemExit("No usable audio rows were found after filtering.")

    all_rows.sort(key=lambda row: (row["source_bundle"], row["word_slug"], row["recording_type"], row["audio_relpath"]))
    return all_rows


def generate_predictions(
    rows: list[dict[str, str]],
    model_source: str,
    batch_size: int,
    device_name: str,
    precision: str,
    num_beams: int,
    max_length: int,
) -> None:
    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source)
    tokenizer = WhisperTokenizer.from_pretrained(model_source)
    processor = WhisperProcessor.from_pretrained(model_source)
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
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
    feature_dtype = resolve_feature_dtype(model)

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            audio_arrays = []
            for row in chunk:
                audio_array, _ = librosa.load(row["audio_posix"], sr=TARGET_SR, mono=True)
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

            for row, predicted in zip(chunk, predicted_text):
                row["predicted_ipa"] = normalize_ipa_text(predicted)

            print(f"Decoded {min(start + batch_size, len(rows))}/{len(rows)} clips")


def write_outputs(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_tsv": str(output_path),
        "total_predictions": len(rows),
        "with_existing_ipa": sum(1 for row in rows if row["existing_ipa"]),
        "without_existing_ipa": sum(1 for row in rows if not row["existing_ipa"]),
        "by_bundle": dict(Counter(row["source_bundle"] for row in rows)),
        "by_recording_type": dict(Counter(row["recording_type"] for row in rows)),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote predictions: {output_path}")
    print(f"Wrote summary:     {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else default_output_path(script_dir, args.model_source).resolve()
    )
    rows = load_prediction_rows(args)
    device_name = choose_device(args.device)

    print(f"Rows to decode: {len(rows)}")
    print(f"Using device:   {device_name}")
    print(f"Model source:   {args.model_source}")

    generate_predictions(
        rows=rows,
        model_source=args.model_source,
        batch_size=max(1, args.batch_size),
        device_name=device_name,
        precision=args.precision,
        num_beams=max(1, args.num_beams),
        max_length=max(32, args.max_length),
    )
    write_outputs(output_path, rows)


if __name__ == "__main__":
    main()
