#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text

TARGET_SR = 16000


def compact_ipa_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_ipa_text(value))


def normalize_simple_text(value: str) -> str:
    text = normalize_plain_text(value)
    text = text.lower()
    text = re.sub(r"[.?!,:;()]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("â", "e")
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def compute_cer(pairs: list[tuple[str, str]]) -> float:
    total_edits = 0
    total_chars = 0
    for ref, hyp in pairs:
        total_edits += levenshtein_distance(ref, hyp)
        total_chars += len(ref)
    return (total_edits / total_chars) if total_chars else 0.0


def load_eval_rows(manifest_path: Path, *, target_field: str, target_mode: str) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("split") or "").strip().lower() != "eval":
                continue
            audio_candidates = [
                str(row.get("audio") or "").strip(),
                str(row.get("audio_windows") or "").strip(),
                str(row.get("audio_posix") or "").strip(),
            ]
            audio_path = ""
            for candidate in audio_candidates:
                if candidate and os.path.isfile(candidate):
                    audio_path = candidate
                    break
            raw_target = str(row.get(target_field) or "")
            target = normalize_simple_text(raw_target) if target_mode == "plain" else normalize_ipa_text(raw_target)
            if not audio_path or not target:
                continue
            rows.append(
                {
                    "audio": audio_path,
                    "target_text": target,
                    "word_title": str(row.get("word_title") or "").strip(),
                    "recording_type": str(row.get("recording_type") or "").strip(),
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Whisper model on a manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to a training/eval manifest.")
    parser.add_argument("--model", required=True, help="Local Whisper model directory or HF id")
    parser.add_argument(
        "--processor-source",
        default="",
        help="Optional processor/tokenizer source when evaluating a checkpoint directory that only contains model weights",
    )
    parser.add_argument(
        "--target-field",
        default="target_ipa",
        help="Manifest column containing the reference text.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["ipa", "plain"],
        default="ipa",
        help="Normalization mode for the reference and decoded text.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for metrics and predictions")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(
        manifest_path,
        target_field=str(args.target_field or "target_ipa"),
        target_mode=args.target_mode,
    )
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No eval rows found in {manifest_path}")

    model_kwargs = {}
    if args.device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    processor_source = args.processor_source or args.model

    print(f"Loading model: {args.model}")
    if processor_source != args.model:
        print(f"Loading processor: {processor_source}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_source)
    tokenizer = WhisperTokenizer.from_pretrained(processor_source)
    processor = WhisperProcessor.from_pretrained(processor_source)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
    model = model.to(args.device).eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.num_beams = max(1, args.generation_num_beams)

    model_dtype = next(model.parameters()).dtype
    pairs_normalized: list[tuple[str, str]] = []
    pairs_compact: list[tuple[str, str]] = []
    prediction_rows: list[dict] = []

    started = time.time()
    for start_idx in range(0, len(rows), max(1, args.batch_size)):
        batch_rows = rows[start_idx : start_idx + max(1, args.batch_size)]
        audio_arrays = [librosa.load(item["audio"], sr=TARGET_SR, mono=True)[0] for item in batch_rows]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(args.device)
        if torch.is_floating_point(input_features):
            input_features = input_features.to(dtype=model_dtype)

        with torch.inference_mode():
            generated = model.generate(
                input_features=input_features,
                max_length=int(args.generation_max_length),
                num_beams=max(1, args.generation_num_beams),
            )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if args.target_mode == "plain":
            predictions = [normalize_simple_text(item) for item in decoded]
        else:
            predictions = [normalize_ipa_text(item) for item in decoded]
        for row, prediction in zip(batch_rows, predictions):
            target = row["target_text"]
            if args.target_mode == "plain":
                compact_target = target.replace(" ", "")
                compact_prediction = prediction.replace(" ", "")
            else:
                compact_target = compact_ipa_text(target)
                compact_prediction = compact_ipa_text(prediction)
            pairs_normalized.append((target, prediction))
            pairs_compact.append((compact_target, compact_prediction))
            prediction_rows.append(
                {
                    "audio": row["audio"],
                    "word_title": row["word_title"],
                    "recording_type": row["recording_type"],
                    "reference_text": target,
                    "predicted_text": prediction,
                    "reference_compact": compact_target,
                    "predicted_compact": compact_prediction,
                }
            )

        processed = min(start_idx + len(batch_rows), len(rows))
        if processed % 10 == 0 or processed == len(rows):
            print(f"{processed}/{len(rows)}")

    elapsed = time.time() - started
    metrics = {
        "model": args.model,
        "target_field": args.target_field,
        "target_mode": args.target_mode,
        "eval_examples": len(rows),
        "test_cer": compute_cer(pairs_normalized),
        "test_exact_match": sum(1 for ref, hyp in pairs_normalized if ref == hyp) / len(pairs_normalized),
        "compact_test_cer": compute_cer(pairs_compact),
        "compact_test_exact_match": sum(1 for ref, hyp in pairs_compact if ref == hyp) / len(pairs_compact),
        "runtime_seconds": elapsed,
        "avg_seconds_per_clip": elapsed / len(rows),
        "generation_num_beams": max(1, args.generation_num_beams),
        "generation_max_length": int(args.generation_max_length),
        "device": args.device,
    }

    metrics_path = output_dir / "eval_metrics.json"
    predictions_path = output_dir / "eval_predictions.tsv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio",
                "word_title",
                "recording_type",
                "reference_text",
                "predicted_text",
                "reference_compact",
                "predicted_compact",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
