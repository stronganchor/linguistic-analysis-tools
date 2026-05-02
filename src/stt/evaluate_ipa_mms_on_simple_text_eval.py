#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from build_recording_text_orthography_import import orthography_word_from_ipa
from model_paths import stt_model_path
from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text


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
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer(pairs: list[tuple[str, str]]) -> float:
    total_edits = 0
    total_chars = 0
    for ref, hyp in pairs:
        total_edits += levenshtein_distance(ref, hyp)
        total_chars += len(ref)
    return (total_edits / total_chars) if total_chars else 0.0


def normalize_simple_text(value: str) -> str:
    text = normalize_plain_text(value)
    text = text.lower()
    text = re.sub(r"[.?!,:;()]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("â", "e")
    return text


def ipa_phrase_to_simple_orth(value: str) -> str:
    ipa = normalize_ipa_text(value)
    words = [orthography_word_from_ipa(part) for part in ipa.split() if part]
    return normalize_simple_text(" ".join(words))


def load_eval_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split") != "eval":
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
            rows.append(
                {
                    "audio_path": audio_path,
                    "target_text": normalize_simple_text(str(row.get("target_text") or "")),
                    "recording_type": str(row.get("recording_type") or ""),
                    "word_title": str(row.get("word_title") or ""),
                    "audio_relpath": str(row.get("audio_relpath") or ""),
                }
            )
    rows = [row for row in rows if row["audio_path"] and row["target_text"]]
    if not rows:
        raise RuntimeError("No eval rows found in manifest")
    return rows


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.numpy()


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate an IPA MMS model against the simple orthography eval split.")
    parser.add_argument(
        "--manifest",
        default=str(script_dir / "training_data" / "genc-palu-recording_text-20260409-071931-simple-v001" / "manifest_text.tsv"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-warmstart-v003-20260407-124628")),
    )
    parser.add_argument(
        "--processor-dir",
        default="",
        help="Optional processor/tokenizer directory if it is separate from the model checkpoint directory.",
    )
    parser.add_argument(
        "--prediction-format",
        choices=["ipa", "text"],
        default="ipa",
        help="Whether the model outputs IPA that must be converted to simplified orthography, or direct text targets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "evals" / "mms-v003-ipa-on-simple-orth-eval"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    model_dir = Path(args.model_dir).resolve()
    processor_dir = Path(args.processor_dir).resolve() if args.processor_dir else model_dir
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    processor = Wav2Vec2Processor.from_pretrained(str(processor_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    target_sr = int(processor.feature_extractor.sampling_rate)

    prediction_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start : start + args.batch_size]
            audio_arrays = [load_audio(row["audio_path"], target_sr) for row in batch_rows]
            inputs = processor.feature_extractor(
                audio_arrays,
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
            decoded = processor.batch_decode(pred_ids)

            for row, raw_pred in zip(batch_rows, decoded):
                if args.prediction_format == "ipa":
                    pred_ipa = normalize_ipa_text(raw_pred)
                    orth_pred = ipa_phrase_to_simple_orth(pred_ipa)
                else:
                    pred_ipa = ""
                    orth_pred = normalize_simple_text(raw_pred.replace("|", " "))

                prediction_rows.append(
                    {
                        "word_title": row["word_title"],
                        "recording_type": row["recording_type"],
                        "audio_relpath": row["audio_relpath"],
                        "target_text": row["target_text"],
                        "predicted_ipa": pred_ipa,
                        "predicted_simple_orth": orth_pred,
                    }
                )
            print(f"Decoded {min(start + args.batch_size, len(rows))}/{len(rows)} eval rows", flush=True)

    pairs = [(row["target_text"], row["predicted_simple_orth"]) for row in prediction_rows]
    compact_pairs = [(ref.replace(" ", ""), hyp.replace(" ", "")) for ref, hyp in pairs]
    cer = compute_cer(pairs)
    exact = sum(1 for ref, hyp in pairs if ref == hyp) / len(pairs)
    compact_cer = compute_cer(compact_pairs)
    compact_exact = sum(1 for ref, hyp in compact_pairs if ref == hyp) / len(compact_pairs)

    summary = {
        "manifest": str(manifest_path),
        "model_dir": str(model_dir),
        "eval_examples": len(prediction_rows),
        "cer": cer,
        "exact_match": exact,
        "compact_cer": compact_cer,
        "compact_exact_match": compact_exact,
    }

    (output_dir / "eval_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "predictions.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["word_title", "recording_type", "audio_relpath", "target_text", "predicted_simple_orth", "predicted_ipa"])
        for row in prediction_rows:
            writer.writerow(
                [
                    row["word_title"],
                    row["recording_type"],
                    row["audio_relpath"],
                    row["target_text"],
                    row["predicted_simple_orth"],
                    row["predicted_ipa"],
                ]
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
