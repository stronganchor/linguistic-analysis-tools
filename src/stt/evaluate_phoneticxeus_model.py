#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import torch
import torchaudio


STRESS_MARK_PATTERN = re.compile(r"[ˈˌ']")


def normalize_ipa_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = STRESS_MARK_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_ipa_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_ipa_text(value))


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


def load_eval_rows(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split") != "eval":
                continue
            rows.append(row)
    return rows


def resolve_audio_path(row: dict) -> Path:
    raw_value = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw_value = str(raw_value).strip()
    if not raw_value:
        raise RuntimeError("Missing audio path in manifest row")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail

    return Path(raw_value).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhoneticXeus on manifest_ipa.tsv."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest_ipa.tsv")
    parser.add_argument(
        "--repo-root",
        default="~/zazaki-stt-linux/alt-models/PhoneticXeus",
        help="Path to cloned PhoneticXeus repo",
    )
    parser.add_argument(
        "--work-dir",
        default="~/zazaki-stt-linux/alt-models/cache/xeus",
        help="Directory containing espnet/xeus backbone files",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/huggingface/hub/models--changelinglab--PhoneticXeus/snapshots/91240a28a35578f34482341fc46822589da4a6c8/checkpoint-22000.ckpt",
        help="Path to PhoneticXeus checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write eval metrics and predictions",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="float32 or float16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of eval rows for smoke testing",
    )
    parser.add_argument(
        "--no-conditioning",
        action="store_true",
        help="Disable interctc conditioning even if README example enables it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    from src.model.xeusphoneme.builders import build_xeus_pr_inference

    rows = load_eval_rows(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError(f"No eval rows found in {manifest_path}")

    print(f"Loading PhoneticXeus from: {checkpoint}")
    inference = build_xeus_pr_inference(
        work_dir=str(work_dir),
        checkpoint=str(checkpoint),
        vocab_file=str(repo_root / "src/model/xeusphoneme/resources/ipa_vocab.json"),
        hf_repo="espnet/xeus",
        device=args.device,
        dtype=args.dtype,
        interctc_use_conditioning=not args.no_conditioning,
    )

    pairs_normalized: list[tuple[str, str]] = []
    pairs_compact: list[tuple[str, str]] = []
    prediction_rows: list[dict[str, str | float]] = []
    total_audio_seconds = 0.0

    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row)
        target = normalize_ipa_text(str(row.get("target_ipa") or ""))
        compact_target = compact_ipa_text(target)

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        total_audio_seconds += waveform.numel() / 16000.0
        result = inference(waveform)
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        raw_prediction = result[0].get("processed_transcript") or ""
        prediction = normalize_ipa_text(str(raw_prediction))
        compact_prediction = compact_ipa_text(prediction)

        pairs_normalized.append((target, prediction))
        pairs_compact.append((compact_target, compact_prediction))
        prediction_rows.append(
            {
                "audio": str(audio_path),
                "reference_ipa": target,
                "predicted_ipa": prediction,
                "reference_compact": compact_target,
                "predicted_compact": compact_prediction,
            }
        )

        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)}")

    elapsed = time.perf_counter() - started
    metrics = {
        "model": "phonetic_xeus",
        "checkpoint": str(checkpoint),
        "work_dir": str(work_dir),
        "eval_examples": len(rows),
        "test_cer": compute_cer(pairs_normalized),
        "test_exact_match": sum(1 for ref, hyp in pairs_normalized if ref == hyp) / len(pairs_normalized),
        "compact_test_cer": compute_cer(pairs_compact),
        "compact_test_exact_match": sum(1 for ref, hyp in pairs_compact if ref == hyp) / len(pairs_compact),
        "runtime_seconds": elapsed,
        "avg_seconds_per_clip": elapsed / len(rows),
        "total_audio_seconds": total_audio_seconds,
        "realtime_factor": elapsed / total_audio_seconds if total_audio_seconds else 0.0,
    }

    metrics_path = output_dir / "eval_metrics.json"
    predictions_path = output_dir / "eval_predictions.tsv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio",
                "reference_ipa",
                "predicted_ipa",
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
