#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import laryngeal_onset_classifier_v3 as classifier

VOWELS = set("aeiouyøœɨɪʊɛæɑɔə")
LARYNGEAL_MAP = {
    "ʔ": "glottal_stop",
    "ʕ": "pharyngeal",
    "ʡ": "epiglottal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-review initial laryngeal/vowel onsets against current IPA transcriptions.")
    parser.add_argument(
        "--dataset-zip",
        default=str(SCRIPT_DIR.parent / "zazacaogren" / "zip files" / "ll-tools-stt-training-genc-palu-recording_ipa-20260408-134225.zip"),
        help="Path to the reviewed recording_ipa training zip.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "laryngeal_onset_review_april_2026"),
        help="Directory for JSONL/TSV/summary outputs.",
    )
    return parser.parse_args()


def first_symbol(text: str) -> Optional[str]:
    text = (text or "").strip()
    return text[0] if text else None


def current_broad_label(text: str) -> Optional[str]:
    symbol = first_symbol(text)
    if symbol is None:
        return None
    if symbol in LARYNGEAL_MAP:
        return LARYNGEAL_MAP[symbol]
    if symbol in VOWELS:
        return "none"
    return None


def first_recording_type(row: Dict[str, Any]) -> str:
    types = row.get("recording_types") or []
    if types:
        return str(types[0])
    return ""


def candidate_threshold(current_label: str, recording_type: str) -> float:
    if current_label == "none":
        return 0.70 if recording_type == "isolation" else 0.78
    if current_label == "epiglottal":
        return 0.45
    return 0.60 if recording_type == "isolation" else 0.68


def should_flag_candidate(current_label: str, predicted_label: str, confidence: float, recording_type: str) -> bool:
    if predicted_label == current_label:
        return False
    return confidence >= candidate_threshold(current_label, recording_type)


def json_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False)


def iter_target_rows(dataset_zip: Path) -> Iterable[Dict[str, Any]]:
    with zipfile.ZipFile(dataset_zip) as zf, zf.open("metadata.jsonl") as handle:
        for line in handle:
            row = json.loads(line)
            broad = current_broad_label(row.get("recording_ipa") or "")
            if broad is None:
                continue
            row["_current_broad_label"] = broad
            row["_current_symbol"] = first_symbol(row.get("recording_ipa") or "")
            yield row


def main() -> int:
    args = parse_args()
    dataset_zip = Path(args.dataset_zip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "laryngeal_onset_results.jsonl"
    candidates_path = output_dir / "laryngeal_onset_candidate_changes.jsonl"
    candidates_tsv_path = output_dir / "laryngeal_onset_candidate_changes.tsv"
    summary_path = output_dir / "laryngeal_onset_summary.json"

    counters = {
        "current": Counter(),
        "predicted": Counter(),
        "recording_type": Counter(),
        "agreement_by_type": Counter(),
    }
    confusion: Dict[str, Counter] = defaultdict(Counter)
    total = 0
    agree = 0
    candidate_rows = []
    error_rows = []

    with tempfile.TemporaryDirectory(prefix="laryngeal_onset_batch_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        with zipfile.ZipFile(dataset_zip) as zf, \
                results_path.open("w", encoding="utf-8") as results_file, \
                candidates_path.open("w", encoding="utf-8") as candidates_file:
            for row in iter_target_rows(dataset_zip):
                total += 1
                recording_type = first_recording_type(row)
                current_label = row["_current_broad_label"]
                counters["current"][current_label] += 1
                counters["recording_type"][recording_type] += 1

                extracted = Path(zf.extract(row["audio"], path=tmpdir_path))
                try:
                    y, sr = classifier.load_audio(extracted)
                    features = classifier.extract_features(extracted, y, sr)
                    result = classifier.classify(features)
                    pred = classifier.result_to_jsonable(result)
                    predicted_label = result.label
                    counters["predicted"][predicted_label] += 1
                    confusion[current_label][predicted_label] += 1
                    match = predicted_label == current_label
                    if match:
                        agree += 1
                        counters["agreement_by_type"][recording_type] += 1

                    record = {
                        "recording_id": row["recording_id"],
                        "audio": row["audio"],
                        "word_id": row["word_id"],
                        "word_title": row.get("word_title"),
                        "recording_type": recording_type,
                        "speaker_user_id": row.get("speaker_user_id"),
                        "recording_ipa": row.get("recording_ipa"),
                        "current_symbol": row["_current_symbol"],
                        "current_broad_label": current_label,
                        "predicted_label": predicted_label,
                        "predicted_confidence": pred["confidence"],
                        "scores": pred["scores"],
                        "match": match,
                        "notes": pred["notes"],
                        "features": pred["features"],
                    }
                    results_file.write(json_dumps(record) + "\n")

                    if should_flag_candidate(current_label, predicted_label, float(result.confidence), recording_type):
                        candidate = {
                            "recording_id": row["recording_id"],
                            "word_id": row["word_id"],
                            "word_title": row.get("word_title"),
                            "recording_type": recording_type,
                            "audio": row["audio"],
                            "recording_ipa": row.get("recording_ipa"),
                            "current_broad_label": current_label,
                            "predicted_label": predicted_label,
                            "predicted_confidence": pred["confidence"],
                            "scores": pred["scores"],
                            "notes": pred["notes"],
                        }
                        candidate_rows.append(candidate)
                        candidates_file.write(json_dumps(candidate) + "\n")
                except Exception as exc:
                    error_rows.append({
                        "recording_id": row["recording_id"],
                        "audio": row["audio"],
                        "recording_ipa": row.get("recording_ipa"),
                        "error": str(exc),
                    })

    candidate_rows.sort(key=lambda row: (-float(row["predicted_confidence"]), row["recording_id"]))
    with candidates_tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "recording_id", "word_id", "word_title", "recording_type", "current_broad_label",
            "predicted_label", "predicted_confidence", "recording_ipa", "audio",
        ])
        for row in candidate_rows:
            writer.writerow([
                row["recording_id"],
                row["word_id"],
                row["word_title"],
                row["recording_type"],
                row["current_broad_label"],
                row["predicted_label"],
                row["predicted_confidence"],
                row["recording_ipa"],
                row["audio"],
            ])

    summary = {
        "dataset_zip": str(dataset_zip),
        "total_rows_analyzed": total,
        "total_matches": agree,
        "agreement_rate": (agree / total) if total else 0.0,
        "candidate_rows": len(candidate_rows),
        "error_rows": len(error_rows),
        "current_label_counts": dict(counters["current"]),
        "predicted_label_counts": dict(counters["predicted"]),
        "recording_type_counts": dict(counters["recording_type"]),
        "agreement_by_recording_type": dict(counters["agreement_by_type"]),
        "confusion_matrix": {key: dict(value) for key, value in confusion.items()},
        "results_path": str(results_path),
        "candidate_changes_path": str(candidates_path),
        "candidate_changes_tsv_path": str(candidates_tsv_path),
        "errors": error_rows[:50],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "total_rows_analyzed": total,
        "agreement_rate": round(summary["agreement_rate"], 4),
        "candidate_rows": len(candidate_rows),
        "error_rows": len(error_rows),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
