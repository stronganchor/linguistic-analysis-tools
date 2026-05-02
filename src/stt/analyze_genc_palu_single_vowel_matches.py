#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vowel_segment_tools import (
    DEFAULT_PRAAT_EXE,
    IPA_TO_BROAD_VOWEL,
    count_ipa_vowels,
    detect_vowel_segments_for_file,
    ensure_praat_script,
    load_settings,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IPA_SINGLE_VOWELS = {"i", "ɪ", "e", "ɛ", "æ", "ɨ", "u", "o", "a"}
CLOSE_VOWEL_NEIGHBORS = {
    "i": {"ɪ", "e"},
    "ɪ": {"i", "e", "ɛ", "ɨ"},
    "e": {"i", "ɪ", "ɛ"},
    "ɛ": {"ɪ", "e", "æ", "a"},
    "æ": {"ɛ", "a"},
    "ɨ": {"ɪ", "e", "o", "u"},
    "u": {"o", "ɨ"},
    "o": {"u", "ɨ"},
    "a": {"æ", "ɛ"},
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_metadata = (
        script_dir
        / "extracted_bundles_whisper_20260410"
        / "ll-tools-stt-training-genc-palu-recording_ipa-20260408-134225"
        / "metadata.jsonl"
    )
    default_output_dir = script_dir / "vowel_segment_reports" / "genc-palu-single-vowel-matches"
    default_cache_dir = script_dir / ".vowel-segment-cache-single-vowel-matches"
    default_settings = Path("$PRAAT_SCRIPTS_DIR/vowel_finder_settings.json")

    parser = argparse.ArgumentParser(
        description=(
            "Analyze single-vowel IPA recordings in the genc-palu dataset and compare "
            "the detector's best vowel guess against the transcription."
        )
    )
    parser.add_argument("--metadata-path", default=str(default_metadata))
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--cache-dir", default=str(default_cache_dir))
    parser.add_argument("--settings-path", default=str(default_settings))
    parser.add_argument("--praat-exe", default=DEFAULT_PRAAT_EXE)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Optional speaker filter. Use 0 to keep all speakers.",
    )
    parser.add_argument(
        "--recording-types",
        nargs="*",
        default=["all"],
        help="Recording types to keep. Use `all` to disable this filter.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_single_expected_vowel(ipa_value: str) -> str | None:
    text = unicodedata.normalize("NFC", ipa_value or "")
    vowels = [char for char in text if char in IPA_SINGLE_VOWELS]
    if len(vowels) != 1:
        return None
    return vowels[0]


def filter_rows(rows: list[dict], speaker_id: int, recording_types: list[str], limit: int) -> list[dict]:
    keep_all_types = any(item.lower() == "all" for item in recording_types)
    allowed_types = {item.lower() for item in recording_types}
    filtered = []
    for row in rows:
        if speaker_id and int(row.get("speaker_user_id") or 0) != speaker_id:
            continue
        expected = extract_single_expected_vowel(row.get("recording_ipa", ""))
        if not expected:
            continue
        row_types = [str(item).lower() for item in row.get("recording_types", [])]
        if not keep_all_types and not any(item in allowed_types for item in row_types):
            continue
        if not row.get("audio"):
            continue
        filtered.append(row)
        if limit and len(filtered) >= limit:
            break
    return filtered


def pick_best_segment(segments: list[dict]) -> dict | None:
    if not segments:
        return None
    return max(
        segments,
        key=lambda segment: (
            float(segment.get("peak_score", 0.0)),
            float(segment.get("mean_score", 0.0)),
            float(segment.get("duration_s", 0.0)),
            int(segment.get("frames", 0)),
        ),
    )


def classify_match(expected_vowel: str, detected_vowel: str | None) -> tuple[str, str]:
    if not detected_vowel or detected_vowel == "?":
        return "way_off", "no detected vowel"
    if detected_vowel == expected_vowel:
        return "exact", "same vowel"
    if detected_vowel in CLOSE_VOWEL_NEIGHBORS.get(expected_vowel, set()):
        return "close", "neighbor vowel"
    expected_broad = IPA_TO_BROAD_VOWEL.get(expected_vowel)
    detected_broad = IPA_TO_BROAD_VOWEL.get(detected_vowel)
    if expected_broad and detected_broad and expected_broad == detected_broad:
        return "close", "same broad vowel region"
    return "way_off", "different vowel region"


def analyze_one_row(
    row: dict,
    metadata_path: Path,
    settings: dict,
    praat_exe: str,
    script_path: Path,
    cache_dir: Path,
) -> dict:
    recording_id = int(row["recording_id"])
    audio_path = metadata_path.parent / row["audio"]
    wav_path = cache_dir / "wav" / f"{recording_id}.wav"
    tsv_path = cache_dir / "formants" / f"{recording_id}.tsv"

    analysis = detect_vowel_segments_for_file(audio_path, settings, praat_exe, script_path, wav_path, tsv_path)
    segments = analysis["segments"]
    best_segment = pick_best_segment(segments)
    expected_vowel = extract_single_expected_vowel(row.get("recording_ipa", "")) or "?"
    detected_vowel = best_segment["ipa_guess"] if best_segment else None
    match_class, match_reason = classify_match(expected_vowel, detected_vowel)

    return {
        "recording_id": recording_id,
        "word_id": int(row.get("word_id") or 0),
        "word_title": row.get("word_title", ""),
        "speaker_user_id": int(row.get("speaker_user_id") or 0),
        "speaker_name": row.get("speaker_name", ""),
        "recording_types": row.get("recording_types", []),
        "recording_ipa": row.get("recording_ipa", ""),
        "expected_vowel": expected_vowel,
        "detected_vowel": detected_vowel or "",
        "match_class": match_class,
        "match_reason": match_reason,
        "audio": row.get("audio", ""),
        "audio_posix": str(audio_path),
        "sample_rate": analysis["sample_rate"],
        "formant_row_count": analysis["formant_rows"],
        "vowel_frame_count": analysis["vowel_frames"],
        "detected_segment_count": len(segments),
        "detected_ipa_sequence": "".join(segment["ipa_guess"] for segment in segments),
        "best_segment": best_segment,
        "segments": segments,
    }


def write_detail_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "recording_id",
                "word_id",
                "word_title",
                "speaker_user_id",
                "speaker_name",
                "recording_types",
                "recording_ipa",
                "expected_vowel",
                "detected_vowel",
                "match_class",
                "match_reason",
                "detected_segment_count",
                "detected_ipa_sequence",
                "best_segment_start_s",
                "best_segment_end_s",
                "best_segment_median_f1",
                "best_segment_median_f2",
                "best_segment_peak_score",
                "audio",
            ]
        )
        for row in rows:
            best = row.get("best_segment") or {}
            writer.writerow(
                [
                    row["recording_id"],
                    row["word_id"],
                    row["word_title"],
                    row["speaker_user_id"],
                    row["speaker_name"],
                    ",".join(row["recording_types"]),
                    row["recording_ipa"],
                    row["expected_vowel"],
                    row["detected_vowel"],
                    row["match_class"],
                    row["match_reason"],
                    row["detected_segment_count"],
                    row["detected_ipa_sequence"],
                    f"{float(best.get('start_s', 0.0)):.4f}" if best else "",
                    f"{float(best.get('end_s', 0.0)):.4f}" if best else "",
                    f"{float(best.get('median_f1', 0.0)):.1f}" if best else "",
                    f"{float(best.get('median_f2', 0.0)):.1f}" if best else "",
                    f"{float(best.get('peak_score', 0.0)):.3f}" if best else "",
                    row["audio"],
                ]
            )


def write_confusion_tsv(path: Path, counts: Counter) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["expected_vowel", "detected_vowel", "count"])
        for (expected, detected), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])):
            writer.writerow([expected, detected, count])


def build_summary(rows: list[dict], errors: list[dict], requested_rows: int) -> dict:
    class_counts = Counter(row["match_class"] for row in rows)
    per_expected = defaultdict(lambda: Counter())
    confusion = Counter()
    speaker_counts = Counter()
    type_counts = Counter()
    no_best_segment = 0

    for row in rows:
        expected = row["expected_vowel"]
        detected = row["detected_vowel"] or "(none)"
        per_expected[expected]["rows"] += 1
        per_expected[expected][row["match_class"]] += 1
        confusion[(expected, detected)] += 1
        speaker_counts[f"{row['speaker_user_id']}:{row['speaker_name']}"] += 1
        type_counts[",".join(row["recording_types"])] += 1
        if not row.get("best_segment"):
            no_best_segment += 1

    per_expected_summary = {}
    for expected, counts in sorted(per_expected.items()):
        total = counts["rows"]
        per_expected_summary[expected] = {
            "rows": total,
            "exact": counts["exact"],
            "close": counts["close"],
            "way_off": counts["way_off"],
            "exact_rate": (counts["exact"] / total) if total else 0.0,
            "close_or_better_rate": ((counts["exact"] + counts["close"]) / total) if total else 0.0,
        }

    return {
        "requested_rows": requested_rows,
        "successful_rows": len(rows),
        "error_rows": len(errors),
        "exact_matches": class_counts["exact"],
        "close_matches": class_counts["close"],
        "way_off_matches": class_counts["way_off"],
        "exact_match_rate": (class_counts["exact"] / len(rows)) if rows else 0.0,
        "close_match_rate": (class_counts["close"] / len(rows)) if rows else 0.0,
        "way_off_match_rate": (class_counts["way_off"] / len(rows)) if rows else 0.0,
        "close_or_better_rate": ((class_counts["exact"] + class_counts["close"]) / len(rows)) if rows else 0.0,
        "rows_with_no_best_segment": no_best_segment,
        "detected_segment_count_histogram": dict(sorted(Counter(row["detected_segment_count"] for row in rows).items())),
        "expected_vowel_counts": dict(sorted(Counter(row["expected_vowel"] for row in rows).items())),
        "recording_type_counts": dict(sorted(type_counts.items())),
        "speaker_counts": dict(sorted(speaker_counts.items())),
        "per_expected_vowel": per_expected_summary,
        "top_confusions": [
            {
                "expected_vowel": expected,
                "detected_vowel": detected,
                "count": count,
            }
            for (expected, detected), count in confusion.most_common(20)
        ],
        "sample_way_off_rows": [
            {
                "recording_id": row["recording_id"],
                "word_title": row["word_title"],
                "recording_ipa": row["recording_ipa"],
                "expected_vowel": row["expected_vowel"],
                "detected_vowel": row["detected_vowel"],
                "detected_ipa_sequence": row["detected_ipa_sequence"],
                "speaker_name": row["speaker_name"],
            }
            for row in rows
            if row["match_class"] == "way_off"
        ][:15],
        "sample_errors": errors[:15],
    }


def main() -> int:
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    settings_path = Path(args.settings_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(settings_path)
    script_path = ensure_praat_script(cache_dir / "export_formants_batch.praat")
    rows = load_jsonl(metadata_path)
    target_rows = filter_rows(rows, args.speaker_id, args.recording_types, args.limit)

    print(
        json.dumps(
            {
                "metadata_rows": len(rows),
                "target_rows": len(target_rows),
                "speaker_id": args.speaker_id,
                "recording_types": args.recording_types,
            },
            ensure_ascii=False,
        )
    )

    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {
            pool.submit(
                analyze_one_row,
                row,
                metadata_path,
                settings,
                args.praat_exe,
                script_path,
                cache_dir,
            ): row
            for row in target_rows
        }
        completed = 0
        for future in as_completed(future_map):
            row = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "recording_id": int(row["recording_id"]),
                        "audio": row.get("audio", ""),
                        "error": str(exc),
                    }
                )
            completed += 1
            if completed % 25 == 0 or completed == len(target_rows):
                print(
                    json.dumps(
                        {
                            "progress": completed,
                            "total": len(target_rows),
                            "successful": len(results),
                            "errors": len(errors),
                        },
                        ensure_ascii=False,
                    )
                )

    results.sort(key=lambda row: row["recording_id"])
    errors.sort(key=lambda row: row["recording_id"])

    summary = build_summary(results, errors, len(target_rows))
    confusion = Counter((row["expected_vowel"], row["detected_vowel"] or "(none)") for row in results)

    detail_jsonl = output_dir / "single_vowel_matches.jsonl"
    detail_tsv = output_dir / "single_vowel_matches.tsv"
    errors_jsonl = output_dir / "single_vowel_match_errors.jsonl"
    confusion_tsv = output_dir / "single_vowel_confusions.tsv"
    summary_json = output_dir / "single_vowel_match_summary.json"

    write_jsonl(detail_jsonl, results)
    write_detail_tsv(detail_tsv, results)
    write_jsonl(errors_jsonl, errors)
    write_confusion_tsv(confusion_tsv, confusion)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
