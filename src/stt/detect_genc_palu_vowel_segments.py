#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vowel_segment_tools import (
    DEFAULT_PRAAT_EXE,
    count_ipa_vowels,
    detect_vowel_segments_for_file,
    ensure_praat_script,
    load_settings,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_metadata = (
        script_dir
        / "extracted_bundles_whisper_20260410"
        / "ll-tools-stt-training-genc-palu-recording_ipa-20260408-134225"
        / "metadata.jsonl"
    )
    default_output_dir = script_dir / "vowel_segment_reports" / "genc-palu-ak-isolation"
    default_cache_dir = script_dir / ".vowel-segment-cache"
    default_settings = Path("$PRAAT_SCRIPTS_DIR/vowel_finder_settings.json")

    parser = argparse.ArgumentParser(
        description=(
            "Batch-extract likely vowel segments from the genc-palu dataset, "
            "with speaker filtering and per-recording segment summaries."
        )
    )
    parser.add_argument("--metadata-path", default=str(default_metadata))
    parser.add_argument("--speaker-id", type=int, default=8)
    parser.add_argument(
        "--recording-types",
        nargs="*",
        default=["isolation"],
        help="Recording types to keep. Use `all` to disable this filter.",
    )
    parser.add_argument("--output-dir", default=str(default_output_dir))
    parser.add_argument("--cache-dir", default=str(default_cache_dir))
    parser.add_argument("--settings-path", default=str(default_settings))
    parser.add_argument("--praat-exe", default=DEFAULT_PRAAT_EXE)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
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


def filter_rows(rows: list[dict], speaker_id: int, recording_types: list[str], limit: int) -> list[dict]:
    keep_all_types = any(item.lower() == "all" for item in recording_types)
    allowed_types = {item.lower() for item in recording_types}
    filtered = []
    for row in rows:
        if int(row.get("speaker_user_id") or 0) != speaker_id:
            continue
        row_types = [str(item).lower() for item in row.get("recording_types", [])]
        if not keep_all_types and not any(item in allowed_types for item in row_types):
            continue
        filtered.append(row)
        if limit and len(filtered) >= limit:
            break
    return filtered


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
    expected_vowels = count_ipa_vowels(row.get("recording_ipa", ""))
    analysis = detect_vowel_segments_for_file(audio_path, settings, praat_exe, script_path, wav_path, tsv_path)
    segments = analysis["segments"]
    segment_sequence = "".join(segment["ipa_guess"] for segment in segments)
    return {
        "recording_id": recording_id,
        "word_id": int(row["word_id"]),
        "word_title": row.get("word_title", ""),
        "recording_types": row.get("recording_types", []),
        "speaker_user_id": int(row.get("speaker_user_id") or 0),
        "speaker_name": row.get("speaker_name", ""),
        "recording_ipa": row.get("recording_ipa", ""),
        "expected_vowel_count": expected_vowels,
        "detected_segment_count": len(segments),
        "vowel_count_delta": len(segments) - expected_vowels,
        "detected_ipa_sequence": segment_sequence,
        "audio": row.get("audio", ""),
        "audio_posix": str(audio_path),
        "formant_row_count": analysis["formant_rows"],
        "vowel_frame_count": analysis["vowel_frames"],
        "sample_rate": analysis["sample_rate"],
        "segments": segments,
    }


def write_summary_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "recording_id",
                "word_id",
                "word_title",
                "recording_types",
                "speaker_user_id",
                "speaker_name",
                "expected_vowel_count",
                "detected_segment_count",
                "vowel_count_delta",
                "detected_ipa_sequence",
                "recording_ipa",
                "segment_starts_s",
                "segment_ends_s",
                "segment_labels",
                "audio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["recording_id"],
                    row["word_id"],
                    row["word_title"],
                    ",".join(row["recording_types"]),
                    row["speaker_user_id"],
                    row["speaker_name"],
                    row["expected_vowel_count"],
                    row["detected_segment_count"],
                    row["vowel_count_delta"],
                    row["detected_ipa_sequence"],
                    row["recording_ipa"],
                    ",".join(f"{segment['start_s']:.4f}" for segment in row["segments"]),
                    ",".join(f"{segment['end_s']:.4f}" for segment in row["segments"]),
                    ",".join(segment["ipa_guess"] for segment in row["segments"]),
                    row["audio"],
                ]
            )


def build_summary(rows: list[dict], errors: list[dict], requested_rows: int) -> dict:
    def stats_for_subset(subset: list[dict]) -> dict:
        exact = sum(1 for row in subset if row["vowel_count_delta"] == 0)
        return {
            "rows": len(subset),
            "exact_vowel_count_matches": exact,
            "exact_vowel_count_match_rate": (exact / len(subset)) if subset else 0.0,
            "mean_abs_vowel_count_error": (
                sum(abs(row["vowel_count_delta"]) for row in subset) / len(subset) if subset else 0.0
            ),
            "delta_histogram": dict(sorted(Counter(row["vowel_count_delta"] for row in subset).items())),
        }

    count_delta = Counter(row["vowel_count_delta"] for row in rows)
    type_counts = Counter(",".join(row["recording_types"]) for row in rows)
    exact_matches = sum(1 for row in rows if row["vowel_count_delta"] == 0)
    single_token_rows = [
        row
        for row in rows
        if row["recording_ipa"]
        and " " not in row["recording_ipa"]
        and row["expected_vowel_count"] <= 3
    ]
    multiword_rows = [row for row in rows if " " in row["recording_ipa"]]
    return {
        "requested_rows": requested_rows,
        "successful_rows": len(rows),
        "error_rows": len(errors),
        "exact_vowel_count_matches": exact_matches,
        "exact_vowel_count_match_rate": (exact_matches / len(rows)) if rows else 0.0,
        "mean_abs_vowel_count_error": (
            sum(abs(row["vowel_count_delta"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "detected_zero_segment_rows": sum(1 for row in rows if row["detected_segment_count"] == 0),
        "rows_with_more_segments_than_vowels": sum(1 for row in rows if row["vowel_count_delta"] > 0),
        "rows_with_fewer_segments_than_vowels": sum(1 for row in rows if row["vowel_count_delta"] < 0),
        "vowel_count_delta_histogram": dict(sorted(count_delta.items())),
        "recording_type_counts": dict(sorted(type_counts.items())),
        "single_token_expected_leq_3": stats_for_subset(single_token_rows),
        "multiword_or_phrase_rows": stats_for_subset(multiword_rows),
        "sample_rows": [
            {
                "recording_id": row["recording_id"],
                "recording_ipa": row["recording_ipa"],
                "detected_ipa_sequence": row["detected_ipa_sequence"],
                "vowel_count_delta": row["vowel_count_delta"],
            }
            for row in rows[:10]
        ],
        "sample_errors": errors[:10],
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
    segments_jsonl = output_dir / "vowel_segments.jsonl"
    segments_tsv = output_dir / "vowel_segments.tsv"
    errors_jsonl = output_dir / "vowel_segment_errors.jsonl"
    summary_json = output_dir / "vowel_segment_summary.json"

    write_jsonl(segments_jsonl, results)
    write_summary_tsv(segments_tsv, results)
    write_jsonl(errors_jsonl, errors)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
