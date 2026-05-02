#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from prepare_lltools_ipa_dataset import normalize_ipa_text, to_windows_path
from review_lltools_ipa_comparison import ipa_to_canonical


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a review bundle for phase1a rows using MMS, Whisper, reviewed examples, and A.K. references."
    )
    parser.add_argument(
        "--compare-dir",
        action="append",
        dest="compare_dirs",
        default=[],
        help="One or more compare output directories containing phase_split.tsv and phase1a_mms_vs_whisper_compare.tsv.",
    )
    parser.add_argument(
        "--first100-status-tsv",
        default=str(script_dir / "transcription_outputs" / "phase1a_first100_multi_ipa_cpu_20260410-144110" / "phase1a_first100_decision_status_20260410.tsv"),
    )
    parser.add_argument(
        "--manual-import-jsonl",
        default="$HOME/Downloads/unresolved-ipa-review-import.jsonl",
    )
    parser.add_argument(
        "--ak-metadata-jsonl",
        default=str(
            script_dir
            / "extracted_bundles_20260403"
            / "ll-tools-stt-training-genc-palu-recording_ipa-20260403-090111"
            / "metadata.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Defaults to transcription_outputs/phase1a_unreviewed_001_300_review_<timestamp>.",
    )
    return parser.parse_args()


def now_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def levenshtein(a: str, b: str) -> int:
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


def clean_guess(value: str) -> str:
    text = normalize_ipa_text(value)
    if not text:
        return ""
    replacements = {
        "t̪ʃ": "t̪͡ʃ",
        "tʃ": "t͡ʃ",
        "dʒ": "d͡ʒ",
        "cç": "c͡ç",
        "tç": "t͡ç",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\bɨla jo\b", "ɨna jo", text)
    text = re.sub(r"\bʰɨna jo\b", "ɨna jo", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manual_import(path: Path) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path):
        rows[int(row["recording_id"])] = row
    return rows


def load_first100_status(path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    for row in read_tsv(path):
        rows[int(row["clip_id"])] = row
    return rows


def load_ak_reference_forms(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    by_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
    unique_forms: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in read_jsonl(path):
        ipa = clean_guess(str(row.get("recording_ipa") or ""))
        if not ipa:
            continue
        canonical = ipa_to_canonical(ipa)
        item = {
            "ipa": ipa,
            "canonical": canonical,
            "word_title": str(row.get("word_title") or ""),
            "word_translation": str(row.get("word_translation") or ""),
        }
        by_canonical[canonical].append(item)
        unique_forms[(ipa, item["word_title"], item["word_translation"])] = item
    return by_canonical, list(unique_forms.values())


def best_reference_for_candidate(
    candidate: str,
    refs_by_canonical: dict[str, list[dict[str, str]]],
    all_refs: list[dict[str, str]],
) -> tuple[str, str, str, int | None]:
    candidate = clean_guess(candidate)
    if not candidate:
        return "", "", "", None
    canonical = ipa_to_canonical(candidate)
    if canonical in refs_by_canonical:
        ref = refs_by_canonical[canonical][0]
        return ref["ipa"], ref["word_title"], ref["word_translation"], 0

    candidate_tokens = max(1, len(candidate.split()))
    candidate_initial = canonical[:1]
    best_ref = None
    best_dist = None
    for ref in all_refs:
        if abs(len(ref["ipa"].split()) - candidate_tokens) > 1:
            continue
        ref_initial = ref["canonical"][:1]
        if candidate_initial and ref_initial and candidate_initial != ref_initial:
            continue
        dist = levenshtein(canonical, ref["canonical"])
        if best_dist is None or dist < best_dist:
            best_ref = ref
            best_dist = dist
    if best_ref is None:
        return "", "", "", None
    return best_ref["ipa"], best_ref["word_title"], best_ref["word_translation"], best_dist


def choose_best_guess(
    mms_ipa: str,
    whisper_ipa: str,
    refs_by_canonical: dict[str, list[dict[str, str]]],
    all_refs: list[dict[str, str]],
) -> tuple[str, str, str, str]:
    mms_ipa = clean_guess(mms_ipa)
    whisper_ipa = clean_guess(whisper_ipa)
    candidates = []
    for source_name, ipa in (("mms", mms_ipa), ("whisper", whisper_ipa)):
        if not ipa:
            continue
        ref_ipa, ref_title, ref_translation, ref_distance = best_reference_for_candidate(ipa, refs_by_canonical, all_refs)
        score = 0.0
        if source_name == "mms":
            score += 1.0  # favored by reviewed rows
        if ref_distance == 0:
            score += 2.0
        elif ref_distance is not None and ref_distance <= 4:
            score += max(0.2, 1.4 - (0.3 * ref_distance))
        if " jo " in f" {ipa} ":
            score += 0.15
        candidates.append(
            {
                "source": source_name,
                "ipa": ipa,
                "score": score,
                "ref_ipa": ref_ipa,
                "ref_title": ref_title,
                "ref_translation": ref_translation,
                "ref_distance": ref_distance,
            }
        )

    if not candidates:
        return "", "no_model_output", "", ""

    candidates.sort(key=lambda item: (-item["score"], 0 if item["source"] == "mms" else 1, item["ipa"]))
    best = candidates[0]
    reason_parts = [f"base={best['source']}"]
    if best["ref_ipa"]:
        if best["ref_distance"] == 0:
            reason_parts.append("ak_exact")
        elif best["ref_distance"] is not None:
            reason_parts.append(f"ak_near_{best['ref_distance']}")
    else:
        reason_parts.append("no_ak_match")
    return best["ipa"], ";".join(reason_parts), best["ref_title"], best["ref_translation"]


def load_compare_rows(compare_dirs: list[Path]) -> dict[int, dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for compare_dir in compare_dirs:
        compare_path = compare_dir / "phase1a_mms_vs_whisper_compare.tsv"
        split_path = compare_dir / "phase_split.tsv"
        if not compare_path.is_file() or not split_path.is_file():
            raise SystemExit(f"Missing compare artifacts in {compare_dir}")
        compare_rows = {int(row["clip_id"]): row for row in read_tsv(compare_path)}
        split_rows = {int(row["clip_id"]): row for row in read_tsv(split_path)}
        for clip_id, compare_row in compare_rows.items():
            merged = dict(compare_row)
            merged.update(split_rows.get(clip_id, {}))
            by_id[clip_id] = merged
    return by_id


def build_rows(
    compare_rows_by_id: dict[int, dict[str, Any]],
    manual_import: dict[int, dict[str, Any]],
    first100_status: dict[int, dict[str, str]],
    refs_by_canonical: dict[str, list[dict[str, str]]],
    all_refs: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    review_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stats = Counter()

    for clip_id in sorted(compare_rows_by_id):
        if clip_id in manual_import:
            continue
        row = compare_rows_by_id[clip_id]
        mms_ipa = clean_guess(str(row.get("new_mms_ipa") or ""))
        whisper_ipa = clean_guess(str(row.get("whisper_ipa") or ""))
        exact_agree = str(row.get("exact_agree") or "") == "yes"
        canonical_agree = str(row.get("canonical_agree") or "") == "yes"
        quality_flag = ""
        assistant_guess_ipa = ""
        assistant_reason = ""
        ref_title = ""
        ref_translation = ""

        if clip_id in first100_status:
            status_row = first100_status[clip_id]
            quality_flag = status_row.get("status", "")
            assistant_guess_ipa = clean_guess(status_row.get("chosen_ipa") or "")
            assistant_reason = status_row.get("reason") or ""
            if quality_flag == "flagged_skip":
                assistant_guess_ipa = ""
        if not assistant_guess_ipa:
            assistant_guess_ipa, assistant_reason, ref_title, ref_translation = choose_best_guess(
                mms_ipa,
                whisper_ipa,
                refs_by_canonical,
                all_refs,
            )
            if not quality_flag:
                quality_flag = "not_yet_reviewed"

        if exact_agree and not assistant_guess_ipa:
            assistant_guess_ipa = clean_guess(str(row.get("auto_accept_ipa") or ""))
            assistant_reason = "mms_whisper_exact"

        review_rows.append(
            {
                "clip_id": clip_id,
                "audio_filename": row.get("audio_filename", ""),
                "start_sec": row.get("start_sec", ""),
                "end_sec": row.get("end_sec", ""),
                "duration_sec": row.get("duration_sec", ""),
                "assistant_guess_ipa": assistant_guess_ipa,
                "assistant_reason": assistant_reason,
                "assistant_ref_title": ref_title,
                "assistant_ref_translation": ref_translation,
                "new_mms_ipa": mms_ipa,
                "whisper_ipa": whisper_ipa,
                "exact_agree": "yes" if exact_agree else "no",
                "canonical_agree": "yes" if canonical_agree else "no",
                "quality_flag": quality_flag,
                "chosen_ipa": "",
                "notes": "",
            }
        )
        prediction_rows.append(
            {
                "recording_id": clip_id,
                "word_id": clip_id,
                "word_title": row.get("audio_filename", ""),
                "word_translation": "",
                "recording_type": "isolation_auto",
                "audio_posix": row.get("audio_posix", ""),
                "audio_windows": row.get("audio_windows", ""),
                "audio_relpath": row.get("audio_relpath", ""),
                "audio_filename": row.get("audio_filename", ""),
                "source_bundle": "phase1a_assistant_review",
                "category_name": "Quizlet phase1a",
                "category_slug": "quizlet-phase1a",
                "speaker_user_id": 0,
                "speaker_name": "C.K.",
                "word_slug": "",
                "recording_slug": f"phase1a-clip-{clip_id:04d}",
                "start_sec": float(row.get("start_sec") or 0.0),
                "end_sec": float(row.get("end_sec") or 0.0),
                "duration_sec": float(row.get("duration_sec") or 0.0),
            }
        )
        audit_rows.append(
            {
                "word_id": clip_id,
                "recording_id": clip_id,
                "word_title": row.get("audio_filename", ""),
                "word_translation": "",
                "recording_type": "isolation_auto",
                "current_recording_ipa": "",
                "current_rule_count": 0,
                "current_canonical": "",
                "csv_candidates": [],
                "assistant_guess_ipa": assistant_guess_ipa,
                "new_mms_recording_ipa": mms_ipa,
                "old_mms_recording_ipa": "",
                "whisper_recording_ipa": whisper_ipa,
                "candidate_buckets": [
                    {
                        "canonical": ipa_to_canonical(value) or value,
                        "chosen_ipa": value,
                        "support_count": 1,
                        "models": [label],
                        "best_rule_count": 0,
                        "csv_support_any": False,
                        "csv_support_typed": False,
                        "supports_new_mms": label == "new_mms",
                        "supports_old_mms": False,
                        "supports_whisper": label == "whisper",
                    }
                    for label, value in (("new_mms", mms_ipa), ("whisper", whisper_ipa))
                    if value
                ],
                "accepted": False,
                "quality_flag": quality_flag,
                "assistant_reason": assistant_reason,
                "assistant_ref_title": ref_title,
                "assistant_ref_translation": ref_translation,
            }
        )
        stats["rows"] += 1
        if assistant_guess_ipa:
            stats["assistant_guess_nonempty"] += 1
        if exact_agree:
            stats["exact_agree"] += 1
        if canonical_agree:
            stats["canonical_agree"] += 1
        stats[f"quality_{quality_flag or 'unset'}"] += 1

    return review_rows, prediction_rows, audit_rows, dict(stats)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    compare_dirs = [Path(item).expanduser().resolve() for item in args.compare_dirs]
    if not compare_dirs:
        compare_dirs = [
            script_dir / "transcription_outputs" / "phase1a_first100_multi_ipa_cpu_20260410-144110",
            script_dir / "transcription_outputs" / "phase1a_101_200_multi_ipa_cpu_20260410-152916",
            script_dir / "transcription_outputs" / "phase1a_201_300_multi_ipa_cpu_20260410-154547",
        ]
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        script_dir / "transcription_outputs" / f"phase1a_unreviewed_001_300_review_{now_slug()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_rows_by_id = load_compare_rows(compare_dirs)
    manual_import = load_manual_import(Path(args.manual_import_jsonl).expanduser().resolve())
    first100_status = load_first100_status(Path(args.first100_status_tsv).expanduser().resolve())
    refs_by_canonical, all_refs = load_ak_reference_forms(Path(args.ak_metadata_jsonl).expanduser().resolve())

    review_rows, prediction_rows, audit_rows, stats = build_rows(
        compare_rows_by_id,
        manual_import,
        first100_status,
        refs_by_canonical,
        all_refs,
    )

    review_tsv = output_dir / "phase1a_unreviewed_001_300_review.tsv"
    predictions_jsonl = output_dir / "phase1a_unreviewed_001_300_review_predictions.jsonl"
    audit_jsonl = output_dir / "phase1a_unreviewed_001_300_review_audit.jsonl"
    summary_json = output_dir / "summary.json"

    write_tsv(
        review_tsv,
        review_rows,
        [
            "clip_id",
            "audio_filename",
            "start_sec",
            "end_sec",
            "duration_sec",
            "assistant_guess_ipa",
            "assistant_reason",
            "assistant_ref_title",
            "assistant_ref_translation",
            "new_mms_ipa",
            "whisper_ipa",
            "exact_agree",
            "canonical_agree",
            "quality_flag",
            "chosen_ipa",
            "notes",
        ],
    )
    write_jsonl(predictions_jsonl, prediction_rows)
    write_jsonl(audit_jsonl, audit_rows)

    open_review_ui_cmd = output_dir / "open_review_ui.cmd"
    open_review_files_cmd = output_dir / "open_review_files.cmd"
    open_review_ui_cmd.write_text(
        "@echo off\n"
        f"bash -lc \"python3 '$Z_WORKSPACE/Speech to Text IPA/serve_unresolved_ipa_review.py' --audit-path '{audit_jsonl.as_posix()}' --predictions-path '{predictions_jsonl.as_posix()}' --speaker-user-id 0 --open-browser\"\n",
        encoding="utf-8",
    )
    clips_dir = script_dir / "transcription_outputs" / "phase1a_processed_multi_ipa_cpu_20260410-133537" / "clips"
    open_review_files_cmd.write_text(
        "@echo off\n"
        f"start \"\" \"{to_windows_path(review_tsv)}\"\n"
        f"start \"\" \"{to_windows_path(clips_dir)}\"\n",
        encoding="utf-8",
    )

    summary = {
        "rows": stats.get("rows", 0),
        "assistant_guess_nonempty": stats.get("assistant_guess_nonempty", 0),
        "exact_agree": stats.get("exact_agree", 0),
        "canonical_agree": stats.get("canonical_agree", 0),
        "quality_counts": {key: value for key, value in stats.items() if key.startswith("quality_")},
        "outputs": {
            "review_tsv": review_tsv.as_posix(),
            "predictions_jsonl": predictions_jsonl.as_posix(),
            "audit_jsonl": audit_jsonl.as_posix(),
            "open_review_ui_cmd": open_review_ui_cmd.as_posix(),
            "open_review_files_cmd": open_review_files_cmd.as_posix(),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
