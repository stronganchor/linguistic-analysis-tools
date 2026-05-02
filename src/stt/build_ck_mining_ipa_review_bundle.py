#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues
from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text, to_windows_path
from review_lltools_ipa_comparison import ipa_to_canonical, orthography_to_canonical


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES_TSV = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "transcription_outputs"
    / "ck_corpus_dual_ctc_20260411-131711"
    / "mining"
    / "ck_comprehensible_phrase_candidates_novel.tsv"
)
DEFAULT_AK_TEXT_MANIFEST = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "genc-palu-recording_text-20260409-071931-simple-v002-whisper-medium"
    / "manifest_text.tsv"
)
DEFAULT_AK_IPA_MANIFEST = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "genc-palu-recording_ipa-20260410-065107-whisper-medium-v001"
    / "manifest_ipa.tsv"
)
DEFAULT_CK_IPA_MANIFEST = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "ck-adaptation-tier1-20260410"
    / "manifest_ipa_ck_only.tsv"
)
DEFAULT_CK_TEXT_MANIFEST = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "ck-adaptation-tier1-20260410"
    / "manifest_text_ck_only.tsv"
)


def discover_default_decision_jsonls() -> list[str]:
    decision_paths: list[Path] = []
    base = ROOT_DIR / "Speech to Text IPA" / "transcription_outputs"
    if base.exists():
        for path in sorted(base.glob("**/review_ui_state/unresolved-ipa-review-decisions.jsonl")):
            decision_paths.append(path)
    return [str(path) for path in decision_paths]


def discover_default_exclude_manifests() -> list[str]:
    manifests: list[Path] = []
    for path in [DEFAULT_CK_IPA_MANIFEST, DEFAULT_CK_TEXT_MANIFEST]:
        if path.exists():
            manifests.append(path)
    training_data_root = ROOT_DIR / "Speech to Text IPA" / "training_data"
    if training_data_root.exists():
        for path in sorted(training_data_root.glob("ck-reviewed-mined-ipa-*/manifest_ipa_ck_only_plus_reviewed.tsv")):
            manifests.append(path)
    return [str(path) for path in manifests]


DEFAULT_EXCLUDE_MANIFESTS = discover_default_exclude_manifests()
DEFAULT_EXCLUDE_DECISION_JSONLS = discover_default_decision_jsonls()

EXCLUDED_TRAINING_SOURCE_IDS = {
    "full_orho_s_story",
}

EXCLUDED_TRAINING_SOURCE_TITLES = {
    "orho's story",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an unresolved-IPA review UI bundle from mined C.K. candidate segments."
    )
    parser.add_argument("--candidates-tsv", default=str(DEFAULT_CANDIDATES_TSV))
    parser.add_argument("--ak-text-manifest", default=str(DEFAULT_AK_TEXT_MANIFEST))
    parser.add_argument("--ak-ipa-manifest", default=str(DEFAULT_AK_IPA_MANIFEST))
    parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=list(DEFAULT_EXCLUDE_MANIFESTS),
        help="Manifest whose existing C.K. source_ids should be excluded from review bundles. Can be repeated.",
    )
    parser.add_argument(
        "--exclude-decision-jsonl",
        action="append",
        default=list(DEFAULT_EXCLUDE_DECISION_JSONLS),
        help="Review decision JSONL whose recording_ids should also be excluded, including skips.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to include.")
    parser.add_argument("--output-dir", default="")
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


def normalize_surface(value: str) -> str:
    return normalize_plain_text(value).strip()


def clean_guess(value: str) -> tuple[str, list[str]]:
    text = normalize_ipa_text(value)
    if not text:
        return "", []
    notes: list[str] = []
    replacements = {
        "t̪ʃ": "t̪͡ʃ",
        "d̪ʒ": "d̪͡ʒ",
        "tʃ": "t͡ʃ",
        "dʒ": "d͡ʒ",
        "cç": "c͡ç",
        "tç": "t͡ç",
    }
    for src, dst in replacements.items():
        if src in text:
            text = text.replace(src, dst)
            notes.append(f"replace:{src}>{dst}")
    repaired = basic_repair_ipa_text(text)
    if repaired != text:
        notes.append("basic_repair")
        text = repaired
    issue_summary = summarize_ipa_rule_issues(text)
    if int(issue_summary.get("count") or 0):
        notes.append("rule_issues_remain")
    return text, notes


def stable_id(segment_id: str) -> int:
    return zlib.crc32(segment_id.encode("utf-8")) & 0x7FFFFFFF


def load_ak_reference_maps(
    text_manifest_path: Path,
    ipa_manifest_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    text_rows = read_tsv(text_manifest_path)
    ipa_rows = read_tsv(ipa_manifest_path)
    text_by_relpath = {
        str(row.get("audio_relpath") or "").strip(): row
        for row in text_rows
        if str(row.get("audio_relpath") or "").strip()
    }

    by_orth_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_ipa_canonical: dict[str, list[dict[str, str]]] = defaultdict(list)

    for ipa_row in ipa_rows:
        relpath = str(ipa_row.get("audio_relpath") or "").strip()
        if not relpath or relpath not in text_by_relpath:
            continue
        text_row = text_by_relpath[relpath]
        orth_text = normalize_surface(str(text_row.get("target_text") or text_row.get("recording_text") or ""))
        ipa_text = normalize_ipa_text(str(ipa_row.get("target_ipa") or ""))
        if not orth_text or not ipa_text:
            continue
        entry = {
            "audio_relpath": relpath,
            "orth_text": orth_text,
            "orth_canonical": orthography_to_canonical(orth_text),
            "ipa": ipa_text,
            "ipa_canonical": ipa_to_canonical(ipa_text),
            "word_title": str(text_row.get("word_title") or ipa_row.get("word_title") or ""),
            "recording_type": str(text_row.get("recording_type") or ipa_row.get("recording_type") or ""),
            "recording_translation": str(text_row.get("recording_translation") or ""),
        }
        by_orth_canonical[entry["orth_canonical"]].append(entry)
        by_ipa_canonical[entry["ipa_canonical"]].append(entry)

    return by_orth_canonical, by_ipa_canonical


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def load_excluded_source_ids(manifest_paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        try:
            rows = read_tsv(manifest_path)
        except FileNotFoundError:
            continue
        for row in rows:
            for key in ("source_id", "word_slug"):
                value = str(row.get(key) or "").strip()
                if value:
                    excluded.add(value)
    return excluded


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_excluded_segment_ids_from_decisions(decision_paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for decision_path in decision_paths:
        if not decision_path.exists():
            continue
        bundle_dir = decision_path.parent.parent
        manifest_path = bundle_dir / "ck_phrase_review_manifest.tsv"
        if not manifest_path.exists():
            continue
        manifest_rows = read_tsv(manifest_path)
        manifest_by_recording = {
            str(row.get("recording_id") or "").strip(): str(row.get("segment_id") or "").strip()
            for row in manifest_rows
            if str(row.get("recording_id") or "").strip() and str(row.get("segment_id") or "").strip()
        }
        for row in read_jsonl(decision_path):
            recording_id = str(row.get("recording_id") or "").strip()
            segment_id = manifest_by_recording.get(recording_id)
            if segment_id:
                excluded.add(segment_id)
    return excluded


def build_bundle_rows(
    candidate_rows: list[dict[str, str]],
    ak_by_orth: dict[str, list[dict[str, str]]],
    ak_by_ipa: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    stats = Counter()

    for source_row in candidate_rows:
        segment_id = str(source_row.get("segment_id") or "").strip()
        if not segment_id:
            continue
        recording_id = stable_id(segment_id)
        raw_ipa = normalize_ipa_text(str(source_row.get("predicted_ipa") or ""))
        raw_whisper = normalize_ipa_text(
            str(source_row.get("whisper_predicted_ipa_repaired") or source_row.get("whisper_predicted_ipa") or "")
        )
        assistant_seed = str(source_row.get("assistant_guess_ipa") or raw_ipa or raw_whisper)
        assistant_guess_ipa, repair_notes = clean_guess(assistant_seed)
        assistant_guess_source = str(source_row.get("assistant_guess_source") or "")
        predicted_orthography = normalize_surface(str(source_row.get("predicted_orthography") or ""))
        orth_canonical = str(source_row.get("predicted_orthography_canonical") or orthography_to_canonical(predicted_orthography))
        ipa_canonical = ipa_to_canonical(assistant_guess_ipa or raw_ipa)

        ak_refs_from_orth = ak_by_orth.get(orth_canonical, [])
        ak_refs_from_ipa = ak_by_ipa.get(ipa_canonical, [])
        ak_ref_ipas = unique_keep_order([row["ipa"] for row in ak_refs_from_orth + ak_refs_from_ipa])
        ak_ref_titles = unique_keep_order([row["word_title"] for row in ak_refs_from_orth + ak_refs_from_ipa if row["word_title"]])

        assistant_reason_bits = [
            f"candidate_kind={source_row.get('candidate_kind') or ''}",
            f"score={source_row.get('score') or ''}",
        ]
        if assistant_guess_source:
            assistant_reason_bits.append(f"assistant_source={assistant_guess_source}")
        if repair_notes:
            assistant_reason_bits.append("repairs=" + ",".join(repair_notes))
        if ak_refs_from_orth:
            assistant_reason_bits.append(f"ak_exact_orth={len(ak_refs_from_orth)}")
        if ak_refs_from_ipa:
            assistant_reason_bits.append(f"ak_exact_ipa={len(ak_refs_from_ipa)}")
        assistant_reason = ";".join(bit for bit in assistant_reason_bits if bit)

        duration_sec = float(source_row.get("duration_sec") or 0.0)
        start_sec = float(source_row.get("start_sec") or 0.0)
        end_sec = float(source_row.get("end_sec") or 0.0)
        source_title = str(source_row.get("source_title") or "")
        source_group = str(source_row.get("source_group") or "")
        audio_posix = str(source_row.get("clip_audio_posix") or "")
        audio_windows = str(source_row.get("clip_audio_windows") or "")
        audio_relpath = str(source_row.get("clip_audio_relpath") or "")
        audio_filename = Path(audio_posix).name if audio_posix else Path(audio_windows).name
        subtitle = f"{source_title} | {start_sec:.3f}-{end_sec:.3f}s | {duration_sec:.3f}s"

        review_rows.append(
            {
                "recording_id": recording_id,
                "segment_id": segment_id,
                "source_title": source_title,
                "source_group": source_group,
                "audio_filename": audio_filename,
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "duration_sec": f"{duration_sec:.3f}",
                "predicted_orthography": predicted_orthography,
                "assistant_guess_ipa": assistant_guess_ipa,
                "assistant_reason": assistant_reason,
                "ak_reference_titles": " | ".join(ak_ref_titles),
                "new_mms_ipa": raw_ipa,
                "whisper_ipa": raw_whisper,
                "quality_flag": str(source_row.get("candidate_kind") or ""),
                "matched_ak_titles": str(source_row.get("matched_ak_titles") or ""),
                "matched_dict_entries": str(source_row.get("matched_dict_entries") or ""),
                "chosen_ipa": "",
                "notes": "",
            }
        )

        prediction_rows.append(
            {
                "recording_id": recording_id,
                "word_id": recording_id,
                "word_title": predicted_orthography or source_title or segment_id,
                "word_translation": subtitle,
                "recording_type": str(source_row.get("candidate_kind") or "phrase_candidate"),
                "audio_posix": audio_posix,
                "audio_windows": audio_windows,
                "audio_relpath": audio_relpath,
                "audio_filename": audio_filename,
                "source_bundle": "ck_mining_phrase_review",
                "category_name": source_group,
                "category_slug": source_group.lower().replace(" ", "-"),
                "speaker_user_id": 0,
                "speaker_name": "C.K.",
                "word_slug": segment_id,
                "recording_slug": segment_id,
                "source_title": source_title,
                "segment_id": segment_id,
                "predicted_orthography": predicted_orthography,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration_sec,
            }
        )

        audit_rows.append(
            {
                "word_id": recording_id,
                "recording_id": recording_id,
                "word_title": predicted_orthography or source_title or segment_id,
                "word_translation": subtitle,
                "recording_type": str(source_row.get("candidate_kind") or "phrase_candidate"),
                "current_recording_ipa": "",
                "current_rule_count": 0,
                "current_canonical": "",
                "csv_candidates": ak_ref_ipas,
                "assistant_guess_ipa": assistant_guess_ipa,
                "new_mms_recording_ipa": raw_ipa,
                "old_mms_recording_ipa": "",
                "whisper_recording_ipa": raw_whisper,
                "candidate_buckets": [],
                "accepted": False,
                "quality_flag": str(source_row.get("candidate_kind") or ""),
                "assistant_reason": assistant_reason,
                "assistant_ref_title": " | ".join(ak_ref_titles),
                "assistant_ref_translation": str(source_row.get("matched_dict_entries") or ""),
            }
        )

        manifest_rows.append(
            {
                "recording_id": recording_id,
                "segment_id": segment_id,
                "source_id": segment_id,
                "bundle_source_id": str(source_row.get("source_id") or ""),
                "source_title": source_title,
                "source_group": source_group,
                "candidate_kind": str(source_row.get("candidate_kind") or ""),
                "score": str(source_row.get("score") or ""),
                "tier": str(source_row.get("tier") or ""),
                "clip_audio_windows": audio_windows,
                "clip_audio_posix": audio_posix,
                "clip_audio_relpath": audio_relpath,
                "source_audio_windows": str(source_row.get("source_audio_windows") or ""),
                "source_audio_posix": str(source_row.get("source_audio_posix") or ""),
                "source_audio_relpath": str(source_row.get("source_audio_relpath") or ""),
                "start_sec": f"{start_sec:.3f}",
                "end_sec": f"{end_sec:.3f}",
                "duration_sec": f"{duration_sec:.3f}",
                "predicted_orthography": predicted_orthography,
                "assistant_guess_ipa": assistant_guess_ipa,
                "new_mms_ipa": raw_ipa,
                "whisper_ipa": raw_whisper,
                "ak_reference_ipas": " | ".join(ak_ref_ipas),
                "ak_reference_titles": " | ".join(ak_ref_titles),
                "matched_ak_titles": str(source_row.get("matched_ak_titles") or ""),
                "matched_dict_entries": str(source_row.get("matched_dict_entries") or ""),
            }
        )

        buckets_by_canonical: dict[str, dict[str, Any]] = {}
        for model_name, value in [("new_mms", raw_ipa), ("whisper", raw_whisper)]:
            if not value:
                continue
            canonical = ipa_to_canonical(value) or value
            bucket = buckets_by_canonical.setdefault(
                canonical,
                {
                    "canonical": canonical,
                    "chosen_ipa": value,
                    "support_count": 0,
                    "models": [],
                    "best_rule_count": 0,
                    "csv_support_any": False,
                    "csv_support_typed": False,
                    "supports_new_mms": False,
                    "supports_old_mms": False,
                    "supports_whisper": False,
                },
            )
            if model_name not in bucket["models"]:
                bucket["models"].append(model_name)
                bucket["support_count"] += 1
            if model_name == "new_mms":
                bucket["supports_new_mms"] = True
            if model_name == "whisper":
                bucket["supports_whisper"] = True

        for value in ak_ref_ipas:
            if not value:
                continue
            canonical = ipa_to_canonical(value) or value
            bucket = buckets_by_canonical.setdefault(
                canonical,
                {
                    "canonical": canonical,
                    "chosen_ipa": value,
                    "support_count": 0,
                    "models": [],
                    "best_rule_count": 0,
                    "csv_support_any": False,
                    "csv_support_typed": False,
                    "supports_new_mms": False,
                    "supports_old_mms": False,
                    "supports_whisper": False,
                },
            )
            if "ak_ref" not in bucket["models"]:
                bucket["models"].append("ak_ref")
                bucket["support_count"] += 1

        audit_rows[-1]["candidate_buckets"] = sorted(
            buckets_by_canonical.values(),
            key=lambda item: (-int(item["support_count"]), item["canonical"]),
        )

        stats["rows"] += 1
        stats[f"source_group_{source_group}"] += 1
        stats[f"candidate_kind_{source_row.get('candidate_kind') or 'unknown'}"] += 1
        if assistant_guess_ipa:
            stats["assistant_guess_nonempty"] += 1
        if repair_notes:
            stats["assistant_repaired"] += 1
        if ak_ref_ipas:
            stats["rows_with_ak_reference_ipa"] += 1

    return review_rows, prediction_rows, audit_rows, manifest_rows, dict(stats)


def main() -> None:
    args = parse_args()
    candidates_tsv = Path(args.candidates_tsv).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else candidates_tsv.parent / f"ck_phrase_ipa_review_{now_slug()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_manifest_paths = [Path(path).expanduser().resolve() for path in args.exclude_manifest if path]
    exclude_decision_paths = [Path(path).expanduser().resolve() for path in args.exclude_decision_jsonl if path]
    excluded_source_ids = load_excluded_source_ids(exclude_manifest_paths)
    excluded_segment_ids_from_decisions = load_excluded_segment_ids_from_decisions(exclude_decision_paths)

    candidate_rows_all = read_tsv(candidates_tsv)
    excluded_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    for row in candidate_rows_all:
        segment_id = str(row.get("segment_id") or "").strip()
        bundle_source_id = str(row.get("source_id") or "").strip()
        source_title = normalize_surface(str(row.get("source_title") or ""))
        if (
            (segment_id and segment_id in excluded_source_ids)
            or (bundle_source_id and bundle_source_id in excluded_source_ids)
            or (segment_id and segment_id in excluded_segment_ids_from_decisions)
            or (bundle_source_id and bundle_source_id in EXCLUDED_TRAINING_SOURCE_IDS)
            or (source_title and source_title in EXCLUDED_TRAINING_SOURCE_TITLES)
        ):
            excluded_rows.append(row)
        else:
            candidate_rows.append(row)
    if args.limit > 0:
        candidate_rows = candidate_rows[: args.limit]

    ak_by_orth, ak_by_ipa = load_ak_reference_maps(
        Path(args.ak_text_manifest).expanduser().resolve(),
        Path(args.ak_ipa_manifest).expanduser().resolve(),
    )
    review_rows, prediction_rows, audit_rows, manifest_rows, stats = build_bundle_rows(
        candidate_rows,
        ak_by_orth,
        ak_by_ipa,
    )

    review_tsv = output_dir / "ck_phrase_ipa_review.tsv"
    manifest_tsv = output_dir / "ck_phrase_review_manifest.tsv"
    predictions_jsonl = output_dir / "ck_phrase_ipa_review_predictions.jsonl"
    audit_jsonl = output_dir / "ck_phrase_ipa_review_audit.jsonl"
    excluded_tsv = output_dir / "excluded_already_trusted.tsv"
    summary_json = output_dir / "summary.json"
    open_review_ui_cmd = output_dir / "open_review_ui.cmd"
    open_review_files_cmd = output_dir / "open_review_files.cmd"
    review_ui_state_dir = output_dir / "review_ui_state"

    write_tsv(
        review_tsv,
        review_rows,
        [
            "recording_id",
            "segment_id",
            "source_title",
            "source_group",
            "audio_filename",
            "start_sec",
            "end_sec",
            "duration_sec",
            "predicted_orthography",
            "assistant_guess_ipa",
            "assistant_reason",
            "ak_reference_titles",
            "new_mms_ipa",
            "whisper_ipa",
            "quality_flag",
            "matched_ak_titles",
            "matched_dict_entries",
            "chosen_ipa",
            "notes",
        ],
    )
    write_tsv(review_tsv.with_name("ck_phrase_ipa_review_pending.tsv"), review_rows, list(review_rows[0].keys()) if review_rows else [])
    write_tsv(manifest_tsv, manifest_rows, list(manifest_rows[0].keys()) if manifest_rows else [])
    write_tsv(excluded_tsv, excluded_rows, list(candidate_rows_all[0].keys()) if candidate_rows_all else [])
    write_jsonl(predictions_jsonl, prediction_rows)
    write_jsonl(audit_jsonl, audit_rows)

    open_review_ui_cmd.write_text(
        "@echo off\n"
        f"bash -lc \"python3 '$Z_WORKSPACE/Speech to Text IPA/serve_unresolved_ipa_review.py' --audit-path '{audit_jsonl.as_posix()}' --predictions-path '{predictions_jsonl.as_posix()}' --output-dir '{review_ui_state_dir.as_posix()}' --speaker-user-id 0 --include-live-csv-covered-rows --open-browser\"\n",
        encoding="utf-8",
    )
    open_review_files_cmd.write_text(
        "@echo off\n"
        f"start \"\" \"{to_windows_path(review_tsv)}\"\n"
        f"start \"\" \"{to_windows_path(manifest_tsv)}\"\n",
        encoding="utf-8",
    )

    summary = {
        "rows": stats.get("rows", 0),
        "assistant_guess_nonempty": stats.get("assistant_guess_nonempty", 0),
        "assistant_repaired": stats.get("assistant_repaired", 0),
        "rows_with_ak_reference_ipa": stats.get("rows_with_ak_reference_ipa", 0),
        "excluded_already_trusted_rows": len(excluded_rows),
        "excluded_decision_jsonls": [path.as_posix() for path in exclude_decision_paths],
        "source_group_counts": {key.removeprefix("source_group_"): value for key, value in stats.items() if key.startswith("source_group_")},
        "candidate_kind_counts": {key.removeprefix("candidate_kind_"): value for key, value in stats.items() if key.startswith("candidate_kind_")},
        "inputs": {
            "candidates_tsv": candidates_tsv.as_posix(),
            "ak_text_manifest": Path(args.ak_text_manifest).expanduser().resolve().as_posix(),
            "ak_ipa_manifest": Path(args.ak_ipa_manifest).expanduser().resolve().as_posix(),
            "exclude_manifests": [path.as_posix() for path in exclude_manifest_paths],
        },
        "outputs": {
            "review_tsv": review_tsv.as_posix(),
            "manifest_tsv": manifest_tsv.as_posix(),
            "predictions_jsonl": predictions_jsonl.as_posix(),
            "audit_jsonl": audit_jsonl.as_posix(),
            "excluded_tsv": excluded_tsv.as_posix(),
            "review_ui_state_dir": review_ui_state_dir.as_posix(),
            "open_review_ui_cmd": open_review_ui_cmd.as_posix(),
            "open_review_files_cmd": open_review_files_cmd.as_posix(),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
