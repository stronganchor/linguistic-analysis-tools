#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ipa_rule_checks import summarize_ipa_rule_issues
from prepare_lltools_ipa_dataset import normalize_ipa_text, to_windows_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_IMPORT_JSONL = Path("$HOME/Downloads/unresolved-ipa-review-import (2).jsonl")
DEFAULT_REVIEW_MANIFEST = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "transcription_outputs"
    / "ck_corpus_dual_ctc_20260411-131711"
    / "mining"
    / "ck_phrase_ipa_review_isolated_20260411"
    / "ck_phrase_review_manifest.tsv"
)
DEFAULT_REVIEW_MANIFEST_ROOT = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "transcription_outputs"
    / "ck_corpus_dual_ctc_20260411-131711"
    / "mining"
)
DEFAULT_EXISTING_CK_ONLY = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "ck-adaptation-tier1-20260410"
    / "manifest_ipa_ck_only.tsv"
)
DEFAULT_EXISTING_CK_AK = (
    ROOT_DIR
    / "Speech to Text IPA"
    / "training_data"
    / "ck-adaptation-tier1-20260410"
    / "manifest_ipa_ck_plus_ak_anchor.tsv"
)

EXCLUDED_TRAINING_SOURCE_IDS = {
    "full_orho_s_story",
}

EXCLUDED_TRAINING_SOURCE_TITLES = {
    "orho's story",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import accepted IPA decisions from a C.K. mining review bundle into training manifests."
    )
    parser.add_argument("--import-jsonl", default=str(DEFAULT_IMPORT_JSONL))
    parser.add_argument("--review-manifest", default=str(DEFAULT_REVIEW_MANIFEST))
    parser.add_argument("--existing-ck-only", default=str(DEFAULT_EXISTING_CK_ONLY))
    parser.add_argument("--existing-ck-ak", default=str(DEFAULT_EXISTING_CK_AK))
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_matching_review_manifest(recording_ids: set[int], preferred: Path | None = None) -> Path | None:
    manifest_paths: list[Path] = []
    if preferred and preferred.exists():
        manifest_paths.append(preferred)
    if DEFAULT_REVIEW_MANIFEST_ROOT.exists():
        for candidate in sorted(DEFAULT_REVIEW_MANIFEST_ROOT.rglob("ck_phrase_review_manifest.tsv")):
            if candidate not in manifest_paths:
                manifest_paths.append(candidate)

    best: tuple[int, int, int, Path] | None = None
    for manifest_path in manifest_paths:
        try:
            rows = read_tsv(manifest_path)
        except FileNotFoundError:
            continue
        manifest_ids = {int(row["recording_id"]) for row in rows if str(row.get("recording_id") or "").strip()}
        if not manifest_ids:
            continue
        match_count = len(recording_ids & manifest_ids)
        if not match_count:
            continue
        missing_count = len(recording_ids - manifest_ids)
        score = (missing_count, len(rows), -match_count, manifest_path)
        if best is None or score < best:
            best = score
        if missing_count == 0 and match_count == len(recording_ids):
            return manifest_path
    return best[3] if best else None


def category_slug(source_group: str) -> str:
    return source_group.replace("_", "-")


def category_name(source_group: str) -> str:
    parts = source_group.split("_")
    return " ".join(part.capitalize() for part in parts)


def normalize_surface(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def main() -> None:
    args = parse_args()
    import_jsonl = Path(args.import_jsonl).expanduser().resolve()
    review_manifest = Path(args.review_manifest).expanduser().resolve()
    existing_ck_only_path = Path(args.existing_ck_only).expanduser().resolve()
    existing_ck_ak_path = Path(args.existing_ck_ak).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT_DIR / "Speech to Text IPA" / "training_data" / f"ck-reviewed-mined-ipa-{now_slug()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows = read_jsonl(import_jsonl)
    accepted_by_id = {
        int(row["recording_id"]): row
        for row in accepted_rows
        if str(row.get("recording_ipa") or "").strip()
    }
    review_rows = read_tsv(review_manifest)
    review_by_id = {int(row["recording_id"]): row for row in review_rows}

    missing_ids = sorted(recording_id for recording_id in accepted_by_id if recording_id not in review_by_id)
    if missing_ids:
        discovered_manifest = find_matching_review_manifest(set(accepted_by_id), preferred=review_manifest)
        if discovered_manifest and discovered_manifest != review_manifest:
            review_manifest = discovered_manifest
            review_rows = read_tsv(review_manifest)
            review_by_id = {int(row["recording_id"]): row for row in review_rows}
            missing_ids = sorted(recording_id for recording_id in accepted_by_id if recording_id not in review_by_id)
        if missing_ids:
            raise SystemExit(f"Accepted recording_ids not found in review manifest: {missing_ids[:10]}")

    bundle_name = review_manifest.parent.name
    import_name = import_jsonl.name

    addon_rows: list[dict[str, Any]] = []
    addon_provenance_rows: list[dict[str, Any]] = []
    excluded_non_ck_rows: list[dict[str, Any]] = []
    issue_counts = Counter()

    for recording_id, accepted in sorted(accepted_by_id.items()):
        review = review_by_id[recording_id]
        target_ipa = normalize_ipa_text(str(accepted.get("recording_ipa") or ""))
        issue_summary = summarize_ipa_rule_issues(target_ipa)
        for issue_type in issue_summary["types"]:
            issue_counts[str(issue_type)] += 1

        predicted_orthography = str(review.get("predicted_orthography") or "").strip()
        source_group = str(review.get("source_group") or "").strip()
        source_title = str(review.get("source_title") or "").strip()
        segment_id = str(review.get("segment_id") or f"recording_{recording_id}")
        bundle_source_id = str(review.get("bundle_source_id") or review.get("source_id") or "").strip()

        if (
            bundle_source_id in EXCLUDED_TRAINING_SOURCE_IDS
            or normalize_surface(source_title) in EXCLUDED_TRAINING_SOURCE_TITLES
        ):
            excluded_non_ck_rows.append(
                {
                    "recording_id": recording_id,
                    "segment_id": segment_id,
                    "bundle_source_id": bundle_source_id,
                    "source_title": source_title,
                    "reason": "excluded_non_ck_speaker_source",
                }
            )
            continue

        ck_row = {
            "split": "train",
            "audio": str(review.get("clip_audio_windows") or ""),
            "audio_windows": str(review.get("clip_audio_windows") or ""),
            "audio_posix": str(review.get("clip_audio_posix") or ""),
            "target_ipa": target_ipa,
            "recording_text": "",
            "recording_translation": "",
            "recording_type": f"{source_group}_reviewed_mined",
            "word_title": predicted_orthography or source_title or segment_id,
            "word_slug": segment_id,
            "categories": category_slug(source_group),
            "category_names": category_name(source_group),
            "speaker_user_id": "0",
            "source_bundle": bundle_name,
            "audio_relpath": str(review.get("clip_audio_relpath") or ""),
            "source_id": segment_id,
            "notes": (
                f"accepted by user in {import_name}; "
                f"candidate_kind={review.get('candidate_kind') or ''}; "
                f"source_title={source_title}; "
                f"orthography_hint={predicted_orthography}"
            ),
        }
        addon_rows.append(ck_row)
        addon_provenance_rows.append(
            {
                "recording_id": recording_id,
                "segment_id": segment_id,
                "source_group": source_group,
                "source_title": source_title,
                "clip_audio_windows": str(review.get("clip_audio_windows") or ""),
                "clip_audio_posix": str(review.get("clip_audio_posix") or ""),
                "start_sec": str(review.get("start_sec") or ""),
                "end_sec": str(review.get("end_sec") or ""),
                "duration_sec": str(review.get("duration_sec") or ""),
                "accepted_ipa": target_ipa,
                "predicted_orthography": predicted_orthography,
                "candidate_kind": str(review.get("candidate_kind") or ""),
                "score": str(review.get("score") or ""),
                "matched_ak_titles": str(review.get("matched_ak_titles") or ""),
                "matched_dict_entries": str(review.get("matched_dict_entries") or ""),
                "ipa_rule_issue_count": int(issue_summary["count"]),
                "ipa_rule_issue_types": "|".join(issue_summary["types"]),
            }
        )

    skipped_rows = [row for row in review_rows if int(row["recording_id"]) not in accepted_by_id]

    fieldnames = [
        "split",
        "audio",
        "audio_windows",
        "audio_posix",
        "target_ipa",
        "recording_text",
        "recording_translation",
        "recording_type",
        "word_title",
        "word_slug",
        "categories",
        "category_names",
        "speaker_user_id",
        "source_bundle",
        "audio_relpath",
        "source_id",
        "notes",
    ]
    addon_manifest = output_dir / "manifest_ipa_reviewed.tsv"
    addon_manifest_new_only = output_dir / "manifest_ipa_reviewed_new_only.tsv"
    addon_provenance = output_dir / "accepted_reviewed_segments.tsv"
    addon_already_present = output_dir / "accepted_already_present_in_ck_only.tsv"
    skipped_tsv = output_dir / "skipped_or_rejected_candidates.tsv"
    excluded_non_ck_tsv = output_dir / "excluded_non_ck_speaker_candidates.tsv"
    accepted_import_copy = output_dir / import_jsonl.name
    summary_json = output_dir / "summary.json"

    write_tsv(addon_manifest, addon_rows, fieldnames)
    write_tsv(addon_provenance, addon_provenance_rows, list(addon_provenance_rows[0].keys()) if addon_provenance_rows else [])
    write_tsv(skipped_tsv, skipped_rows, list(review_rows[0].keys()) if review_rows else [])
    write_tsv(excluded_non_ck_tsv, excluded_non_ck_rows, list(excluded_non_ck_rows[0].keys()) if excluded_non_ck_rows else [])
    write_jsonl(accepted_import_copy, accepted_rows)

    existing_ck_only_rows = read_tsv(existing_ck_only_path)
    existing_ck_ak_rows = read_tsv(existing_ck_ak_path)
    existing_ck_only_ids = {row["source_id"] for row in existing_ck_only_rows}
    existing_ck_ak_ids = {row["source_id"] for row in existing_ck_ak_rows}
    addon_unique_rows = [row for row in addon_rows if row["source_id"] not in existing_ck_only_ids]
    addon_already_present_rows = [row for row in addon_rows if row["source_id"] in existing_ck_only_ids]
    addon_unique_rows_ck_ak = [row for row in addon_rows if row["source_id"] not in existing_ck_ak_ids]

    merged_ck_only = output_dir / "manifest_ipa_ck_only_plus_reviewed.tsv"
    merged_ck_ak = output_dir / "manifest_ipa_ck_plus_ak_anchor_plus_reviewed.tsv"
    write_tsv(addon_manifest_new_only, addon_unique_rows, fieldnames)
    write_tsv(addon_already_present, addon_already_present_rows, fieldnames)
    write_tsv(merged_ck_only, existing_ck_only_rows + addon_unique_rows, fieldnames)
    write_tsv(merged_ck_ak, existing_ck_ak_rows + addon_unique_rows_ck_ak, fieldnames)

    summary = {
        "import_jsonl": import_jsonl.as_posix(),
        "review_manifest": review_manifest.as_posix(),
        "accepted_rows": len(addon_rows),
        "excluded_non_ck_speaker_rows": len(excluded_non_ck_rows),
        "accepted_new_only_rows": len(addon_unique_rows),
        "accepted_already_present_rows": len(addon_already_present_rows),
        "skipped_or_rejected_rows": len(skipped_rows),
        "source_group_counts": dict(Counter(row["categories"] for row in addon_rows)),
        "ipa_rule_issue_type_counts": dict(issue_counts),
        "existing_ck_only_rows": len(existing_ck_only_rows),
        "existing_ck_ak_rows": len(existing_ck_ak_rows),
        "merged_ck_only_rows": len(existing_ck_only_rows) + len(addon_unique_rows),
        "merged_ck_ak_rows": len(existing_ck_ak_rows) + len(addon_unique_rows_ck_ak),
        "outputs": {
            "addon_manifest": addon_manifest.as_posix(),
            "addon_manifest_new_only": addon_manifest_new_only.as_posix(),
            "addon_provenance": addon_provenance.as_posix(),
            "addon_already_present": addon_already_present.as_posix(),
            "skipped_tsv": skipped_tsv.as_posix(),
            "excluded_non_ck_tsv": excluded_non_ck_tsv.as_posix(),
            "accepted_import_copy": accepted_import_copy.as_posix(),
            "merged_ck_only": merged_ck_only.as_posix(),
            "merged_ck_ak": merged_ck_ak.as_posix(),
            "output_dir_windows": to_windows_path(output_dir),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
