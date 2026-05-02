#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import DEZD_TSV_PATH, TSV_OUTPUT_DIR, clean_cell, ensure_parent_dir, write_json, write_tsv


HARUN_TM_TSV_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
SHARD_GLOB_DEFAULT = str(Path(__file__).resolve().parent / "translation_qa_20260416" / "shards" / "mt_*.tsv")
SUMMARY_PATH_DEFAULT = Path(__file__).resolve().parent / "translation_qa_20260416" / "mt_translation_apply_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply MT QA shard reviews back into Harun TM and DEZD TSVs.")
    parser.add_argument("--harun-tm-path", default=str(HARUN_TM_TSV_PATH))
    parser.add_argument("--dezd-path", default=str(DEZD_TSV_PATH))
    parser.add_argument("--shard-glob", default=SHARD_GLOB_DEFAULT)
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH_DEFAULT))
    parser.add_argument("--backup-suffix", default=".pre_translation_qa_apply_20260416.bak")
    parser.add_argument("--corrected-label", default="manual_qa_corrected_20260416")
    parser.add_argument("--checked-label", default="manual_qa_checked_20260416")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def backup_once(path: Path, suffix: str) -> Path:
    backup_path = path.with_name(path.name + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def shard_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        _, shard = read_tsv(path)
        rows.extend(shard)
    return rows


def status_after_review(action: str, prior_source: str, corrected: bool, *, corrected_label: str, checked_label: str) -> tuple[str, str]:
    if corrected:
        return corrected_label, corrected_label
    if action.startswith("ok"):
        return prior_source, checked_label
    return prior_source, ""


def main() -> None:
    args = parse_args()
    harun_tm_path = Path(args.harun_tm_path)
    dezd_path = Path(args.dezd_path)
    summary_path = Path(args.summary_path)

    shard_paths = [Path(path) for path in sorted(glob.glob(args.shard_glob))]
    review_rows = shard_rows(shard_paths)

    harun_fields, harun_rows = read_tsv(harun_tm_path)
    dezd_fields, dezd_rows = read_tsv(dezd_path)
    harun_by_key = {clean_cell(row.get("tm_key")): row for row in harun_rows if clean_cell(row.get("tm_key"))}
    dezd_by_key = {
        f"{clean_cell(row.get('source_row_idx'))}|{clean_cell(row.get('entry'))}": row
        for row in dezd_rows
        if clean_cell(row.get("source_row_idx")) and clean_cell(row.get("entry"))
    }

    applied = Counter()
    for review in review_rows:
        action = clean_cell(review.get("review_action")).lower()
        if action not in {"ok", "fix_en", "fix_de", "fix_both", "skip"}:
            continue
        if action == "skip":
            applied["skipped"] += 1
            continue

        source_dataset = clean_cell(review.get("source_dataset"))
        source_key = clean_cell(review.get("source_key"))
        reviewed_en = clean_cell(review.get("reviewed_en"))
        reviewed_de = clean_cell(review.get("reviewed_de"))
        review_notes = clean_cell(review.get("review_notes"))

        if source_dataset == "harun" and source_key in harun_by_key:
            row = harun_by_key[source_key]
            corrected = False
            if action in {"fix_en", "fix_both"} and reviewed_en:
                row["definition_en"] = reviewed_en
                row["translation_source_en"] = args.corrected_label
                corrected = True
                applied["harun_en_corrected"] += 1
            old_de_source = clean_cell(row.get("translation_source_de"))
            if action in {"fix_de", "fix_both"} and reviewed_de:
                row["definition_de"] = reviewed_de
                row["translation_source_de"] = args.corrected_label
                corrected = True
                applied["harun_de_corrected"] += 1
            if action == "ok":
                row["translation_source_en"] = clean_cell(row.get("translation_source_en")) or "machine_translated_tr_en"
                row["translation_source_de"] = old_de_source or "machine_translated_en_de"
                applied["harun_checked_ok"] += 1
            _, review_status = status_after_review(
                action,
                clean_cell(row.get("translation_source_en")),
                corrected,
                corrected_label=args.corrected_label,
                checked_label=args.checked_label,
            )
            if review_status:
                row["mt_review_status"] = review_status
            if review_notes:
                row["notes"] = (clean_cell(row.get("notes")) + " | " + review_notes).strip(" |")
            applied["harun_rows_touched"] += 1
        elif source_dataset == "dezd" and source_key in dezd_by_key:
            row = dezd_by_key[source_key]
            if action in {"fix_en", "fix_both"} and reviewed_en:
                row["definition_full_en"] = reviewed_en
                row["definition_full_en_source"] = args.corrected_label
                row["definition_full_en_review_status"] = args.corrected_label
                applied["dezd_en_corrected"] += 1
            elif action == "ok":
                row["definition_full_en_review_status"] = args.checked_label
            if action in {"fix_de", "fix_both"} and reviewed_de:
                row["definition_full_de"] = reviewed_de
                row["definition_full_de_source"] = args.corrected_label
                row["definition_full_de_review_status"] = args.corrected_label
                applied["dezd_de_corrected"] += 1
            elif action == "ok":
                row["definition_full_de_review_status"] = args.checked_label
            if action == "ok":
                applied["dezd_checked_ok"] += 1
            if review_notes:
                row["qa_notes"] = (clean_cell(row.get("qa_notes")) + " | " + review_notes).strip(" |")
            applied["dezd_rows_touched"] += 1

    harun_backup = backup_once(harun_tm_path, args.backup_suffix)
    dezd_backup = backup_once(dezd_path, args.backup_suffix)
    write_tsv(harun_tm_path, harun_fields, harun_rows)
    write_tsv(dezd_path, dezd_fields, dezd_rows)

    summary = {
        "review_row_count": len(review_rows),
        "shard_count": len(shard_paths),
        "harun_backup": str(harun_backup),
        "dezd_backup": str(dezd_backup),
        "applied": dict(applied),
    }
    ensure_parent_dir(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
