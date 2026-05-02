#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
ROUND_DIR_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418" / "retranslation_round_01"
SNAPSHOT_PATH_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-20260418-134935.json"
OUTPUT_SNAPSHOT_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-20260418-134935.retranslation-round-01.json"

VALID_ACTIONS = {"", "ok", "skip", "fix_both", "fix_en", "fix_de"}
MERGED_FIELDS = [
    "round_id",
    "shard_id",
    "row_id",
    "issue_type",
    "priority",
    "confidence_score",
    "import_key",
    "sense_index",
    "source_id",
    "source_row_idx",
    "title",
    "definition_tr",
    "definition_en",
    "definition_de",
    "suggested_en_seed",
    "suggested_de_seed",
    "heuristic_flags",
    "reason",
    "review_action",
    "reviewed_en",
    "reviewed_de",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and apply reviewed retranslation shards to a snapshot JSON.")
    parser.add_argument("--round-dir", default=str(ROUND_DIR_DEFAULT))
    parser.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH_DEFAULT))
    parser.add_argument("--output-snapshot-path", default=str(OUTPUT_SNAPSHOT_DEFAULT))
    parser.add_argument("--merged-tsv-path", default="")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def load_review_rows(round_dir: Path) -> list[dict[str, str]]:
    shard_paths = sorted(Path(path) for path in glob.glob(str(round_dir / "shards" / "*.tsv")))
    review_rows: list[dict[str, str]] = []
    for path in shard_paths:
        review_rows.extend(read_tsv(path))
    return review_rows


def main() -> None:
    args = parse_args()
    round_dir = Path(args.round_dir)
    snapshot_path = Path(args.snapshot_path)
    output_snapshot_path = Path(args.output_snapshot_path)
    merged_tsv_path = (
        Path(args.merged_tsv_path)
        if clean_cell(args.merged_tsv_path)
        else round_dir / f"{round_dir.name}_reviewed.tsv"
    )
    summary_path = Path(args.summary_path) if clean_cell(args.summary_path) else round_dir / "apply_summary.json"

    review_rows = load_review_rows(round_dir)
    merged_rows: list[dict[str, str]] = []
    action_counts = Counter()
    duplicate_ids: list[str] = []
    seen_ids: set[tuple[str, str]] = set()
    for row in review_rows:
        action = clean_cell(row.get("review_action")).lower()
        if action not in VALID_ACTIONS:
            raise SystemExit(f"Invalid review_action `{action}` in row {clean_cell(row.get('row_id'))}")
        row_id = (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index")))
        if row_id in seen_ids:
            duplicate_ids.append(f"{row_id[0]}:{row_id[1]}")
        seen_ids.add(row_id)
        merged_rows.append({field: clean_cell(row.get(field)) for field in MERGED_FIELDS})
        action_counts[action or "blank"] += 1
    if duplicate_ids:
        raise SystemExit(f"Duplicate reviewed rows found: {duplicate_ids}")

    write_tsv(merged_tsv_path, MERGED_FIELDS, merged_rows)

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    entries = snapshot.get("entries") or []
    review_by_id = {
        (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index"))): row
        for row in merged_rows
    }

    applied_counts = Counter()
    for entry in entries:
        import_key = clean_cell(entry.get("import_key"))
        senses = entry.get("senses") or []
        for sense_index, sense in enumerate(senses):
            key = (import_key, str(sense_index))
            review = review_by_id.get(key)
            if not review:
                continue
            action = clean_cell(review.get("review_action")).lower()
            reviewed_en = clean_cell(review.get("reviewed_en"))
            reviewed_de = clean_cell(review.get("reviewed_de"))
            if action in {"fix_both", "fix_en"} and reviewed_en:
                (sense.setdefault("translations", {}))["en"] = reviewed_en
                applied_counts["en_updated"] += 1
            if action in {"fix_both", "fix_de"} and reviewed_de:
                (sense.setdefault("translations", {}))["de"] = reviewed_de
                applied_counts["de_updated"] += 1
            if action and action != "skip":
                sense["retranslation_review_action"] = action
                sense["retranslation_review_notes"] = clean_cell(review.get("review_notes"))
                applied_counts["rows_touched"] += 1

    ensure_parent_dir(output_snapshot_path)
    output_snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "round_dir": str(round_dir),
        "snapshot_path": str(snapshot_path),
        "output_snapshot_path": str(output_snapshot_path),
        "merged_tsv_path": str(merged_tsv_path),
        "review_row_count": len(merged_rows),
        "review_action_counts": dict(action_counts),
        "applied_counts": dict(applied_counts),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
