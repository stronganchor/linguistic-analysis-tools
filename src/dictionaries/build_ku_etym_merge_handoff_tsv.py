#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import SCRIPT_DIR, clean_cell, ensure_parent_dir


QUEUE_PATH_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review" / "open_source_import_review_queue.tsv"
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review"
OUTPUT_TSV_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "ku_etym_merge_handoff.tsv"
OUTPUT_SUMMARY_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "ku_etym_merge_handoff_summary.json"

HANDOFF_FIELDS = [
    "review_id",
    "merge_mode",
    "source_family",
    "source_label",
    "source_path",
    "source_locator",
    "source_item_id",
    "permission_status",
    "headword",
    "headword_variants",
    "headword_search",
    "source_gloss_en",
    "source_etymology",
    "source_notes",
    "source_tags",
    "raw_source_excerpt",
    "master_match_status",
    "master_match_titles",
    "curation_status",
    "merge_decision",
    "curated_entry",
    "curated_parent",
    "curated_entry_type",
    "curated_gender_number",
    "curated_dialect",
    "curated_definition_tr",
    "curated_definition_en",
    "curated_definition_de",
    "curated_etymology_note",
    "curated_aliases",
    "curated_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a clean ku.etymology-only merge handoff TSV from the reviewed open-source queue.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--output-path", default=str(OUTPUT_TSV_PATH_DEFAULT))
    parser.add_argument("--summary-path", default=str(OUTPUT_SUMMARY_PATH_DEFAULT))
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HANDOFF_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in HANDOFF_FIELDS})


def merge_mode_from_notes(notes: str) -> str:
    for part in clean_cell(notes).split(" | "):
        if part.startswith("merge_mode="):
            return part.removeprefix("merge_mode=")
    return ""


def main() -> None:
    args = parse_args()
    rows = read_tsv(Path(args.queue_path))

    handoff_rows: list[dict[str, str]] = []
    for row in rows:
        if clean_cell(row.get("source_family")) != "ku_wiktionary_etymology":
            continue
        if clean_cell(row.get("merge_decision")) != "merge":
            continue
        enriched = dict(row)
        enriched["merge_mode"] = merge_mode_from_notes(row.get("curated_notes", ""))
        handoff_rows.append(enriched)

    handoff_rows.sort(
        key=lambda row: (
            clean_cell(row.get("merge_mode")),
            clean_cell(row.get("curated_entry")) or clean_cell(row.get("headword")),
            clean_cell(row.get("review_id")),
        )
    )

    write_tsv(Path(args.output_path), handoff_rows)

    counters = Counter(clean_cell(row.get("merge_mode")) for row in handoff_rows)
    summary = {
        "queue_path": str(Path(args.queue_path)),
        "output_path": str(Path(args.output_path)),
        "row_count": len(handoff_rows),
        "counts_by_merge_mode": dict(counters),
        "filters": {
            "source_family": "ku_wiktionary_etymology",
            "merge_decision": "merge",
        },
    }
    ensure_parent_dir(Path(args.summary_path))
    Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
