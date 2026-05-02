#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TSV_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418" / "reorganization_candidates.tsv"
ROUND_ROOT_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418"

SHARD_FIELDS = [
    "round_id",
    "shard_id",
    "row_id",
    "issue_type",
    "priority",
    "confidence_score",
    "entry_index",
    "import_key",
    "sense_index",
    "source_id",
    "source_row_idx",
    "title",
    "current_parent",
    "suggested_parent",
    "definition_tr",
    "definition_en",
    "definition_de",
    "heuristic_flags",
    "reason",
    "review_action",
    "reviewed_parent",
    "review_notes",
]

MANIFEST_FIELDS = ["shard_id", "path", "row_count", "titles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reorganization review round from candidate rows.")
    parser.add_argument("--input-path", default=str(INPUT_TSV_DEFAULT))
    parser.add_argument("--round-root", default=str(ROUND_ROOT_DEFAULT))
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--row-count", type=int, default=30)
    parser.add_argument("--shard-count", type=int, default=10)
    parser.add_argument("--exclude-reviewed-glob", default="")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def reviewed_ids(round_root: Path, custom_glob: str = "") -> set[tuple[str, str]]:
    if clean_cell(custom_glob):
        paths = [Path(path) for path in sorted(Path().glob(custom_glob))]
    else:
        paths = sorted(round_root.glob("reorganization_round_*/*_reviewed.tsv"))
    ids: set[tuple[str, str]] = set()
    for path in paths:
        try:
            rows = read_tsv(path)
        except FileNotFoundError:
            continue
        for row in rows:
            if clean_cell(row.get("review_action")):
                ids.add((clean_cell(row.get("import_key")), clean_cell(row.get("sense_index"))))
    return ids


def chunk_rows(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    shard_count = max(1, min(shard_count, len(rows)))
    shard_size = math.ceil(len(rows) / shard_count)
    return [rows[index : index + shard_size] for index in range(0, len(rows), shard_size)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    round_root = Path(args.round_root)
    round_dir = round_root / f"reorganization_{args.round_id}"

    rows = read_tsv(input_path)
    done_ids = reviewed_ids(round_root, args.exclude_reviewed_glob)

    selected: list[dict[str, str]] = []
    for row in rows:
        if clean_cell(row.get("issue_type")) != "reorganize_as_example":
            continue
        row_id = (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index")))
        if row_id in done_ids:
            continue
        selected.append(row)
        if len(selected) >= args.row_count:
            break

    shard_input: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        shard_input.append(
            {
                "round_id": args.round_id,
                "shard_id": "",
                "row_id": f"{args.round_id}_{index:03d}",
                "issue_type": clean_cell(row.get("issue_type")),
                "priority": clean_cell(row.get("priority")),
                "confidence_score": clean_cell(row.get("confidence_score")),
                "entry_index": clean_cell(row.get("entry_index")),
                "import_key": clean_cell(row.get("import_key")),
                "sense_index": clean_cell(row.get("sense_index")),
                "source_id": clean_cell(row.get("source_id")),
                "source_row_idx": clean_cell(row.get("source_row_idx")),
                "title": clean_cell(row.get("title")),
                "current_parent": clean_cell(row.get("current_parent")),
                "suggested_parent": clean_cell(row.get("suggested_parent")),
                "definition_tr": clean_cell(row.get("definition_tr")),
                "definition_en": clean_cell(row.get("definition_en")),
                "definition_de": clean_cell(row.get("definition_de")),
                "heuristic_flags": clean_cell(row.get("heuristic_flags")),
                "reason": clean_cell(row.get("reason")),
                "review_action": "",
                "reviewed_parent": "",
                "review_notes": "",
            }
        )

    groups = chunk_rows(shard_input, args.shard_count)
    ensure_parent_dir(round_dir / "placeholder")
    manifest_rows: list[dict[str, str]] = []
    for index, shard_rows in enumerate(groups, start=1):
        shard_id = f"{args.round_id}_{index:02d}"
        path = round_dir / "shards" / f"{shard_id}.tsv"
        for row in shard_rows:
            row["shard_id"] = shard_id
        write_tsv(path, SHARD_FIELDS, shard_rows)
        manifest_rows.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "row_count": str(len(shard_rows)),
                "titles": " | ".join(clean_cell(row.get("title")) for row in shard_rows),
            }
        )

    manifest_path = round_dir / "shard_manifest.tsv"
    write_tsv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    summary = {
        "round_id": args.round_id,
        "input_path": str(input_path),
        "round_dir": str(round_dir),
        "selected_row_count": len(shard_input),
        "shard_count": len(groups),
        "manifest_path": str(manifest_path),
        "excluded_reviewed_count": len(done_ids),
    }
    write_json(round_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
