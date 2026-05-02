#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, normalize_search_text


def load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def save_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_cell(row.get(name, "")) for name in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply structured manual fixes to Harun page-batch TSV files."
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        help="Path to a JSON spec file. If omitted, the script reads JSON from stdin.",
    )
    return parser.parse_args()


def read_spec(spec_path: Path | None) -> dict[str, object]:
    if spec_path is not None:
        return json.loads(spec_path.read_text(encoding="utf-8"))
    return json.loads(sys.stdin.read())


def row_index_by_id(rows: list[dict[str, str]]) -> dict[str, int]:
    return {row["entry_id"]: index for index, row in enumerate(rows)}


def apply_field_updates(row: dict[str, str], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        row[key] = clean_cell(value)

    if "entry_raw" in updates and "entry_display" not in updates:
        row["entry_display"] = row["entry_raw"]
    if "entry_raw" in updates and "entry_search" not in updates:
        row["entry_search"] = normalize_search_text(row["entry_raw"])
    if "definition_tr_raw" in updates and "definition_tr_display" not in updates:
        row["definition_tr_display"] = row["definition_tr_raw"]


def build_row(template_row: dict[str, str], payload: dict[str, object], next_original_row_index: int) -> dict[str, str]:
    row = dict(template_row)
    row["entry_id"] = clean_cell(payload["entry_id"])
    row["source_block_id"] = clean_cell(
        payload.get("source_block_id", row["entry_id"].split(":")[0])
    )
    row["original_row_index"] = clean_cell(payload.get("original_row_index", next_original_row_index))
    row["old_corrected_nonaffix_children"] = clean_cell(payload.get("old_corrected_nonaffix_children", "0"))
    row["old_corrected_affix_children"] = clean_cell(payload.get("old_corrected_affix_children", "0"))
    row["corrected_child_entries_preview"] = clean_cell(payload.get("corrected_child_entries_preview", ""))
    row["corrected_child_definitions_preview"] = clean_cell(payload.get("corrected_child_definitions_preview", ""))
    row["seed_needs_review"] = clean_cell(payload.get("seed_needs_review", ""))

    apply_field_updates(row, payload)

    if not row.get("entry_display"):
        row["entry_display"] = row["entry_raw"]
    if not row.get("entry_search"):
        row["entry_search"] = normalize_search_text(row["entry_raw"])
    if not row.get("definition_tr_display"):
        row["definition_tr_display"] = row["definition_tr_raw"]
    if not row.get("source_text_raw"):
        row["source_text_raw"] = f"{row['entry_raw']}: {row['definition_tr_raw']}"

    return row


def apply_ops(page_batch_path: Path, ops: dict[str, object]) -> None:
    fieldnames, rows = load_tsv(page_batch_path)
    if not fieldnames:
        raise ValueError(f"{page_batch_path} is empty")

    if "update_all" in ops:
        for row in rows:
            apply_field_updates(row, dict(ops["update_all"]))

    for entry_id in ops.get("delete_rows", []):
        entry_id = clean_cell(entry_id)
        rows = [row for row in rows if row["entry_id"] != entry_id]

    index = row_index_by_id(rows)
    for update in ops.get("update_rows", []):
        entry_id = clean_cell(update["entry_id"])
        if entry_id not in index:
            raise KeyError(f"{page_batch_path}: missing entry_id for update: {entry_id}")
        apply_field_updates(rows[index[entry_id]], dict(update["set"]))

    for insertion in ops.get("insert_after", []):
        index = row_index_by_id(rows)
        after_entry_id = clean_cell(insertion["after_entry_id"])
        template_entry_id = clean_cell(insertion.get("template_entry_id", after_entry_id))
        if after_entry_id not in index:
            raise KeyError(f"{page_batch_path}: missing after_entry_id: {after_entry_id}")
        if template_entry_id not in index:
            raise KeyError(f"{page_batch_path}: missing template_entry_id: {template_entry_id}")
        insert_at = index[after_entry_id] + 1
        next_original_row_index = max(int(clean_cell(row.get("original_row_index", "0")) or "0") for row in rows) + 1
        template_row = rows[index[template_entry_id]]
        new_row = build_row(template_row, dict(insertion["row"]), next_original_row_index)
        rows.insert(insert_at, new_row)

    if "rewrite_page" in ops:
        rewrite = dict(ops["rewrite_page"])
        template_entry_id = clean_cell(rewrite["template_entry_id"])
        index = row_index_by_id(rows)
        if template_entry_id not in index:
            raise KeyError(f"{page_batch_path}: missing template_entry_id for rewrite: {template_entry_id}")
        template_row = rows[index[template_entry_id]]
        next_original_row_index = int(clean_cell(template_row.get("original_row_index", "0")) or "0")
        rewritten_rows: list[dict[str, str]] = []
        for payload in rewrite["rows"]:
            next_original_row_index += 1
            rewritten_rows.append(build_row(template_row, dict(payload), next_original_row_index))
        rows = rewritten_rows

    save_tsv(page_batch_path, fieldnames, rows)


def main() -> None:
    args = parse_args()
    spec = read_spec(args.spec_path)
    for page_batch, ops in spec["page_batches"].items():
        apply_ops(Path(page_batch), dict(ops))


if __name__ == "__main__":
    main()
