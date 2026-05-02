#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_MASTER = Path(
    "$Z_WORKSPACE/Dictionaries/tsv_output/harun_entries_review.tsv"
)
DEFAULT_PAGE_BATCHES = Path(
    "$Z_WORKSPACE/Dictionaries/tsv_output/harun_review/page_batches"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync reviewed Harun page-batch TSVs back into the master entries TSV."
    )
    parser.add_argument(
        "--master-path",
        type=Path,
        default=DEFAULT_MASTER,
        help="Path to harun_entries_review.tsv",
    )
    parser.add_argument(
        "--page-batches-dir",
        type=Path,
        default=DEFAULT_PAGE_BATCHES,
        help="Directory containing page_XXX.tsv review files",
    )
    return parser.parse_args()


def load_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return [], []

    fieldnames = lines[0].split("\t")
    expected_len = len(fieldnames)
    rows: list[dict[str, str]] = []
    for lineno, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) == expected_len + 1 and parts[13] == "" and parts[14].isdigit():
            parts.pop(13)
        if len(parts) != expected_len:
            raise ValueError(
                f"{path} line {lineno} has {len(parts)} fields; expected {expected_len}"
            )
        rows.append(dict(zip(fieldnames, parts)))

    return fieldnames, rows


def save_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    batch_paths = sorted(args.page_batches_dir.glob("page_*.tsv"))
    if not batch_paths:
        raise FileNotFoundError(f"No page-batch files found in {args.page_batches_dir}")

    master_fields: list[str] = []
    synced_rows: list[dict[str, str]] = []
    for batch_path in batch_paths:
        batch_fields, batch_rows = load_tsv(batch_path)
        if not master_fields:
            master_fields = batch_fields
        elif batch_fields != master_fields:
            raise ValueError(f"Field mismatch in {batch_path}")
        synced_rows.extend(batch_rows)

    save_tsv(args.master_path, master_fields, synced_rows)

    status_counts: dict[str, int] = {}
    for row in synced_rows:
        status = row.get("review_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"Rebuilt master from {len(batch_paths)} page-batch files.")
    print(f"rows: {len(synced_rows)}")
    for status, count in sorted(status_counts.items()):
        print(f"{status or '<blank>'}: {count}")


if __name__ == "__main__":
    main()
