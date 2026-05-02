#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


def truthy_flag(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_row(fieldnames: list[str], row: dict[str | None, str | list[str]]) -> dict[str, str]:
    normalized = {name: str(row.get(name, "") or "") for name in fieldnames}
    overflow = row.get(None) or []
    if overflow:
        extras = [str(value) for value in overflow if str(value).strip()]
        if extras:
            # Some shard edits ended up one column to the right, e.g.:
            # qa_dialect_tag="" qa_exclude_from_export="" qa_notes="1" overflow=["note"]
            if not normalized.get("qa_exclude_from_export") and truthy_flag(normalized.get("qa_notes", "")):
                normalized["qa_exclude_from_export"] = normalized["qa_notes"]
                normalized["qa_notes"] = " ".join(extras)
            else:
                tail = " ".join(extras)
                normalized["qa_notes"] = f"{normalized.get('qa_notes', '').strip()} {tail}".strip()
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge QA shard TSVs back into their source dictionary TSVs.")
    parser.add_argument(
        "--manifest-path",
        default="$Z_WORKSPACE/Dictionaries/qa_pass_20260416/shard_manifest.tsv",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".pre_qa_pass_20260416.bak",
        help="Suffix used for one-time source backups before overwriting.",
    )
    return parser.parse_args()


def read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [normalize_row(fieldnames, row) for row in reader]
        return fieldnames, rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = [dict(row) for row in csv.DictReader(handle, delimiter="\t")]

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[row["input_path"]].append(row)

    for input_path_str, shard_rows in grouped.items():
        input_path = Path(input_path_str)
        shard_rows.sort(key=lambda row: int(row["start_row_1based"]))

        merged_fieldnames: list[str] = []
        merged_rows: list[dict[str, str]] = []
        for shard_row in shard_rows:
            fieldnames, rows = read_tsv_rows(Path(shard_row["path"]))
            if not merged_fieldnames:
                merged_fieldnames = fieldnames
            merged_rows.extend(rows)

        backup_path = input_path.with_name(input_path.name + args.backup_suffix)
        if not backup_path.exists():
            shutil.copy2(input_path, backup_path)

        write_tsv(input_path, merged_fieldnames, merged_rows)

        excluded = sum(1 for row in merged_rows if truthy_flag(row.get("qa_exclude_from_export", "")))
        dialect_overrides = sum(1 for row in merged_rows if str(row.get("qa_dialect_tag", "")).strip())
        print(
            f"{input_path} rows={len(merged_rows)} excluded={excluded} dialect_overrides={dialect_overrides} backup={backup_path}"
        )


if __name__ == "__main__":
    main()
