#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import SCRIPT_DIR, clean_cell, ensure_parent_dir


QUEUE_PATH_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review" / "open_source_import_review_queue.tsv"
KAIKKI_PATH_DEFAULT = SCRIPT_DIR / "open_sources_20260416" / "wikimedia" / "kaikki-zazaki" / "kaikki.org-dictionary-Zazaki.jsonl"

DIALECT_FIELD = "curated_dialect"
NORTHERN_TAG = "Northern-Zazaki"
SOUTHERN_TAG = "Southern-Zazaki"
DIMLI_TAG = "Dimli"
GENERIC_DIALECT_TAG = "dialectal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate Kaikki review-queue rows with explicit single-dialect labels.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--kaikki-path", default=str(KAIKKI_PATH_DEFAULT))
    parser.add_argument("--backup-suffix", default=".pre_kaikki_dialect_20260417.bak")
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fieldnames})


def backup_once(path: Path, suffix: str) -> Path:
    backup_path = path.with_name(path.name + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def ensure_field(fieldnames: list[str], field: str, *, after: str) -> list[str]:
    if field in fieldnames:
        return fieldnames
    if after in fieldnames:
        index = fieldnames.index(after) + 1
        return fieldnames[:index] + [field] + fieldnames[index:]
    return fieldnames + [field]


def has_raw_prefix(raw_glosses: list[str], prefix: str) -> bool:
    target = prefix.casefold()
    return any(clean_cell(gloss).casefold().startswith(target) for gloss in raw_glosses)


def dialect_label_for_record(record: dict[str, object], sense: dict[str, object]) -> str:
    sound_tags = {
        clean_cell(tag)
        for sound in record.get("sounds", []) or []
        for tag in (sound.get("tags", []) or [])
        if clean_cell(tag)
    }
    sense_tags = {clean_cell(tag) for tag in (sense.get("tags", []) or []) if clean_cell(tag)}
    raw_glosses = [clean_cell(gloss) for gloss in (sense.get("raw_glosses", []) or []) if clean_cell(gloss)]

    if DIMLI_TAG in sense_tags or has_raw_prefix(raw_glosses, "(Dimli)"):
        return "Dimili"

    dialect_tags = {tag for tag in sound_tags | sense_tags if tag in {NORTHERN_TAG, SOUTHERN_TAG, GENERIC_DIALECT_TAG}}
    if dialect_tags == {NORTHERN_TAG}:
        return "Dersim"
    if dialect_tags == {SOUTHERN_TAG}:
        return "Dimili"
    return ""


def build_review_id_to_dialect(path: Path) -> dict[str, str]:
    review_id_to_dialect: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            record = json.loads(raw_line)
            for sense_index, sense in enumerate(record.get("senses", []) or [], 1):
                review_id = f"kaikki:{line_number:04d}:{sense_index:02d}"
                review_id_to_dialect[review_id] = dialect_label_for_record(record, sense)
    return review_id_to_dialect


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue_path)
    kaikki_path = Path(args.kaikki_path)

    fieldnames, rows = read_tsv(queue_path)
    fieldnames = ensure_field(fieldnames, DIALECT_FIELD, after="curated_gender_number")
    review_id_to_dialect = build_review_id_to_dialect(kaikki_path)
    backup_path = backup_once(queue_path, args.backup_suffix)

    stats: Counter = Counter()
    for row in rows:
        if clean_cell(row.get("source_family")) != "kaikki":
            row.setdefault(DIALECT_FIELD, clean_cell(row.get(DIALECT_FIELD)))
            continue

        review_id = clean_cell(row.get("review_id"))
        dialect_label = review_id_to_dialect.get(review_id, "")
        row[DIALECT_FIELD] = dialect_label
        if dialect_label:
            stats["labeled_rows"] += 1
            stats[f"label:{dialect_label}"] += 1
        else:
            stats["blank_rows"] += 1

    write_tsv(queue_path, fieldnames, rows)
    print(
        json.dumps(
            {
                "queue_path": str(queue_path),
                "backup_path": str(backup_path),
                "labeled_rows": stats["labeled_rows"],
                "blank_rows": stats["blank_rows"],
                "labels": {key.removeprefix("label:"): value for key, value in stats.items() if key.startswith("label:")},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
