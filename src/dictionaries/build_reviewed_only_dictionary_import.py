#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from _dictionary_pipeline_common import TSV_OUTPUT_DIR, clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = TSV_OUTPUT_DIR / "lltools_import"
COMBINED_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_combined_public.tsv"
SOURCES_PATH = OUTPUT_DIR / "lltools_dictionary_sources.tsv"
FILTERED_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_combined_public_reviewed_only.tsv"
FILTERED_SOURCES_PATH = OUTPUT_DIR / "lltools_dictionary_sources_reviewed_only.tsv"
SUMMARY_PATH = OUTPUT_DIR / "lltools_dictionary_reviewed_only_summary.json"
SNAPSHOT_PATH = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-reviewed-only-20260417.json"

SOURCE_TAG_MAP = {
    "harun": "harun-turgut",
    "dezd": "dezd",
    "kaikki": "kaikki",
}

APPROVED_MACHINE_REVIEW_STATUSES = {
    "trusted_pre_5_3_import_20260417",
    "manual_qa_checked_20260416",
    "manual_qa_corrected_20260416",
    "manual_qa_corrected_20260417",
    "gemma3_safe_checked_20260417",
    "gemma3_safe_apply_20260417",
    "ollama_qa_checked_20260416",
    "ollama_qa_corrected_20260416",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reviewed-only dictionary import export for zazacaogren.com.")
    parser.add_argument("--input-path", default=str(COMBINED_PUBLIC_PATH))
    parser.add_argument("--sources-path", default=str(SOURCES_PATH))
    parser.add_argument("--output-path", default=str(FILTERED_PUBLIC_PATH))
    parser.add_argument("--output-sources-path", default=str(FILTERED_SOURCES_PATH))
    parser.add_argument("--snapshot-output-path", default=str(SNAPSHOT_PATH))
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    return parser.parse_args()


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def remap_source_tag(value: str) -> str:
    source_tag = clean_cell(value)
    return SOURCE_TAG_MAP.get(source_tag, source_tag)


def keep_definition(source: str, review_status: str) -> bool:
    source_clean = clean_cell(source)
    review_clean = clean_cell(review_status)
    if not source_clean:
        return False
    if "machine" not in source_clean.casefold():
        return True
    return review_clean in APPROVED_MACHINE_REVIEW_STATUSES


def blank_definition_fields(row: dict[str, str], lang: str) -> None:
    row[f"definition_{lang}"] = ""
    row[f"definition_{lang}_source"] = ""
    row[f"definition_{lang}_is_machine_generated"] = ""
    row[f"definition_{lang}_mt_model"] = ""
    row[f"definition_{lang}_review_status"] = ""


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    sources_path = Path(args.sources_path)
    output_path = Path(args.output_path)
    output_sources_path = Path(args.output_sources_path)
    snapshot_output_path = Path(args.snapshot_output_path)
    summary_path = Path(args.summary_path)

    import_fields, rows = read_tsv(input_path)
    source_fields, source_rows = read_tsv(sources_path)

    filtered_rows: list[dict[str, str]] = []
    removed_en = 0
    removed_de = 0
    kept_machine_en = 0
    kept_machine_de = 0

    for raw_row in rows:
        row = dict(raw_row)
        row["source_tag"] = remap_source_tag(row.get("source_tag", ""))

        en_source = clean_cell(row.get("definition_en_source"))
        en_review_status = clean_cell(row.get("definition_en_review_status"))
        if keep_definition(en_source, en_review_status):
            if "machine" in en_source.casefold():
                kept_machine_en += 1
        else:
            if clean_cell(row.get("definition_en")):
                removed_en += 1
            blank_definition_fields(row, "en")

        de_source = clean_cell(row.get("definition_de_source"))
        de_review_status = clean_cell(row.get("definition_de_review_status"))
        if keep_definition(de_source, de_review_status):
            if "machine" in de_source.casefold():
                kept_machine_de += 1
        else:
            if clean_cell(row.get("definition_de")):
                removed_de += 1
            blank_definition_fields(row, "de")

        filtered_rows.append(row)

    filtered_source_rows: list[dict[str, str]] = []
    for raw_row in source_rows:
        row = dict(raw_row)
        row["source_tag"] = remap_source_tag(row.get("source_tag", ""))
        filtered_source_rows.append(row)

    write_tsv(output_path, import_fields, filtered_rows)
    write_tsv(output_sources_path, source_fields, filtered_source_rows)

    subprocess.run(
        [
            "python3",
            str(SCRIPT_DIR / "build_dictionary_snapshot_json.py"),
            "--input-path",
            str(output_path),
            "--sources-path",
            str(output_sources_path),
            "--output-path",
            str(snapshot_output_path),
        ],
        check=True,
    )

    summary = {
        "input_path": str(input_path),
        "sources_path": str(sources_path),
        "output_path": str(output_path),
        "output_sources_path": str(output_sources_path),
        "snapshot_output_path": str(snapshot_output_path),
        "row_count": len(filtered_rows),
        "source_count": len(filtered_source_rows),
        "removed_unreviewed_machine_en": removed_en,
        "removed_unreviewed_machine_de": removed_de,
        "kept_reviewed_machine_en": kept_machine_en,
        "kept_reviewed_machine_de": kept_machine_de,
        "source_tag_map": SOURCE_TAG_MAP,
    }
    ensure_parent_dir(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
