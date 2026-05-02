#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import fitz
from pypdf import PdfReader

from _dictionary_pipeline_common import (
    HARUN_ORIGINAL_TSV_PATH,
    HARUN_PDF_PATH,
    HARUN_RECOVERED_TSV_PATH,
    TSV_OUTPUT_DIR,
    clean_cell,
    ensure_parent_dir,
    normalize_spaces,
    tsv_rows,
    write_json,
    write_tsv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_PIPELINE_SCRIPT = SCRIPT_DIR / "build_harun_review_pipeline.py"
SYNC_MASTER_SCRIPT = SCRIPT_DIR / "sync_harun_review_batches.py"
BUILD_IMPORT_SCRIPT = SCRIPT_DIR / "build_lltools_dictionary_import_tsv.py"

PAGE_BATCH_DIR = TSV_OUTPUT_DIR / "harun_review" / "page_batches"
PAGE_BATCH_BACKUP_DIR = TSV_OUTPUT_DIR / "harun_review" / "page_batches_pre_missing_recovery_backup"
SUMMARY_PATH = TSV_OUTPUT_DIR / "harun_review" / "missing_pages_recovery_summary.json"

RECOVERED_FIELDS = [
    "entry",
    "definition",
    "gender_number",
    "entry_type",
    "parent",
    "needs_review",
    "page_number",
]

GENDER_RE = re.compile(r"^(?P<entry>.+?)\s*\((?P<gender>m|n|ın)\)$", re.IGNORECASE)
VERB_HINTS = (
    " bıyayış",
    " kerdış",
    " dayış",
    " nayış",
    " untış",
    " kotış",
    " gırotış",
    " ameyış",
    " verdayış",
    " verdiyayış",
    " ardış",
    " vatış",
    " resnayış",
    " şayış",
    " kuayış",
    " bestış",
    " piyayış",
    " qeydayış",
    " eştış",
    " vıyayış",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover missing Harun review pages from raw PDF text.")
    parser.add_argument(
        "--rebuild-pipeline",
        action="store_true",
        help="Regenerate the Harun review pipeline after writing the recovered seed TSV.",
    )
    parser.add_argument(
        "--rebuild-imports",
        action="store_true",
        help="Rebuild LL Tools import TSVs after syncing the recovered page batches.",
    )
    parser.add_argument(
        "--render-pages",
        choices=["all", "priority", "none"],
        default="all",
        help="Pass-through render mode for build_harun_review_pipeline.py when rebuilding.",
    )
    return parser.parse_args()


def page_sort_key(value: str) -> tuple[int, str]:
    page = clean_cell(value)
    if page.isdigit():
        return int(page), ""
    return 9999, page


def present_original_pages() -> set[int]:
    pages: set[int] = set()
    for row in tsv_rows(HARUN_ORIGINAL_TSV_PATH):
        page = clean_cell(row.get("page_number"))
        if page.isdigit():
            pages.add(int(page))
    return pages


def missing_pages(pdf_page_count: int) -> list[int]:
    pages = present_original_pages()
    return [page for page in range(7, pdf_page_count + 1) if page not in pages]


def normalize_line(line: str) -> str:
    return normalize_spaces(line.replace("\u00ad", "-").replace("\ufeff", ""))


def append_fragment(base: str, fragment: str) -> str:
    head = clean_cell(base)
    tail = normalize_line(fragment)
    if not tail:
        return head
    if not head:
        return tail
    if head.endswith("-"):
        return head[:-1] + tail
    return normalize_spaces(f"{head} {tail}")


def looks_like_entry_start(line: str) -> bool:
    if ":" not in line:
        return False
    left, right = line.split(":", 1)
    entry = normalize_spaces(left)
    definition = clean_cell(right)
    if not entry or not definition:
        return False
    if len(entry) > 96 or entry.count(" ") > 14:
        return False
    if re.match(r"^[0-9]+[.)]?$", entry):
        return False
    if any(token in entry for token in (";", "„", '"')):
        return False
    return True


def split_entry_line(line: str) -> tuple[str, str]:
    left, right = line.split(":", 1)
    entry = normalize_spaces(left)
    definition = clean_cell(right)
    if right and not right.startswith(" "):
        entry = f"{entry}:"
    return entry, definition


def infer_entry_type(entry: str, gender: str) -> str:
    if entry.startswith("-"):
        return "suffix"
    if entry.endswith("-"):
        return "prefix"
    if gender:
        return "noun"
    lowered = clean_cell(entry).casefold()
    if lowered.endswith(VERB_HINTS):
        return "verb"
    return "word"


def parse_gender(entry: str) -> tuple[str, str]:
    match = GENDER_RE.match(entry)
    if not match:
        return entry, ""
    return normalize_spaces(match.group("entry")), clean_cell(match.group("gender"))


def make_recovered_row(page_number: int, entry: str, definition: str, flags: list[str]) -> dict[str, str]:
    entry_clean, gender = parse_gender(entry)
    return {
        "entry": entry_clean,
        "definition": clean_cell(definition),
        "gender_number": gender,
        "entry_type": infer_entry_type(entry_clean, gender),
        "parent": "",
        "needs_review": " ".join(dict.fromkeys(flag for flag in flags if flag)),
        "page_number": str(page_number),
    }


def extract_page_lines(reader: PdfReader, doc: fitz.Document, page_number: int) -> list[str]:
    primary = (reader.pages[page_number - 1].extract_text() or "").splitlines()
    lines = [normalize_line(line) for line in primary if normalize_line(line)]
    if lines:
        return lines
    fallback = (doc[page_number - 1].get_text("text") or "").splitlines()
    return [normalize_line(line) for line in fallback if normalize_line(line)]


def recover_rows_for_page(
    page_number: int,
    lines: list[str],
    previous_recovered_row: dict[str, str] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    carry_in: list[str] = []
    current_entry = ""
    current_definition = ""

    for line in lines:
        if looks_like_entry_start(line):
            if current_entry:
                rows.append(make_recovered_row(page_number, current_entry, current_definition, ["recovered_seed"]))
            current_entry, current_definition = split_entry_line(line)
            continue

        if current_entry:
            current_definition = append_fragment(current_definition, line)
        else:
            carry_in.append(line)

    if current_entry:
        rows.append(make_recovered_row(page_number, current_entry, current_definition, ["recovered_seed"]))

    if carry_in and previous_recovered_row is not None:
        previous_recovered_row["definition"] = append_fragment(previous_recovered_row["definition"], " ".join(carry_in))
        existing_flags = clean_cell(previous_recovered_row["needs_review"]).split()
        if "cross_page_recovered" not in existing_flags:
            existing_flags.append("cross_page_recovered")
        previous_recovered_row["needs_review"] = " ".join(existing_flags)

    return rows, carry_in


def backup_existing_page_batches() -> None:
    if PAGE_BATCH_BACKUP_DIR.exists():
        shutil.rmtree(PAGE_BATCH_BACKUP_DIR)
    shutil.copytree(PAGE_BATCH_DIR, PAGE_BATCH_BACKUP_DIR)


def restore_reviewed_page_batches() -> None:
    for batch_path in sorted(PAGE_BATCH_BACKUP_DIR.glob("page_*.tsv")):
        shutil.copy2(batch_path, PAGE_BATCH_DIR / batch_path.name)


def run_script(path: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(path), *args], check=True, cwd=str(SCRIPT_DIR.parent))


def main() -> None:
    args = parse_args()
    reader = PdfReader(str(HARUN_PDF_PATH))
    doc = fitz.open(str(HARUN_PDF_PATH))
    missing = missing_pages(len(reader.pages))

    recovered_rows: list[dict[str, str]] = []
    orphan_carry_in: dict[str, list[str]] = {}
    last_recovered_row: dict[str, str] | None = None
    previous_missing_page: int | None = None

    for page_number in missing:
        page_lines = extract_page_lines(reader, doc, page_number)
        page_rows, carry_in = recover_rows_for_page(
            page_number,
            page_lines,
            last_recovered_row if previous_missing_page == page_number - 1 else None,
        )
        if carry_in and previous_missing_page != page_number - 1:
            orphan_carry_in[str(page_number)] = carry_in
        recovered_rows.extend(page_rows)
        last_recovered_row = page_rows[-1] if page_rows else last_recovered_row
        previous_missing_page = page_number

    write_tsv(HARUN_RECOVERED_TSV_PATH, RECOVERED_FIELDS, recovered_rows)

    summary = {
        "missing_pages_recovered": missing,
        "recovered_page_count": len(missing),
        "recovered_row_count": len(recovered_rows),
        "orphan_carry_in_pages": orphan_carry_in,
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Wrote recovered seed TSV: {HARUN_RECOVERED_TSV_PATH}")
    print(f"Recovered missing pages: {len(missing)}")
    print(f"Recovered rows: {len(recovered_rows)}")
    print(f"Orphan carry-in pages: {sorted(orphan_carry_in)}")

    if not args.rebuild_pipeline:
        return

    backup_existing_page_batches()
    run_script(BUILD_PIPELINE_SCRIPT, "--render-pages", args.render_pages)
    restore_reviewed_page_batches()
    run_script(SYNC_MASTER_SCRIPT)

    if args.rebuild_imports:
        run_script(BUILD_IMPORT_SCRIPT)


if __name__ == "__main__":
    main()
