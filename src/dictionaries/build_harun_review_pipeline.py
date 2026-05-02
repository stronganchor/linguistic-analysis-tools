#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from openpyxl import Workbook
from pypdf import PdfReader

from _dictionary_pipeline_common import (
    HARUN_CORRECTED_TSV_PATH,
    HARUN_ORIGINAL_TSV_PATH,
    HARUN_RECOVERED_TSV_PATH,
    HARUN_PDF_PATH,
    HARUN_PRIORITY_FANOUT_PAGES,
    HARUN_PRIORITY_FLAG_PAGES,
    HARUN_SOURCE_METADATA,
    HARUN_TEXT_DIR,
    TSV_OUTPUT_DIR,
    casefold_strip,
    clean_cell,
    compact_text_list,
    detect_page_bleed,
    ensure_parent_dir,
    is_affix_entry,
    normalize_search_text,
    normalize_spaces,
    priority_bucket,
    priority_rank,
    review_status_seed,
    source_confidence_score,
    tsv_rows,
    write_json,
    write_tsv,
)


OUTPUT_DIR = TSV_OUTPUT_DIR / "harun_review"
SOURCE_BLOCKS_PATH = TSV_OUTPUT_DIR / "harun_source_blocks.tsv"
ENTRIES_REVIEW_PATH = TSV_OUTPUT_DIR / "harun_entries_review.tsv"
PRIORITY_MANIFEST_PATH = TSV_OUTPUT_DIR / "harun_priority_manifest.tsv"
WORKBOOK_PATH = OUTPUT_DIR / "harun_review_queue.xlsx"
TRANSLATION_MEMORY_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
SUMMARY_PATH = OUTPUT_DIR / "harun_review_summary.json"
SOURCE_METADATA_PATH = TSV_OUTPUT_DIR / "harun_source_metadata.json"
PAGE_BATCH_DIR = OUTPUT_DIR / "page_batches"
PAGE_IMAGE_DIR = OUTPUT_DIR / "page_images"

PDFMINER_STRICT_PATH = HARUN_TEXT_DIR / "Palu - Bingöl Harun Turgut_p001-400__pdfminer_strict.txt"
PDFMINER_LOOSE_PATH = HARUN_TEXT_DIR / "Palu - Bingöl Harun Turgut_p001-400__pdfminer_loose.txt"

ENGINE_PREFERENCE = ["pymupdf", "pypdf", "pdfminer_loose", "pdfminer_strict"]

SOURCE_BLOCK_FIELDS = [
    "page_number",
    "source_block_id",
    "source_text_raw",
    "ocr_engine_best",
    "ocr_variants",
    "review_status",
    "review_notes",
    "source_confidence",
    "page_asset_path",
    "seed_entry",
    "seed_definition",
    "seed_entry_type",
    "seed_gender_number",
    "seed_affix_parent",
    "seed_needs_review",
    "old_corrected_nonaffix_children",
    "old_corrected_affix_children",
    "corrected_child_entries_preview",
    "corrected_child_definitions_preview",
    "priority_bucket",
    "priority_rank",
    "original_row_index",
]

ENTRY_FIELDS = [
    "entry_id",
    "source_block_id",
    "page_number",
    "entry_raw",
    "entry_display",
    "entry_search",
    "definition_tr_raw",
    "definition_tr_display",
    "entry_type",
    "gender_number",
    "affix_parent",
    "example_text",
    "example_translation_tr",
    "source_confidence",
    "review_status",
    "review_notes",
    "page_asset_path",
    "ocr_engine_best",
    "ocr_variants",
    "priority_bucket",
    "priority_rank",
    "seed_needs_review",
    "old_corrected_nonaffix_children",
    "old_corrected_affix_children",
    "corrected_child_entries_preview",
    "corrected_child_definitions_preview",
    "source_text_raw",
    "original_row_index",
]

TM_FIELDS = [
    "tm_key",
    "definition_tr",
    "definition_en",
    "definition_de",
    "translation_source_en",
    "translation_source_de",
    "mt_model_tr_en",
    "mt_model_en_de",
    "mt_generated_at",
    "mt_review_status",
    "entry_count",
    "verified_entry_count",
    "entry_ids_preview",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Harun Turgut review pipeline outputs.")
    parser.add_argument(
        "--render-pages",
        choices=["all", "priority", "none"],
        default="all",
        help="Render review page PNGs for all pages, only priority pages, or not at all.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.35,
        help="PyMuPDF render scale for page PNG assets.",
    )
    return parser.parse_args()


def load_pdfminer_pages(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split("\f")
    if parts and not parts[-1].strip():
        parts = parts[:-1]
    return {str(index): parts[index - 1] for index in range(1, len(parts) + 1)}


def load_page_witness_texts(pdf_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    reader = PdfReader(str(pdf_path))
    pypdf_pages: dict[str, str] = {}
    for index, page in enumerate(reader.pages, start=1):
        pypdf_pages[str(index)] = page.extract_text() or ""

    doc = fitz.open(str(pdf_path))
    pymupdf_pages: dict[str, str] = {}
    for index in range(doc.page_count):
        pymupdf_pages[str(index + 1)] = doc[index].get_text("text") or ""

    page_texts = {
        "pypdf": pypdf_pages,
        "pymupdf": pymupdf_pages,
        "pdfminer_strict": load_pdfminer_pages(PDFMINER_STRICT_PATH),
        "pdfminer_loose": load_pdfminer_pages(PDFMINER_LOOSE_PATH),
    }
    return page_texts, doc.page_count


def candidate_entry_searches(entry: str) -> list[str]:
    candidates: list[str] = []
    entry_clean = clean_cell(entry)
    if entry_clean:
        candidates.append(normalize_search_text(entry_clean))
    for piece in [part.strip() for part in entry_clean.split(",")]:
        if piece:
            candidates.append(normalize_search_text(piece))
    if entry_clean.startswith("-") and len(entry_clean) > 1:
        candidates.append(normalize_search_text(entry_clean[1:]))
    if entry_clean.endswith("-") and len(entry_clean) > 1:
        candidates.append(normalize_search_text(entry_clean[:-1]))
    if " " in entry_clean:
        first = entry_clean.split(" ", 1)[0]
        candidates.append(normalize_search_text(first))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def page_lines(page_text: str) -> list[str]:
    return [normalize_spaces(line) for line in page_text.splitlines() if normalize_spaces(line)]


def snippet_for_entry(lines: list[str], entry: str) -> tuple[str, int]:
    searches = candidate_entry_searches(entry)
    if not lines or not searches:
        return "", 0

    best_index = -1
    best_score = 0

    for index, line in enumerate(lines):
        line_key = normalize_search_text(line)
        if not line_key:
            continue
        for candidate in searches:
            score = 0
            if line_key == candidate:
                score = 165 + len(candidate)
            elif line_key.startswith(candidate + " "):
                score = 145 + len(candidate)
            elif f" {candidate} " in f" {line_key} ":
                score = 100 + len(candidate)
            elif line_key.startswith(candidate):
                score = 120 + len(candidate)
            elif f"{candidate} :" in line_key or f"{candidate}:" in line_key:
                score = 90 + len(candidate)
            elif candidate in line_key:
                score = 60 + len(candidate)
            elif candidate.split(" ")[0] and candidate.split(" ")[0] in line_key:
                score = 35 + len(candidate.split(" ")[0])
            if score > best_score:
                best_score = score
                best_index = index

    if best_index < 0:
        return "", 0

    snippet_lines = [lines[best_index]]
    for next_index in range(best_index + 1, min(len(lines), best_index + 5)):
        next_line = lines[next_index]
        if ":" in next_line and next_index > best_index + 1:
            break
        snippet_lines.append(next_line)
    return " | ".join(snippet_lines), best_score


def child_preview(rows: list[dict[str, str]]) -> tuple[str, str]:
    entries = [row.get("entry", "") for row in rows]
    definitions = [row.get("definition", "") for row in rows]
    return compact_text_list(entries, limit=12), compact_text_list(definitions, limit=8)


def render_page_images(pdf_path: Path, output_dir: Path, pages: list[str], scale: float) -> dict[str, str]:
    ensure_parent_dir(output_dir / "keep")
    doc = fitz.open(str(pdf_path))
    page_paths: dict[str, str] = {}
    for page_number in sorted({clean_cell(page) for page in pages if clean_cell(page).isdigit()}, key=lambda item: int(item)):
        output_path = output_dir / f"harun_page_{int(page_number):03d}.png"
        if not output_path.exists():
            pixmap = doc[int(page_number) - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pixmap.save(str(output_path))
        page_paths[page_number] = str(output_path)
    return page_paths


def page_asset_reference(page_number: str, rendered_paths: dict[str, str]) -> str:
    page = clean_cell(page_number)
    if page in rendered_paths:
        return rendered_paths[page]
    if page.isdigit():
        return str(PAGE_IMAGE_DIR / f"harun_page_{int(page):03d}.png")
    return ""


def build_workbook(manifest_rows: list[dict[str, str]], page_rows: dict[str, list[dict[str, str]]]) -> None:
    workbook = Workbook()
    manifest_sheet = workbook.active
    manifest_sheet.title = "priority_manifest"
    manifest_sheet.append(list(manifest_rows[0].keys()) if manifest_rows else ["page_number"])
    for row in manifest_rows:
        manifest_sheet.append([row[key] for key in manifest_rows[0].keys()])
    manifest_sheet.freeze_panes = "A2"

    priority_pages = HARUN_PRIORITY_FANOUT_PAGES + HARUN_PRIORITY_FLAG_PAGES
    for page_number in priority_pages:
        if page_number not in page_rows:
            continue
        sheet = workbook.create_sheet(title=f"p{int(page_number):03d}")
        sheet.append(ENTRY_FIELDS)
        for row in page_rows[page_number]:
            sheet.append([row.get(field, "") for field in ENTRY_FIELDS])
        sheet.freeze_panes = "A2"

    ensure_parent_dir(WORKBOOK_PATH)
    workbook.save(WORKBOOK_PATH)


def build_translation_memory(entry_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    buckets: dict[str, dict[str, object]] = {}
    for row in entry_rows:
        definition = clean_cell(row.get("definition_tr_display", ""))
        if not definition:
            continue
        tm_key = normalize_search_text(definition)
        if not tm_key:
            continue
        bucket = buckets.setdefault(
            tm_key,
            {
                "tm_key": tm_key,
                "definition_tr": definition,
                "definition_en": "",
                "definition_de": "",
                "translation_source_en": "",
                "translation_source_de": "",
                "mt_model_tr_en": "",
                "mt_model_en_de": "",
                "mt_generated_at": "",
                "mt_review_status": "",
                "entry_count": 0,
                "verified_entry_count": 0,
                "entry_ids": [],
            },
        )
        bucket["entry_count"] = int(bucket["entry_count"]) + 1
        if clean_cell(row.get("review_status")) == "verified_pdf":
            bucket["verified_entry_count"] = int(bucket["verified_entry_count"]) + 1
        entry_ids = bucket["entry_ids"]
        assert isinstance(entry_ids, list)
        entry_ids.append(clean_cell(row.get("entry_id", "")))

    rows: list[dict[str, str]] = []
    for bucket in buckets.values():
        entry_ids = bucket.pop("entry_ids")
        assert isinstance(entry_ids, list)
        rows.append(
            {
                **bucket,
                "entry_ids_preview": compact_text_list(entry_ids, limit=8, joiner=" | "),
                "notes": "Seeded from harun_entries_review.tsv",
            }
        )
    rows.sort(key=lambda row: (-int(clean_cell(row["entry_count"]) or "0"), clean_cell(row["tm_key"])))
    return rows


def page_sort_value(page_number: str) -> tuple[int, str]:
    page = clean_cell(page_number)
    if page.isdigit():
        return int(page), ""
    return 9999, page


def main() -> None:
    args = parse_args()

    base_original_rows = [row for row in tsv_rows(HARUN_ORIGINAL_TSV_PATH) if clean_cell(row.get("entry_type")) != "header"]
    recovered_rows = (
        [row for row in tsv_rows(HARUN_RECOVERED_TSV_PATH) if clean_cell(row.get("entry_type")) != "header"]
        if HARUN_RECOVERED_TSV_PATH.exists()
        else []
    )
    corrected_rows = [row for row in tsv_rows(HARUN_CORRECTED_TSV_PATH) if clean_cell(row.get("entry_type")) != "header"]
    original_rows = [row for _, _, row in sorted(
        [("original", index, row) for index, row in enumerate(base_original_rows, start=1)]
        + [("recovered", index, row) for index, row in enumerate(recovered_rows, start=1)],
        key=lambda item: (page_sort_value(item[2].get("page_number", "")), item[0], item[1]),
    )]
    page_texts, pdf_page_count = load_page_witness_texts(HARUN_PDF_PATH)

    corrected_children_by_parent: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    corrected_rows_by_entry: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in corrected_rows:
        page_number = clean_cell(row.get("page_number"))
        parent = clean_cell(row.get("parent"))
        entry = clean_cell(row.get("entry"))
        if parent:
            corrected_children_by_parent[(page_number, parent)].append(row)
        corrected_rows_by_entry[(page_number, entry)].append(row)

    if args.render_pages == "all":
        pages_to_render = sorted({clean_cell(row["page_number"]) for row in original_rows if clean_cell(row["page_number"]).isdigit()}, key=lambda item: int(item))
    elif args.render_pages == "priority":
        pages_to_render = HARUN_PRIORITY_FANOUT_PAGES + HARUN_PRIORITY_FLAG_PAGES
    else:
        pages_to_render = []
    rendered_page_paths = render_page_images(HARUN_PDF_PATH, PAGE_IMAGE_DIR, pages_to_render, args.image_scale) if pages_to_render else {}

    page_ordinals: Counter[str] = Counter()
    source_rows: list[dict[str, str]] = []
    entry_rows: list[dict[str, str]] = []

    for original_index, row in enumerate(original_rows, start=1):
        page_number = clean_cell(row.get("page_number"))
        entry = clean_cell(row.get("entry"))
        definition = clean_cell(row.get("definition"))
        entry_type = clean_cell(row.get("entry_type"))
        gender_number = clean_cell(row.get("gender_number"))
        raw_parent = clean_cell(row.get("parent"))
        affix_parent = raw_parent if is_affix_entry(raw_parent) else ""
        raw_needs_review = clean_cell(row.get("needs_review"))

        page_ordinals[page_number] += 1
        source_block_id = f"harun_p{int(page_number):03d}_b{page_ordinals[page_number]:04d}" if page_number.isdigit() else f"harun_page_unknown_b{original_index:04d}"
        entry_id = f"{source_block_id}:entry"
        source_text_raw = f"{entry}: {definition}" if definition else entry

        variant_snippets: dict[str, str] = {}
        variant_scores: dict[str, int] = {}
        for engine_name in ENGINE_PREFERENCE:
            page_text = page_texts.get(engine_name, {}).get(page_number, "")
            snippet, score = snippet_for_entry(page_lines(page_text), entry)
            if snippet:
                variant_snippets[engine_name] = snippet
                variant_scores[engine_name] = score

        best_engine = "seed_tsv"
        if variant_scores:
            best_engine = max(ENGINE_PREFERENCE, key=lambda name: (variant_scores.get(name, -1), -ENGINE_PREFERENCE.index(name)))
            if best_engine not in variant_scores:
                best_engine = max(variant_scores, key=variant_scores.get)
        witness_count = len(variant_snippets)

        corrected_children = corrected_children_by_parent.get((page_number, entry), [])
        non_affix_children = sum(1 for child in corrected_children if not is_affix_entry(clean_cell(child.get("parent"))))
        affix_children = len(corrected_children) - non_affix_children
        child_entries_preview, child_definitions_preview = child_preview(corrected_children)

        page_bleed = detect_page_bleed(definition, page_number)
        seeded_status = review_status_seed(
            entry=entry,
            definition=definition,
            raw_needs_review=raw_needs_review,
            non_affix_children=non_affix_children,
            page_bleed=page_bleed,
        )
        confidence = source_confidence_score(
            witness_count=witness_count,
            raw_needs_review=raw_needs_review,
            non_affix_children=non_affix_children,
            page_bleed=page_bleed,
            best_engine=best_engine,
        )

        note_parts: list[str] = []
        if raw_needs_review:
            note_parts.append(f"raw_flags={raw_needs_review}")
        if non_affix_children:
            note_parts.append(f"old_corrected_nonaffix_children={non_affix_children}")
        if affix_children:
            note_parts.append(f"old_corrected_affix_children={affix_children}")
        if page_bleed:
            note_parts.append("likely_page_number_bleed")
        if witness_count:
            note_parts.append(f"ocr_witnesses={witness_count}")
        if child_entries_preview:
            note_parts.append(f"child_preview={child_entries_preview}")
        review_notes = " | ".join(note_parts)

        page_asset_path = page_asset_reference(page_number, rendered_page_paths)
        bucket = priority_bucket(page_number)
        rank = priority_rank(page_number)
        variants_payload = json.dumps(variant_snippets, ensure_ascii=False, sort_keys=True)

        source_rows.append(
            {
                "page_number": page_number,
                "source_block_id": source_block_id,
                "source_text_raw": source_text_raw,
                "ocr_engine_best": best_engine,
                "ocr_variants": variants_payload,
                "review_status": seeded_status,
                "review_notes": review_notes,
                "source_confidence": str(confidence),
                "page_asset_path": page_asset_path,
                "seed_entry": entry,
                "seed_definition": definition,
                "seed_entry_type": entry_type,
                "seed_gender_number": gender_number,
                "seed_affix_parent": affix_parent,
                "seed_needs_review": raw_needs_review,
                "old_corrected_nonaffix_children": str(non_affix_children),
                "old_corrected_affix_children": str(affix_children),
                "corrected_child_entries_preview": child_entries_preview,
                "corrected_child_definitions_preview": child_definitions_preview,
                "priority_bucket": bucket,
                "priority_rank": str(rank),
                "original_row_index": str(original_index),
            }
        )

        entry_rows.append(
            {
                "entry_id": entry_id,
                "source_block_id": source_block_id,
                "page_number": page_number,
                "entry_raw": entry,
                "entry_display": entry,
                "entry_search": normalize_search_text(entry),
                "definition_tr_raw": definition,
                "definition_tr_display": definition,
                "entry_type": entry_type,
                "gender_number": gender_number,
                "affix_parent": affix_parent,
                "example_text": child_entries_preview if affix_children and is_affix_entry(entry) else "",
                "example_translation_tr": child_definitions_preview if affix_children and is_affix_entry(entry) else "",
                "source_confidence": str(confidence),
                "review_status": seeded_status,
                "review_notes": review_notes,
                "page_asset_path": page_asset_path,
                "ocr_engine_best": best_engine,
                "ocr_variants": variants_payload,
                "priority_bucket": bucket,
                "priority_rank": str(rank),
                "seed_needs_review": raw_needs_review,
                "old_corrected_nonaffix_children": str(non_affix_children),
                "old_corrected_affix_children": str(affix_children),
                "corrected_child_entries_preview": child_entries_preview,
                "corrected_child_definitions_preview": child_definitions_preview,
                "source_text_raw": source_text_raw,
                "original_row_index": str(original_index),
            }
        )

    source_rows.sort(key=lambda row: (priority_rank(row["page_number"]), int(row["original_row_index"])))
    entry_rows.sort(key=lambda row: (priority_rank(row["page_number"]), int(row["original_row_index"])))

    write_tsv(SOURCE_BLOCKS_PATH, SOURCE_BLOCK_FIELDS, source_rows)
    write_tsv(ENTRIES_REVIEW_PATH, ENTRY_FIELDS, entry_rows)

    page_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in entry_rows:
        page_rows[row["page_number"]].append(row)
    for page_number, rows in page_rows.items():
        batch_path = PAGE_BATCH_DIR / f"page_{int(page_number):03d}.tsv"
        write_tsv(batch_path, ENTRY_FIELDS, rows)

    manifest_rows: list[dict[str, str]] = []
    for page_number in sorted(page_rows, key=priority_rank):
        rows = page_rows[page_number]
        status_counts = Counter(clean_cell(row["review_status"]) for row in rows)
        manifest_rows.append(
            {
                "page_number": page_number,
                "priority_bucket": priority_bucket(page_number),
                "priority_rank": str(priority_rank(page_number)),
                "entry_count": str(len(rows)),
                "needs_split_count": str(status_counts.get("needs_split", 0)),
                "needs_merge_count": str(status_counts.get("needs_merge", 0)),
                "ocr_fix_needed_count": str(status_counts.get("ocr_fix_needed", 0)),
                "hold_count": str(status_counts.get("hold", 0)),
                "page_asset_path": page_asset_reference(page_number, rendered_page_paths),
                "page_batch_tsv": str(PAGE_BATCH_DIR / f"page_{int(page_number):03d}.tsv"),
                "top_entries": compact_text_list([row["entry_raw"] for row in rows], limit=6, joiner=" | "),
            }
        )

    write_tsv(PRIORITY_MANIFEST_PATH, list(manifest_rows[0].keys()) if manifest_rows else ["page_number"], manifest_rows)
    build_workbook(manifest_rows, page_rows)

    translation_memory_rows = build_translation_memory(entry_rows)
    write_tsv(TRANSLATION_MEMORY_PATH, TM_FIELDS, translation_memory_rows)
    write_json(SOURCE_METADATA_PATH, HARUN_SOURCE_METADATA)

    summary = {
        "source_blocks_path": str(SOURCE_BLOCKS_PATH),
        "entries_review_path": str(ENTRIES_REVIEW_PATH),
        "priority_manifest_path": str(PRIORITY_MANIFEST_PATH),
        "workbook_path": str(WORKBOOK_PATH),
        "translation_memory_path": str(TRANSLATION_MEMORY_PATH),
        "page_images_rendered": len(rendered_page_paths),
        "page_batches_written": len(page_rows),
        "pdf_page_count": pdf_page_count,
        "source_block_count": len(source_rows),
        "entry_review_count": len(entry_rows),
        "priority_pages": {
            "fanout": HARUN_PRIORITY_FANOUT_PAGES,
            "flags": HARUN_PRIORITY_FLAG_PAGES,
        },
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
