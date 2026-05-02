#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _dictionary_pipeline_common import (
    DEZD_SOURCE_METADATA,
    DEZD_TSV_PATH,
    HARUN_SOURCE_METADATA,
    KAIKKI_HANDOFF_TSV_PATH,
    KAIKKI_SOURCE_METADATA,
    TSV_OUTPUT_DIR,
    clean_cell,
    compact_text_list,
    ensure_parent_dir,
    guess_mt_flag,
    normalize_search_text,
    tsv_rows,
    write_json,
    write_tsv,
)


HARUN_REVIEW_TSV_PATH = TSV_OUTPUT_DIR / "harun_entries_review.tsv"
HARUN_TM_TSV_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
OUTPUT_DIR = TSV_OUTPUT_DIR / "lltools_import"

HARUN_DRAFT_PATH = OUTPUT_DIR / "lltools_dictionary_harun_draft.tsv"
HARUN_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_harun_public.tsv"
DEZD_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_dezd_public.tsv"
KAIKKI_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_kaikki_public.tsv"
COMBINED_DRAFT_PATH = OUTPUT_DIR / "lltools_dictionary_combined_draft.tsv"
COMBINED_PUBLIC_PATH = OUTPUT_DIR / "lltools_dictionary_combined_public.tsv"
COMBINED_MERGED_PUBLIC_TSV_PATH = OUTPUT_DIR / "lltools_dictionary_combined_public_merged_exact.tsv"
COMBINED_MERGED_PUBLIC_JSONL_PATH = OUTPUT_DIR / "lltools_dictionary_combined_public_merged_exact.jsonl"
SOURCES_PATH = OUTPUT_DIR / "lltools_dictionary_sources.tsv"
SUMMARY_PATH = OUTPUT_DIR / "lltools_dictionary_import_summary.json"

HARUN_SOURCE_TAG = "harun"
DEZD_SOURCE_TAG = "dezd"
KAIKKI_SOURCE_TAG = "kaikki"

IMPORT_FIELDS = [
    "entry_id",
    "entry",
    "entry_display",
    "entry_search",
    "parent",
    "parent_search",
    "entry_type",
    "gender_number",
    "definition_tr",
    "definition_tr_source",
    "definition_en",
    "definition_en_source",
    "definition_en_is_machine_generated",
    "definition_en_mt_model",
    "definition_en_review_status",
    "definition_de",
    "definition_de_source",
    "definition_de_is_machine_generated",
    "definition_de_mt_model",
    "definition_de_review_status",
    "source_tag",
    "source_dictionary",
    "source_author",
    "source_title",
    "source_publisher",
    "source_edition",
    "source_year",
    "source_page",
    "source_block_id",
    "source_row_idx",
    "raw_headword",
    "title_keys",
    "dialect_label_source",
    "dialect_label_normalized",
    "entry_review_status",
    "permission_status",
]

SOURCE_FIELDS = [
    "source_tag",
    "source_dictionary",
    "source_author",
    "source_title",
    "source_publisher",
    "source_edition",
    "source_year",
    "dialect_label_source",
    "dialect_label_normalized",
    "permission_status",
    "draft_row_count",
    "public_row_count",
]

MERGED_EXACT_FIELDS = [
    "group_id",
    "entry",
    "entry_display",
    "entry_search",
    "parent",
    "parent_search",
    "entry_variants",
    "entry_type_values",
    "gender_number_values",
    "source_dictionary_count",
    "source_entry_count",
    "has_cross_source_merge",
    "source_tags",
    "source_dictionaries",
    "source_titles",
    "dialect_labels",
    "source_pages",
    "source_entry_ids",
    "definition_tr_sources",
    "definition_en_sources",
    "definition_de_sources",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LL Tools import TSVs from Harun review rows and DEZD source rows.")
    parser.add_argument("--harun-review-path", default=str(HARUN_REVIEW_TSV_PATH))
    parser.add_argument("--harun-translation-memory-path", default=str(HARUN_TM_TSV_PATH))
    parser.add_argument("--dezd-path", default=str(DEZD_TSV_PATH))
    parser.add_argument("--kaikki-path", default=str(KAIKKI_HANDOFF_TSV_PATH))
    parser.add_argument("--harun-permission-status", default=HARUN_SOURCE_METADATA["permission_status"])
    parser.add_argument("--dezd-permission-status", default=DEZD_SOURCE_METADATA["permission_status"])
    parser.add_argument("--kaikki-permission-status", default=KAIKKI_SOURCE_METADATA["permission_status"])
    return parser.parse_args()


def load_translation_memory(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = tsv_rows(path)
    return {clean_cell(row.get("tm_key")): row for row in rows if clean_cell(row.get("tm_key"))}


def source_tag_for_dictionary(source_dictionary: str) -> str:
    source = clean_cell(source_dictionary)
    if source == HARUN_SOURCE_METADATA["source_dictionary"]:
        return HARUN_SOURCE_TAG
    if source == DEZD_SOURCE_METADATA["source_dictionary"]:
        return DEZD_SOURCE_TAG
    if source == KAIKKI_SOURCE_METADATA["source_dictionary"]:
        return KAIKKI_SOURCE_TAG
    fallback = normalize_search_text(source).replace(" ", "_")
    return fallback or "unknown"


def unique_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = clean_cell(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def truthy_flag(value: object) -> bool:
    return clean_cell(value).casefold() in {"1", "true", "yes", "y"}


def row_dialect_label(row: dict[str, str], default_label: str) -> str:
    return clean_cell(row.get("qa_dialect_tag")) or default_label


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def scoped_summary(rows: list[dict[str, str]], field: str, limit: int = 25) -> str:
    values: list[str] = []
    for row in rows:
        value = clean_cell(row.get(field))
        if not value:
            continue
        tag = clean_cell(row.get("source_tag")) or source_tag_for_dictionary(row.get("source_dictionary", ""))
        page = clean_cell(row.get("source_page"))
        scope = f"{tag}@p{page}" if page else tag
        values.append(f"{scope}: {value}")
    return compact_text_list(values, limit=limit, joiner=" || ")


def build_merged_exact_exports(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    # Group by normalized headword while keeping explicit parent/subentry context separate.
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        enriched = dict(row)
        enriched["entry_search"] = clean_cell(enriched.get("entry_search")) or normalize_search_text(clean_cell(enriched.get("entry")) or clean_cell(enriched.get("entry_display")))
        enriched["parent"] = clean_cell(enriched.get("parent"))
        enriched["parent_search"] = clean_cell(enriched.get("parent_search")) or normalize_search_text(enriched["parent"])
        enriched["source_tag"] = clean_cell(enriched.get("source_tag")) or source_tag_for_dictionary(enriched.get("source_dictionary", ""))
        key = (enriched["entry_search"], enriched["parent_search"])
        grouped.setdefault(key, []).append(enriched)

    merged_rows: list[dict[str, str]] = []
    merged_jsonl_rows: list[dict[str, object]] = []

    for entry_search, parent_search in sorted(grouped):
        group_rows = grouped[(entry_search, parent_search)]
        first = group_rows[0]
        entry_variants = unique_values([clean_cell(row.get("entry_display")) or clean_cell(row.get("entry")) for row in group_rows])
        parent_variants = unique_values([clean_cell(row.get("parent")) for row in group_rows])
        entry_types = unique_values([clean_cell(row.get("entry_type")) for row in group_rows])
        gender_values = unique_values([clean_cell(row.get("gender_number")) for row in group_rows])
        source_tags = unique_values([clean_cell(row.get("source_tag")) for row in group_rows])
        source_dictionaries = unique_values([clean_cell(row.get("source_dictionary")) for row in group_rows])
        source_titles = unique_values([clean_cell(row.get("source_title")) for row in group_rows])
        dialect_labels = unique_values([clean_cell(row.get("dialect_label_source")) for row in group_rows])
        source_pages = unique_values(
            [
                f"{clean_cell(row.get('source_tag'))}@p{clean_cell(row.get('source_page'))}"
                for row in group_rows
                if clean_cell(row.get("source_page"))
            ]
        )
        source_entry_ids = [f"{clean_cell(row.get('source_tag'))}:{clean_cell(row.get('entry_id'))}" for row in group_rows if clean_cell(row.get("entry_id"))]
        group_id = entry_search if not parent_search else f"{entry_search}__parent__{parent_search}"

        merged_rows.append(
            {
                "group_id": group_id,
                "entry": clean_cell(first.get("entry")),
                "entry_display": entry_variants[0] if entry_variants else clean_cell(first.get("entry_display")) or clean_cell(first.get("entry")),
                "entry_search": entry_search,
                "parent": parent_variants[0] if parent_variants else "",
                "parent_search": parent_search,
                "entry_variants": " | ".join(entry_variants),
                "entry_type_values": " | ".join(entry_types),
                "gender_number_values": " | ".join(gender_values),
                "source_dictionary_count": str(len(source_dictionaries)),
                "source_entry_count": str(len(group_rows)),
                "has_cross_source_merge": "1" if len(source_dictionaries) > 1 else "0",
                "source_tags": " | ".join(source_tags),
                "source_dictionaries": " | ".join(source_dictionaries),
                "source_titles": " | ".join(source_titles),
                "dialect_labels": " | ".join(dialect_labels),
                "source_pages": " | ".join(source_pages),
                "source_entry_ids": compact_text_list(source_entry_ids, limit=40, joiner=" | "),
                "definition_tr_sources": scoped_summary(group_rows, "definition_tr"),
                "definition_en_sources": scoped_summary(group_rows, "definition_en"),
                "definition_de_sources": scoped_summary(group_rows, "definition_de"),
            }
        )

        merged_jsonl_rows.append(
            {
                "group_id": group_id,
                "entry": clean_cell(first.get("entry")),
                "entry_display": entry_variants[0] if entry_variants else clean_cell(first.get("entry_display")) or clean_cell(first.get("entry")),
                "entry_search": entry_search,
                "parent": parent_variants[0] if parent_variants else "",
                "parent_search": parent_search,
                "entry_variants": entry_variants,
                "entry_type_values": entry_types,
                "gender_number_values": gender_values,
                "source_tags": source_tags,
                "source_dictionaries": source_dictionaries,
                "source_titles": source_titles,
                "dialect_labels": dialect_labels,
                "source_entry_count": len(group_rows),
                "source_dictionary_count": len(source_dictionaries),
                "has_cross_source_merge": len(source_dictionaries) > 1,
                "entries": group_rows,
            }
        )

    return merged_rows, merged_jsonl_rows


def harun_import_rows(review_rows: list[dict[str, str]], tm_rows: dict[str, dict[str, str]], permission_status: str, public_only: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in review_rows:
        if truthy_flag(row.get("qa_exclude_from_export")):
            continue
        review_status = clean_cell(row.get("review_status"))
        if public_only and review_status != "verified_pdf":
            continue

        definition_tr = clean_cell(row.get("definition_tr_display")) or clean_cell(row.get("definition_tr_raw"))
        tm_key = normalize_search_text(definition_tr)
        tm_row = tm_rows.get(tm_key, {})

        definition_en = clean_cell(tm_row.get("definition_en"))
        definition_de = clean_cell(tm_row.get("definition_de"))
        definition_en_source = clean_cell(tm_row.get("translation_source_en"))
        definition_de_source = clean_cell(tm_row.get("translation_source_de"))
        definition_en_review_status = clean_cell(tm_row.get("mt_review_status"))
        definition_de_review_status = clean_cell(tm_row.get("mt_review_status"))

        rows.append(
            {
                "entry_id": clean_cell(row.get("entry_id")),
                "entry": clean_cell(row.get("entry_raw")),
                "entry_display": clean_cell(row.get("entry_display")),
                "entry_search": clean_cell(row.get("entry_search")) or normalize_search_text(clean_cell(row.get("entry_raw"))),
                "parent": "",
                "parent_search": "",
                "entry_type": clean_cell(row.get("entry_type")),
                "gender_number": clean_cell(row.get("gender_number")),
                "definition_tr": definition_tr,
                "definition_tr_source": "source_native",
                "definition_en": definition_en,
                "definition_en_source": definition_en_source,
                "definition_en_is_machine_generated": guess_mt_flag(definition_en_source),
                "definition_en_mt_model": clean_cell(tm_row.get("mt_model_tr_en")),
                "definition_en_review_status": definition_en_review_status,
                "definition_de": definition_de,
                "definition_de_source": definition_de_source,
                "definition_de_is_machine_generated": guess_mt_flag(definition_de_source),
                "definition_de_mt_model": clean_cell(tm_row.get("mt_model_tr_de")) or clean_cell(tm_row.get("mt_model_en_de")),
                "definition_de_review_status": definition_de_review_status,
                "source_tag": HARUN_SOURCE_TAG,
                "source_dictionary": HARUN_SOURCE_METADATA["source_dictionary"],
                "source_author": HARUN_SOURCE_METADATA["source_author"],
                "source_title": HARUN_SOURCE_METADATA["source_title"],
                "source_publisher": HARUN_SOURCE_METADATA["source_publisher"],
                "source_edition": HARUN_SOURCE_METADATA["source_edition"],
                "source_year": HARUN_SOURCE_METADATA["source_year"],
                "source_page": clean_cell(row.get("page_number")),
                "source_block_id": clean_cell(row.get("source_block_id")),
                "source_row_idx": clean_cell(row.get("original_row_index")),
                "raw_headword": clean_cell(row.get("entry_raw")),
                "title_keys": "",
                "dialect_label_source": row_dialect_label(row, HARUN_SOURCE_METADATA["dialect_label_source"]),
                "dialect_label_normalized": HARUN_SOURCE_METADATA["dialect_label_normalized"],
                "entry_review_status": review_status,
                "permission_status": permission_status,
            }
        )
    return rows


def dezd_import_rows(rows: list[dict[str, str]], permission_status: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if truthy_flag(row.get("qa_exclude_from_export")):
            continue
        definition_en = clean_cell(row.get("definition_full_en"))
        definition_de = clean_cell(row.get("definition_full_de"))
        definition_en_source = clean_cell(row.get("definition_full_en_source")) or ("source_native" if definition_en else "")
        definition_de_source = clean_cell(row.get("definition_full_de_source")) or ("source_native" if definition_de else "")
        definition_en_review_status = clean_cell(row.get("definition_full_en_review_status")) or ("source_native" if definition_en else "")
        definition_de_review_status = clean_cell(row.get("definition_full_de_review_status")) or ("source_native" if definition_de else "")
        output.append(
            {
                "entry_id": f"dezd:{clean_cell(row.get('source_row_idx'))}:{normalize_search_text(clean_cell(row.get('entry')))}",
                "entry": clean_cell(row.get("entry")),
                "entry_display": clean_cell(row.get("entry")),
                "entry_search": normalize_search_text(clean_cell(row.get("entry"))),
                "parent": clean_cell(row.get("parent")),
                "parent_search": normalize_search_text(clean_cell(row.get("parent"))),
                "entry_type": clean_cell(row.get("entry_type")) or "word",
                "gender_number": clean_cell(row.get("gender_number")),
                "definition_tr": clean_cell(row.get("definition_full_tr")) or clean_cell(row.get("definition")),
                "definition_tr_source": "source_native",
                "definition_en": definition_en,
                "definition_en_source": definition_en_source,
                "definition_en_is_machine_generated": guess_mt_flag(definition_en_source),
                "definition_en_mt_model": clean_cell(row.get("definition_full_en_mt_model")),
                "definition_en_review_status": definition_en_review_status,
                "definition_de": definition_de,
                "definition_de_source": definition_de_source,
                "definition_de_is_machine_generated": guess_mt_flag(definition_de_source),
                "definition_de_mt_model": clean_cell(row.get("definition_full_de_mt_model")),
                "definition_de_review_status": definition_de_review_status,
                "source_tag": DEZD_SOURCE_TAG,
                "source_dictionary": clean_cell(row.get("source_dictionary")) or DEZD_SOURCE_METADATA["source_dictionary"],
                "source_author": DEZD_SOURCE_METADATA["source_author"],
                "source_title": DEZD_SOURCE_METADATA["source_title"],
                "source_publisher": DEZD_SOURCE_METADATA["source_publisher"],
                "source_edition": DEZD_SOURCE_METADATA["source_edition"],
                "source_year": DEZD_SOURCE_METADATA["source_year"],
                "source_page": clean_cell(row.get("page_number")),
                "source_block_id": "",
                "source_row_idx": clean_cell(row.get("source_row_idx")),
                "raw_headword": clean_cell(row.get("raw_headword")),
                "title_keys": clean_cell(row.get("title_keys")),
                "dialect_label_source": row_dialect_label(row, DEZD_SOURCE_METADATA["dialect_label_source"]),
                "dialect_label_normalized": DEZD_SOURCE_METADATA["dialect_label_normalized"],
                "entry_review_status": "source_native",
                "permission_status": permission_status,
            }
        )
    return output


def kaikki_title_keys(row: dict[str, str]) -> str:
    variants = [clean_cell(value) for value in clean_cell(row.get("headword_variants")).split("|")]
    aliases = [clean_cell(value) for value in clean_cell(row.get("curated_aliases")).split("|")]
    return " | ".join(unique_values(variants + aliases))


def kaikki_import_rows(rows: list[dict[str, str]], permission_status: str) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if clean_cell(row.get("merge_decision")) != "merge":
            continue
        entry = clean_cell(row.get("curated_entry")) or clean_cell(row.get("headword"))
        parent = clean_cell(row.get("curated_parent"))
        definition_en = clean_cell(row.get("curated_definition_en"))
        definition_de = clean_cell(row.get("curated_definition_de"))
        row_permission_status = clean_cell(row.get("permission_status")) or permission_status
        output.append(
            {
                "entry_id": clean_cell(row.get("review_id")),
                "entry": entry,
                "entry_display": entry,
                "entry_search": clean_cell(row.get("headword_search")) or normalize_search_text(entry),
                "parent": parent,
                "parent_search": normalize_search_text(parent),
                "entry_type": clean_cell(row.get("curated_entry_type")) or clean_cell(row.get("entry_type")) or "word",
                "gender_number": clean_cell(row.get("curated_gender_number")) or clean_cell(row.get("gender_number")),
                "definition_tr": clean_cell(row.get("curated_definition_tr")),
                "definition_tr_source": "source_native",
                "definition_en": definition_en,
                "definition_en_source": "source_native" if definition_en else "",
                "definition_en_is_machine_generated": "0" if definition_en else "",
                "definition_en_mt_model": "",
                "definition_en_review_status": "source_native" if definition_en else "",
                "definition_de": definition_de,
                "definition_de_source": "source_native" if definition_de else "",
                "definition_de_is_machine_generated": "0" if definition_de else "",
                "definition_de_mt_model": "",
                "definition_de_review_status": "source_native" if definition_de else "",
                "source_tag": KAIKKI_SOURCE_TAG,
                "source_dictionary": clean_cell(row.get("source_label")) or KAIKKI_SOURCE_METADATA["source_dictionary"],
                "source_author": KAIKKI_SOURCE_METADATA["source_author"],
                "source_title": KAIKKI_SOURCE_METADATA["source_title"],
                "source_publisher": KAIKKI_SOURCE_METADATA["source_publisher"],
                "source_edition": KAIKKI_SOURCE_METADATA["source_edition"],
                "source_year": KAIKKI_SOURCE_METADATA["source_year"],
                "source_page": "",
                "source_block_id": clean_cell(row.get("source_item_id")),
                "source_row_idx": clean_cell(row.get("review_id")) or clean_cell(row.get("source_locator")),
                "raw_headword": clean_cell(row.get("headword")),
                "title_keys": kaikki_title_keys(row),
                "dialect_label_source": clean_cell(row.get("curated_dialect")),
                "dialect_label_normalized": KAIKKI_SOURCE_METADATA["dialect_label_normalized"],
                "entry_review_status": "source_native",
                "permission_status": row_permission_status,
            }
        )
    return output


def source_manifest(
    harun_draft: list[dict[str, str]],
    harun_public: list[dict[str, str]],
    dezd_public: list[dict[str, str]],
    kaikki_public: list[dict[str, str]],
    harun_permission_status: str,
    dezd_permission_status: str,
    kaikki_permission_status: str,
) -> list[dict[str, str]]:
    return [
        {
            "source_tag": HARUN_SOURCE_TAG,
            **HARUN_SOURCE_METADATA,
            "permission_status": harun_permission_status,
            "draft_row_count": str(len(harun_draft)),
            "public_row_count": str(len(harun_public)),
        },
        {
            "source_tag": DEZD_SOURCE_TAG,
            **DEZD_SOURCE_METADATA,
            "permission_status": dezd_permission_status,
            "draft_row_count": str(len(dezd_public)),
            "public_row_count": str(len(dezd_public)),
        },
        {
            "source_tag": KAIKKI_SOURCE_TAG,
            **KAIKKI_SOURCE_METADATA,
            "permission_status": kaikki_permission_status,
            "draft_row_count": str(len(kaikki_public)),
            "public_row_count": str(len(kaikki_public)),
        },
    ]


def main() -> None:
    args = parse_args()
    harun_review_rows = tsv_rows(Path(args.harun_review_path))
    translation_memory = load_translation_memory(Path(args.harun_translation_memory_path))
    dezd_rows = tsv_rows(Path(args.dezd_path))
    kaikki_rows = tsv_rows(Path(args.kaikki_path))

    harun_draft_rows = harun_import_rows(harun_review_rows, translation_memory, args.harun_permission_status, public_only=False)
    harun_public_rows = harun_import_rows(harun_review_rows, translation_memory, args.harun_permission_status, public_only=True)
    dezd_public_rows = dezd_import_rows(dezd_rows, args.dezd_permission_status)
    kaikki_public_rows = kaikki_import_rows(kaikki_rows, args.kaikki_permission_status)

    combined_draft_rows = harun_draft_rows + dezd_public_rows + kaikki_public_rows
    combined_public_rows = harun_public_rows + dezd_public_rows + kaikki_public_rows
    combined_merged_public_rows, combined_merged_public_jsonl_rows = build_merged_exact_exports(combined_public_rows)
    source_rows = source_manifest(
        harun_draft_rows,
        harun_public_rows,
        dezd_public_rows,
        kaikki_public_rows,
        args.harun_permission_status,
        args.dezd_permission_status,
        args.kaikki_permission_status,
    )

    write_tsv(HARUN_DRAFT_PATH, IMPORT_FIELDS, harun_draft_rows)
    write_tsv(HARUN_PUBLIC_PATH, IMPORT_FIELDS, harun_public_rows)
    write_tsv(DEZD_PUBLIC_PATH, IMPORT_FIELDS, dezd_public_rows)
    write_tsv(KAIKKI_PUBLIC_PATH, IMPORT_FIELDS, kaikki_public_rows)
    write_tsv(COMBINED_DRAFT_PATH, IMPORT_FIELDS, combined_draft_rows)
    write_tsv(COMBINED_PUBLIC_PATH, IMPORT_FIELDS, combined_public_rows)
    write_tsv(COMBINED_MERGED_PUBLIC_TSV_PATH, MERGED_EXACT_FIELDS, combined_merged_public_rows)
    write_jsonl(COMBINED_MERGED_PUBLIC_JSONL_PATH, combined_merged_public_jsonl_rows)
    write_tsv(SOURCES_PATH, SOURCE_FIELDS, source_rows)

    summary = {
        "harun_review_path": str(args.harun_review_path),
        "harun_translation_memory_path": str(args.harun_translation_memory_path),
        "dezd_path": str(args.dezd_path),
        "kaikki_path": str(args.kaikki_path),
        "harun_draft_rows": len(harun_draft_rows),
        "harun_public_rows": len(harun_public_rows),
        "dezd_public_rows": len(dezd_public_rows),
        "kaikki_public_rows": len(kaikki_public_rows),
        "combined_draft_rows": len(combined_draft_rows),
        "combined_public_rows": len(combined_public_rows),
        "combined_merged_public_groups": len(combined_merged_public_rows),
        "combined_merged_public_cross_source_groups": sum(1 for row in combined_merged_public_rows if clean_cell(row.get("has_cross_source_merge")) == "1"),
        "outputs": {
            "harun_draft": str(HARUN_DRAFT_PATH),
            "harun_public": str(HARUN_PUBLIC_PATH),
            "dezd_public": str(DEZD_PUBLIC_PATH),
            "kaikki_public": str(KAIKKI_PUBLIC_PATH),
            "combined_draft": str(COMBINED_DRAFT_PATH),
            "combined_public": str(COMBINED_PUBLIC_PATH),
            "combined_merged_public_tsv": str(COMBINED_MERGED_PUBLIC_TSV_PATH),
            "combined_merged_public_jsonl": str(COMBINED_MERGED_PUBLIC_JSONL_PATH),
            "sources": str(SOURCES_PATH),
        },
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
