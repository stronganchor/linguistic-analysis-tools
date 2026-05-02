#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local LL Tools-style dictionary snapshot JSON from import TSVs.")
    parser.add_argument(
        "--input-path",
        default="$Z_WORKSPACE/Dictionaries/tsv_output/lltools_import/lltools_dictionary_combined_public.tsv",
    )
    parser.add_argument(
        "--sources-path",
        default="$Z_WORKSPACE/Dictionaries/tsv_output/lltools_import/lltools_dictionary_sources.tsv",
    )
    parser.add_argument(
        "--output-path",
        default="$Z_WORKSPACE/Dictionaries/zazacaogren-master-dictionary/ll-tools-dictionary-qa-pass.json",
    )
    parser.add_argument("--site-url", default="https://zazacaogren.com")
    return parser.parse_args()


def clean_cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def stable_import_key(title: str) -> str:
    return "dict:" + hashlib.md5(title.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    rows = read_tsv(Path(args.input_path))
    source_rows = read_tsv(Path(args.sources_path))
    source_defaults = {clean_cell(row.get("source_tag")): clean_cell(row.get("dialect_label_source")) for row in source_rows}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        title = clean_cell(row.get("entry_display")) or clean_cell(row.get("entry"))
        if not title:
            continue
        grouped[title].append(row)

    entries: list[dict[str, object]] = []
    for title in sorted(grouped, key=lambda value: value.casefold()):
        group_rows = grouped[title]
        senses: list[dict[str, object]] = []
        for row in group_rows:
            source_tag = clean_cell(row.get("source_tag"))
            dialect_label = clean_cell(row.get("dialect_label_source"))
            default_dialect_label = source_defaults.get(source_tag, "")
            dialects = [dialect_label] if dialect_label and dialect_label != default_dialect_label else []

            translations: dict[str, str] = {}
            definition_tr = clean_cell(row.get("definition_tr"))
            definition_en = clean_cell(row.get("definition_en"))
            definition_de = clean_cell(row.get("definition_de"))
            if definition_tr:
                translations["tr"] = definition_tr
            if definition_en:
                translations["en"] = definition_en
            if definition_de:
                translations["de"] = definition_de

            review_status = clean_cell(row.get("entry_review_status"))
            needs_review = "" if review_status in {"verified_pdf", "source_native"} else review_status

            senses.append(
                {
                    "definition": definition_tr,
                    "gender_number": clean_cell(row.get("gender_number")),
                    "entry_type": clean_cell(row.get("entry_type")),
                    "parent": clean_cell(row.get("parent")),
                    "needs_review": needs_review,
                    "page_number": clean_cell(row.get("source_page")),
                    "entry_lang": "Zazaki",
                    "def_lang": "tr",
                    "search_terms": clean_cell(row.get("title_keys")),
                    "source_id": source_tag,
                    "source_dictionary": clean_cell(row.get("source_dictionary")),
                    "source_row_idx": clean_cell(row.get("source_row_idx")),
                    "raw_headword": clean_cell(row.get("raw_headword")),
                    "title_keys": clean_cell(row.get("title_keys")),
                    "dialects": dialects,
                    "translations": translations,
                }
            )

        first = senses[0]
        entry_translation = first.get("translations", {}).get("tr", "")
        entries.append(
            {
                "import_key": stable_import_key(title),
                "title": title,
                "status": "public",
                "translation": entry_translation,
                "wordset": None,
                "senses": senses,
            }
        )

    payload = {
        "format": "ll-tools-dictionary-snapshot",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "site_url": args.site_url,
        "source_count": len(source_rows),
        "entry_count": len(entries),
        "sources": source_rows,
        "entries": entries,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "entry_count": len(entries), "source_count": len(source_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
