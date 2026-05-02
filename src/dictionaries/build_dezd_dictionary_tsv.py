#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import html
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Manidar/DEZD-Dictionary-Kirmancki"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "tsv_output" / "DEZD - Kirmancki Dictionary.tsv"


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_text(value: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(value or ""))
    text = text.replace("\t", " ")
    return normalize_spaces(text)


def strip_metadata_blocks(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return normalize_spaces(text)


def headword_segment(value: str) -> str:
    text = clean_text(value)
    bracket_index = text.find("[")
    if bracket_index != -1:
        text = text[:bracket_index]
    return normalize_spaces(text)


def clean_base_entry_candidate(value: str) -> str:
    text = headword_segment(value)
    # Treat only spaced parenthetical blocks as alias/context notes; keep
    # inline optional letters such as `ay(ı)v` in the base headword.
    text = re.sub(r"\s+\([^)]*\)", " ", text)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s*[/;]\s*", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = normalize_spaces(text.strip(" ,;/:-"))
    return text


def clean_variant_candidate(value: str) -> str:
    text = clean_text(value)
    text = text.replace("(", "").replace(")", "")
    text = re.sub(r"\s*[/;]\s*", " ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = normalize_spaces(text.strip(" ,;/:-"))
    return text


def should_keep_entry(value: str) -> bool:
    if not value:
        return False
    if any(char.isdigit() for char in value):
        return False
    if len(value) > 96:
        return False
    if looks_like_non_lexeme(value):
        return False
    return True


def normalize_variant_key(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").casefold()
    text = re.sub(r"[^0-9a-zçğıöşüêîûâ']", "", text)
    return text


def variant_char_overlap(candidate_key: str, base_key: str) -> float:
    candidate_chars = set(candidate_key)
    base_chars = set(base_key)
    if not candidate_chars or not base_chars:
        return 0.0
    return len(candidate_chars & base_chars) / len(candidate_chars | base_chars)


CLAUSE_STARTERS = {
    "ana",
    "anu",
    "ez",
    "ma",
    "na",
    "ni",
    "no",
    "nu",
    "o",
    "ow",
    "şıma",
    "tı",
    "to",
    "ê",
    "êya",
}


def looks_like_non_lexeme(value: str) -> bool:
    text = normalize_spaces(value)
    words = text.split()
    if len(words) >= 3 and any(mark in text for mark in ("?", "!")):
        return True
    if len(words) >= 2 and "-" in text:
        return True
    if len(words) >= 4 and words[0].casefold() in CLAUSE_STARTERS:
        return True
    if len(words) >= 3 and words[-1].casefold() == "esto":
        return True
    return False


def plausible_variant(candidate: str, base: str) -> bool:
    candidate_key = normalize_variant_key(candidate)
    base_variants = [normalize_variant_key(part) for part in re.split(r"\s*,\s*", base) if normalize_variant_key(part)]
    if not candidate_key or not base_variants:
        return False
    for base_key in base_variants:
        if candidate_key == base_key:
            return False
        if candidate_key in base_key or base_key in candidate_key:
            return True
        similarity = SequenceMatcher(None, candidate_key, base_key).ratio()
        overlap = variant_char_overlap(candidate_key, base_key)
        if similarity >= 0.33 and overlap >= 0.30:
            return True
    return False


def extract_parenthetical_variants(value: str) -> list[str]:
    variants: list[str] = []
    base = clean_base_entry_candidate(value)
    segment = headword_segment(value)
    match = re.search(r"\s+\(([^)]*)\)", segment)
    if not match:
        return variants
    block = normalize_spaces(match.group(1))
    if not block or len(block) > 96:
        return variants
    parts = re.split(r"\s*,\s*|\s*;\s*", block)
    for part in parts:
        candidate = clean_variant_candidate(part)
        if should_keep_entry(candidate) and plausible_variant(candidate, base):
            variants.append(candidate)
    return variants


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def extract_entries(raw_headword: str) -> list[tuple[str, str]]:
    base = clean_base_entry_candidate(raw_headword)
    candidates: list[tuple[str, str]] = []
    if should_keep_entry(base):
        candidates.append((base, ""))
    for variant in extract_parenthetical_variants(raw_headword):
        if variant != base:
            candidates.append((variant, base))
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry, parent in candidates:
        key = (entry.casefold(), parent.casefold())
        if key in seen:
            continue
        seen.add(key)
        unique.append((entry, parent))
    return unique


def clean_gloss_text(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\b\d+\)", " | ", text)
    text = re.sub(r"\(([^)]*)\)", r" | \1 ", text)
    return normalize_spaces(text)


def looks_like_title_key(value: str) -> bool:
    if not value:
        return False
    if len(value) > 48:
        return False
    if value.count(" ") > 4:
        return False
    return True


def extract_title_keys(*glosses: str) -> list[str]:
    results: list[str] = []
    for gloss in glosses:
        cleaned = clean_gloss_text(gloss)
        if not cleaned:
            continue
        parts = re.split(r"\s*\|\s*|[;/]", cleaned)
        for part in parts:
            bits = [part]
            if "," in part:
                bits.extend(item.strip() for item in part.split(","))
            for bit in bits:
                candidate = normalize_spaces(bit.strip(" ,.;:"))
                if looks_like_title_key(candidate):
                    results.append(candidate)
    return dedupe_preserve_order(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Manidar DEZD Kirmancki dataset as repo-native TSV dictionary.")
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    fieldnames = [
        "entry",
        "definition",
        "gender_number",
        "entry_type",
        "parent",
        "needs_review",
        "page_number",
        "source_dictionary",
        "source_row_idx",
        "raw_headword",
        "title_keys",
        "definition_full_tr",
        "definition_full_de",
        "definition_full_en",
    ]

    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(dataset):
        raw_headword = str(row.get("Kırmancki") or "")
        gloss_tr = clean_text(str(row.get("Türkçe") or ""))
        gloss_de = clean_text(str(row.get("Deutsch") or ""))
        gloss_en = clean_text(str(row.get("English") or ""))
        entries = extract_entries(raw_headword)
        if not entries:
            continue

        title_keys = "|".join(extract_title_keys(gloss_tr, gloss_de, gloss_en))
        definition = gloss_tr or gloss_en or gloss_de
        for entry, parent in entries:
            rows.append(
                {
                    "entry": entry,
                    "definition": definition,
                    "gender_number": "",
                    "entry_type": "word",
                    "parent": parent,
                    "needs_review": "",
                    "page_number": "",
                    "source_dictionary": "DEZD - Kirmancki Dictionary",
                    "source_row_idx": str(row_index),
                    "raw_headword": clean_text(raw_headword),
                    "title_keys": title_keys,
                    "definition_full_tr": gloss_tr,
                    "definition_full_de": gloss_de,
                    "definition_full_en": gloss_en,
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "output_path": str(output_path),
        "rows_written": len(rows),
        "unique_entries": len({str(row["entry"]).casefold() for row in rows}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
