#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TSV_OUTPUT_DIR = SCRIPT_DIR / "tsv_output"

HARUN_PDF_PATH = SCRIPT_DIR / "Palu - Bingöl Harun Turgut.pdf"
HARUN_ORIGINAL_TSV_PATH = TSV_OUTPUT_DIR / "Palu - Bingöl Harun Turgut.tsv"
HARUN_CORRECTED_TSV_PATH = TSV_OUTPUT_DIR / "Palu - Bingöl Harun Turgut - corrected.tsv"
HARUN_RECOVERED_TSV_PATH = TSV_OUTPUT_DIR / "Palu - Bingöl Harun Turgut - recovered-seed.tsv"
HARUN_TEXT_DIR = SCRIPT_DIR / "Palu - Bingöl Harun Turgut_txt"

DEZD_TSV_PATH = TSV_OUTPUT_DIR / "DEZD - Kirmancki Dictionary.tsv"
KAIKKI_HANDOFF_TSV_PATH = TSV_OUTPUT_DIR / "open_source_review" / "kaikki_merge_handoff.tsv"

HARUN_PRIORITY_FANOUT_PAGES = ["282", "154", "70", "108", "263", "353", "173", "90", "147", "382"]
HARUN_PRIORITY_FLAG_PAGES = ["191", "190", "206", "205", "204", "145", "58", "129", "137", "143"]

HARUN_SOURCE_METADATA = {
    "source_dictionary": "Palu - Bingöl Harun Turgut",
    "source_author": "Harun Turgut",
    "source_title": "Zazaca-Türkçe Sözlük",
    "source_publisher": "Vejiyaişê Tiji / Tij Yayınları",
    "source_edition": "2",
    "source_year": "2006",
    "dialect_label_source": "Palu, Bingöl ve çevresinde konuşulan Zazaca",
    "dialect_label_normalized": "",
    "permission_status": "granted",
}

DEZD_SOURCE_METADATA = {
    "source_dictionary": "DEZD - Kirmancki Dictionary",
    "source_author": "",
    "source_title": "DEZD - Kirmancki Dictionary",
    "source_publisher": "",
    "source_edition": "",
    "source_year": "",
    "dialect_label_source": "Kırmancki",
    "dialect_label_normalized": "",
    "permission_status": "public_dataset",
}

KAIKKI_SOURCE_METADATA = {
    "source_dictionary": "Kaikki / Wiktionary Zazaki",
    "source_author": "Wiktionary contributors",
    "source_title": "Kaikki / Wiktionary Zazaki",
    "source_publisher": "Wiktionary / Kaikki.org",
    "source_edition": "",
    "source_year": "",
    "dialect_label_source": "",
    "dialect_label_normalized": "",
    "permission_status": "cc_by_sa",
}

PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")
MULTISPACE_RE = re.compile(r"\s+")
TRAILING_PAGE_NUM_RE = re.compile(r"\b(\d{1,3})$")


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def normalize_spaces(value: str) -> str:
    return MULTISPACE_RE.sub(" ", (value or "").strip())


def clean_cell(value: object) -> str:
    return normalize_spaces("" if value is None else str(value))


def casefold_strip(value: str) -> str:
    return clean_cell(value).casefold()


def normalize_search_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", clean_cell(value).casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("ı", "i")
    text = re.sub(r"['’`\"“”]", "", text)
    text = re.sub(r"[^0-9a-zçğıöşüêîûâ\- ]+", " ", text)
    text = text.replace("ç", "c").replace("ğ", "g").replace("ö", "o")
    text = text.replace("ş", "s").replace("ü", "u")
    text = text.replace("ê", "e").replace("î", "i").replace("û", "u").replace("â", "a")
    text = text.replace("-", " ")
    return normalize_spaces(text)


def tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {name: clean_cell(row.get(name, "")) for name in fieldnames}
            writer.writerow(normalized)


def write_json(path: Path, payload: object) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_affix_entry(value: str) -> bool:
    text = clean_cell(value)
    return bool(text) and (text.startswith("-") or text.endswith("-"))


def priority_bucket(page_number: str) -> str:
    page = clean_cell(page_number)
    if page in HARUN_PRIORITY_FANOUT_PAGES:
        return "fanout"
    if page in HARUN_PRIORITY_FLAG_PAGES:
        return "flags"
    return "alphabetical"


def priority_rank(page_number: str) -> int:
    page = clean_cell(page_number)
    if page in HARUN_PRIORITY_FANOUT_PAGES:
        return HARUN_PRIORITY_FANOUT_PAGES.index(page) + 1
    if page in HARUN_PRIORITY_FLAG_PAGES:
        return 100 + HARUN_PRIORITY_FLAG_PAGES.index(page) + 1
    try:
        return 1000 + int(page)
    except ValueError:
        return 9999


def detect_page_bleed(definition: str, page_number: str) -> bool:
    text = clean_cell(definition)
    if not text:
        return False
    page = clean_cell(page_number)
    if not page.isdigit():
        return False
    near_pages = {str(int(page) + delta) for delta in (-2, -1, 0, 1, 2)}
    match = TRAILING_PAGE_NUM_RE.search(text)
    if match and match.group(1) in near_pages:
        return True
    for nearby in near_pages:
        if f" {nearby} " in f" {text} ":
            return True
    return False


def guess_mt_flag(source_value: str) -> str:
    source = casefold_strip(source_value)
    if not source:
        return ""
    return "1" if "machine" in source or "mt" in source else "0"


def compact_text_list(values: list[str], limit: int = 10, joiner: str = " || ") -> str:
    cleaned = [clean_cell(value) for value in values if clean_cell(value)]
    if len(cleaned) <= limit:
        return joiner.join(cleaned)
    return joiner.join(cleaned[:limit]) + f"{joiner}(+{len(cleaned) - limit} more)"


def review_status_seed(
    *,
    entry: str,
    definition: str,
    raw_needs_review: str,
    non_affix_children: int,
    page_bleed: bool,
) -> str:
    if not clean_cell(entry) or not clean_cell(definition):
        return "hold"
    flags = set(clean_cell(raw_needs_review).split())
    if non_affix_children > 0:
        return "needs_split"
    if "out_of_order" in flags:
        return "needs_merge"
    if flags or page_bleed:
        return "ocr_fix_needed"
    return "seeded"


def source_confidence_score(
    *,
    witness_count: int,
    raw_needs_review: str,
    non_affix_children: int,
    page_bleed: bool,
    best_engine: str,
) -> int:
    score = 35 + (witness_count * 14)
    if best_engine in {"pymupdf", "pypdf"}:
        score += 6
    flags = set(clean_cell(raw_needs_review).split())
    if "out_of_order" in flags:
        score -= 18
    if flags - {"out_of_order"}:
        score -= 12
    if non_affix_children:
        score -= min(20, 6 + (non_affix_children // 75))
    if page_bleed:
        score -= 12
    return max(5, min(99, score))
