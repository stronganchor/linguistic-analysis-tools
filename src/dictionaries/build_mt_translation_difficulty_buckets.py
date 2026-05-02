#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv
from build_mt_translation_qa_shards import AUDIT_FIELDS


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "translation_qa_20260416" / "difficulty_buckets"
AUDIT_PATH_DEFAULT = SCRIPT_DIR / "translation_qa_20260416" / "mt_translation_audit.tsv"
SUMMARY_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "difficulty_summary.json"

DIFFICULTY_FIELDS = AUDIT_FIELDS + [
    "difficulty_bucket",
    "difficulty_score",
    "deepl_risk_score",
    "local_llm_risk_score",
    "hard_for_deepl",
    "hard_for_local_llm",
    "difficulty_reasons",
]

BUCKET_OUTPUT_ORDER = (
    "all_unresolved",
    "hard",
    "medium",
    "easy",
    "hard_for_deepl",
    "hard_for_local_llm",
    "hard_for_both",
    "hard_for_deepl_only",
    "hard_for_local_llm_only",
)

INTERJECTION_PATTERNS = (
    "seslen",
    "seslenme",
    "seslenirken",
    "hitap",
    "ünlem",
    "unlem",
    "sevinç",
    "sevinc",
    "şaşkın",
    "saskin",
    "hey be",
    "vay be",
    "oy be",
    "vah",
    "aman",
    "hay be",
)
GRAMMAR_PATTERNS = (
    "zamir",
    "tekil şahıs",
    "tekil sahıs",
    "çoğul şahıs",
    "cogul sahıs",
    "bükümlü",
    "bukümlü",
    "bukumlu",
    "iyelik",
    "çekimli",
    "cekimli",
    "oblique",
    "vocative",
)
ONOMATOPOEIA_PATTERNS = (
    "gümbür",
    "hüngür",
    "püf",
    "puf",
    "şapır",
    "şap",
    "şak",
    "kıtır",
    "kütür",
    "gurr",
    "çıt",
    "çat diye",
    "pat diye",
    "vır vır",
    "fırıl",
    "fildır",
)
CULTURE_PATTERNS = (
    "kimlik tanımlarından",
    "kimlik tanimlarindan",
    "aynı dili konuşan",
    "ayni dili konusan",
    "yerleşim yerleri",
    "yerlesim yerleri",
    "diaspora",
    "sunni hanefi",
    "sünni hanefi",
    "sembolik",
    "ritüel",
    "rituel",
    "meyve ağacının altına",
    "meyve agacinin altina",
)
IDIOM_BODY_PART_WORDS = {
    "ağız",
    "agiz",
    "ayak",
    "ayağı",
    "ayagi",
    "baş",
    "bas",
    "başına",
    "basina",
    "ciğer",
    "ciger",
    "dil",
    "el",
    "eli",
    "göz",
    "goz",
    "kalp",
    "kol",
    "kıç",
    "kic",
    "omuz",
    "yürek",
    "yurek",
}
REDUPLICATION_RE = re.compile(r"\b([a-zçğıöşüêîûâ]{2,})(?:[- ]\1)\b", flags=re.IGNORECASE)
ENTRY_DIACRITIC_RE = re.compile(r"[çğıöşüêîûâÇĞİÖŞÜÊÎÛÂ]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket MT audit rows by likely model difficulty.")
    parser.add_argument("--audit-path", default=str(AUDIT_PATH_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH_DEFAULT))
    parser.add_argument("--include-resolved", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def has_any(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = clean_cell(text).lower()
    return any(pattern in lowered for pattern in patterns)


def normalized_flags(row: dict[str, str]) -> set[str]:
    return {clean_cell(part) for part in clean_cell(row.get("flags")).split("|") if clean_cell(part)}


def is_resolved(row: dict[str, str]) -> bool:
    blob = " ".join(
        clean_cell(row.get(key))
        for key in (
            "definition_en_source",
            "definition_de_source",
            "definition_en_review_status",
            "definition_de_review_status",
        )
    ).lower()
    return any(
        marker in blob
        for marker in (
            "trusted_pre_5_3_import_20260417",
            "manual_qa_corrected_20260416",
            "manual_qa_corrected_20260417",
            "manual_qa_checked_20260416",
            "ollama_qa_corrected_20260416",
            "ollama_qa_checked_20260416",
        )
    )


def classify_row(row: dict[str, str], variant_family_count: int) -> dict[str, str]:
    tr = clean_cell(row.get("definition_tr"))
    entry = clean_cell(row.get("entry"))
    audit_id = clean_cell(row.get("audit_id"))
    flags = normalized_flags(row)
    risk_score = int(clean_cell(row.get("risk_score")) or "0")

    deepl_score = 0
    local_score = 0
    reasons: list[str] = []

    def add(reason: str, deepl_delta: int, local_delta: int) -> None:
        nonlocal deepl_score, local_score
        reasons.append(reason)
        deepl_score += deepl_delta
        local_score += local_delta

    if has_any(tr, INTERJECTION_PATTERNS):
        add("interjection_or_discourse_particle", 6, 4)
    if has_any(tr, GRAMMAR_PATTERNS):
        add("grammar_or_pronoun_note", 6, 4)
    if has_any(tr, ONOMATOPOEIA_PATTERNS) or REDUPLICATION_RE.search(entry) or REDUPLICATION_RE.search(tr):
        add("onomatopoeia_or_sound_symbolism", 5, 4)
    if has_any(tr, CULTURE_PATTERNS) or "<br>" in tr or ("yerleşim yerleri" in tr.lower() or "yerlesim yerleri" in tr.lower()):
        add("culture_ritual_or_ethnonym_entry", 4, 6)
    if len(tr) >= 120 or tr.count(";") >= 2 or tr.count("2)") + tr.count("3)") + tr.count("4)") >= 3:
        add("long_or_heavily_structured_gloss", 2, 4)
    tr_tokens = {token for token in re.findall(r"[0-9A-Za-zÀ-ÿÇĞİÖŞÜçğıöşüêîûâ]+", tr.lower())}
    if tr_tokens & IDIOM_BODY_PART_WORDS and len(tr_tokens) >= 4:
        add("idiom_or_body_part_expression", 4, 3)
    if variant_family_count > 1:
        add("variant_family_row", 1, 3)
    if ENTRY_DIACRITIC_RE.search(entry) and clean_cell(row.get("source_dataset")) == "dezd":
        add("dezd_diacritic_id_risk", 0, 2)

    if "ellipsis_truncation" in flags:
        add("existing_mt_truncation", 1, 2)
    if "en_placeholder_or_bad_lexeme" in flags or "de_placeholder_or_bad_lexeme" in flags:
        add("existing_mt_placeholder_failure", 1, 2)
    if "en_too_short_for_source" in flags or "de_too_short_for_source" in flags:
        add("existing_mt_severe_shortening", 1, 1)
    if "en_enumeration_mismatch" in flags or "de_enumeration_mismatch" in flags:
        add("existing_mt_numbering_failure", 1, 1)
    if "en_de_identical" in flags:
        add("existing_mt_cross_language_collapse", 1, 1)

    if risk_score >= 16:
        add("very_high_current_risk_score", 1, 1)
    elif risk_score >= 12:
        add("high_current_risk_score", 1, 0)

    hard_for_deepl = deepl_score >= 6
    hard_for_local = local_score >= 6
    difficulty_score = deepl_score + local_score + min(risk_score, 6)
    if hard_for_deepl or hard_for_local:
        bucket = "hard"
    elif deepl_score >= 3 or local_score >= 3 or risk_score >= 10:
        bucket = "medium"
    else:
        bucket = "easy"

    return {
        **row,
        "difficulty_bucket": bucket,
        "difficulty_score": str(difficulty_score),
        "deepl_risk_score": str(deepl_score),
        "local_llm_risk_score": str(local_score),
        "hard_for_deepl": "yes" if hard_for_deepl else "no",
        "hard_for_local_llm": "yes" if hard_for_local else "no",
        "difficulty_reasons": " | ".join(dict.fromkeys(reasons)),
    }


def write_bucket(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    write_tsv(path, DIFFICULTY_FIELDS, rows)


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_path)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary_path)

    rows = read_tsv(audit_path)
    variant_counts = Counter(
        clean_cell(row.get("source_row_idx"))
        for row in rows
        if clean_cell(row.get("source_dataset")) == "dezd" and clean_cell(row.get("source_row_idx"))
    )

    filtered_rows = rows if args.include_resolved else [row for row in rows if not is_resolved(row)]
    classified = [
        classify_row(row, variant_counts.get(clean_cell(row.get("source_row_idx")), 0))
        for row in filtered_rows
    ]
    classified.sort(
        key=lambda row: (
            {"hard": 0, "medium": 1, "easy": 2}[row["difficulty_bucket"]],
            -int(row["difficulty_score"]),
            -int(clean_cell(row.get("risk_score")) or "0"),
            clean_cell(row.get("source_dataset")),
            clean_cell(row.get("source_key")),
        )
    )

    bucket_rows = {
        "all_unresolved": classified,
        "hard": [row for row in classified if row["difficulty_bucket"] == "hard"],
        "medium": [row for row in classified if row["difficulty_bucket"] == "medium"],
        "easy": [row for row in classified if row["difficulty_bucket"] == "easy"],
        "hard_for_deepl": [row for row in classified if row["hard_for_deepl"] == "yes"],
        "hard_for_local_llm": [row for row in classified if row["hard_for_local_llm"] == "yes"],
        "hard_for_both": [
            row
            for row in classified
            if row["hard_for_deepl"] == "yes" and row["hard_for_local_llm"] == "yes"
        ],
        "hard_for_deepl_only": [
            row
            for row in classified
            if row["hard_for_deepl"] == "yes" and row["hard_for_local_llm"] != "yes"
        ],
        "hard_for_local_llm_only": [
            row
            for row in classified
            if row["hard_for_local_llm"] == "yes" and row["hard_for_deepl"] != "yes"
        ],
    }

    for name in BUCKET_OUTPUT_ORDER:
        rows_for_bucket = bucket_rows[name]
        write_bucket(output_dir / f"{name}.tsv", rows_for_bucket)

    summary = {
        "audit_path": str(audit_path),
        "include_resolved": args.include_resolved,
        "row_count": len(classified),
        "bucket_counts": {name: len(bucket_rows[name]) for name in BUCKET_OUTPUT_ORDER},
        "source_counts": Counter(clean_cell(row.get("source_dataset")) for row in classified),
        "difficulty_reason_counts": Counter(
            reason
            for row in classified
            for reason in clean_cell(row.get("difficulty_reasons")).split(" | ")
            if reason
        ),
        "outputs": {name: str(output_dir / f"{name}.tsv") for name in BUCKET_OUTPUT_ORDER},
    }
    ensure_parent_dir(summary_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
