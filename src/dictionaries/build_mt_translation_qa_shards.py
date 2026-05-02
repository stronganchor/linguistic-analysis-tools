#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import DEZD_TSV_PATH, TSV_OUTPUT_DIR, clean_cell, ensure_parent_dir, normalize_search_text, write_json, write_tsv


HARUN_TM_TSV_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
OUTPUT_DIR_DEFAULT = Path(__file__).resolve().parent / "translation_qa_20260416"
MANIFEST_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "mt_shard_manifest.tsv"
AUDIT_TSV_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "mt_translation_audit.tsv"
SUMMARY_JSON_PATH_DEFAULT = OUTPUT_DIR_DEFAULT / "mt_translation_audit_summary.json"

TURKISH_FUNCTION_WORDS = {
    "ve",
    "veya",
    "ile",
    "icin",
    "için",
    "olan",
    "olmak",
    "etmek",
    "gibi",
    "çok",
    "cok",
    "bir",
    "bu",
    "şey",
    "sey",
    "kendi",
    "üstüne",
    "ustune",
    "altına",
    "altina",
    "önce",
    "sonra",
    "daha",
    "kadar",
    "değil",
    "degil",
    "arası",
    "arasi",
    "yapmak",
    "yapın",
    "yapin",
}

ENGLISH_PRONOUNS = {"i", "you", "we", "they", "he", "she", "me", "him", "her", "them", "my", "your", "our"}
GERMAN_PRONOUNS = {"ich", "du", "er", "sie", "wir", "ihr", "ihnen", "mich", "dich", "uns", "euch", "mein", "dein"}
PLACEHOLDERISH_TOKENS = {"inftira", "sweepen", "node"}
TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿÇĞİÖŞÜçğıöşüêîûâ]+")

AUDIT_FIELDS = [
    "audit_id",
    "source_dataset",
    "source_key",
    "source_row_idx",
    "entry",
    "entry_ids_preview",
    "definition_tr",
    "definition_en",
    "definition_de",
    "definition_en_source",
    "definition_de_source",
    "definition_en_review_status",
    "definition_de_review_status",
    "risk_score",
    "flags",
    "review_action",
    "reviewed_en",
    "reviewed_de",
    "review_notes",
]

MANIFEST_FIELDS = [
    "shard_id",
    "path",
    "row_count",
    "risk_min",
    "risk_max",
    "source_breakdown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build machine-translation QA audit shards for Harun and DEZD.")
    parser.add_argument("--harun-tm-path", default=str(HARUN_TM_TSV_PATH))
    parser.add_argument("--dezd-path", default=str(DEZD_TSV_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--audit-path", default=str(AUDIT_TSV_PATH_DEFAULT))
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH_DEFAULT))
    parser.add_argument("--summary-path", default=str(SUMMARY_JSON_PATH_DEFAULT))
    parser.add_argument("--shard-count", type=int, default=12)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_cell(text).lower())


def normalized_tokens(text: str) -> list[str]:
    return [token for token in normalize_search_text(text).split() if token]


def duplicate_token_flag(text: str) -> bool:
    tokens = tokenize(text)
    if len(tokens) < 2:
        return False
    for left, right in zip(tokens, tokens[1:], strict=False):
        if left == right:
            return True
    return len(tokens) != len(set(tokens)) and len(tokens) <= 5


def suspicious_tr_overlap(target_text: str) -> bool:
    tokens = set(normalized_tokens(target_text))
    return bool(tokens & TURKISH_FUNCTION_WORDS)


def enumeration_mismatch(tr: str, target: str) -> bool:
    tr_count = tr.count("2)") + tr.count("3)")
    target_count = target.count("2)") + target.count("3)")
    return tr_count > target_count


def suspicious_length(tr: str, target: str) -> bool:
    tr_len = len(tokenize(tr))
    target_len = len(tokenize(target))
    if tr_len >= 6 and target_len <= 2:
        return True
    if tr_len >= 10 and target_len <= 4:
        return True
    return False


def phrase_mode_mismatch(tr: str, target: str, pronouns: set[str]) -> bool:
    tr_tokens = normalized_tokens(tr)
    target_tokens = tokenize(target)
    if not target_tokens:
        return False
    if any(token in pronouns for token in target_tokens[:2]):
        return True
    if tr_tokens and tr_tokens[0] in {"ben", "sen", "siz", "o", "biz", "onlar"}:
        return False
    if clean_cell(target).endswith("."):
        return True
    return False


def placeholder_flag(text: str) -> bool:
    tokens = set(tokenize(text))
    return bool(tokens & PLACEHOLDERISH_TOKENS)


def unusual_case_flag(text: str) -> bool:
    cleaned = clean_cell(text)
    return bool(cleaned) and cleaned[0].isupper() and cleaned.count(" ") == 0


def build_flags(tr: str, en: str, de: str) -> tuple[int, list[str]]:
    flags: list[str] = []
    score = 0
    if duplicate_token_flag(en):
        flags.append("en_duplicate_tokens")
        score += 4
    if duplicate_token_flag(de):
        flags.append("de_duplicate_tokens")
        score += 4
    if suspicious_tr_overlap(en):
        flags.append("en_turkish_lexeme_leak")
        score += 3
    if suspicious_tr_overlap(de):
        flags.append("de_turkish_lexeme_leak")
        score += 3
    if enumeration_mismatch(tr, en):
        flags.append("en_enumeration_mismatch")
        score += 2
    if enumeration_mismatch(tr, de):
        flags.append("de_enumeration_mismatch")
        score += 2
    if suspicious_length(tr, en):
        flags.append("en_too_short_for_source")
        score += 3
    if suspicious_length(tr, de):
        flags.append("de_too_short_for_source")
        score += 3
    if phrase_mode_mismatch(tr, en, ENGLISH_PRONOUNS):
        flags.append("en_sentence_mode")
        score += 2
    if phrase_mode_mismatch(tr, de, GERMAN_PRONOUNS):
        flags.append("de_sentence_mode")
        score += 2
    if placeholder_flag(en):
        flags.append("en_placeholder_or_bad_lexeme")
        score += 5
    if placeholder_flag(de):
        flags.append("de_placeholder_or_bad_lexeme")
        score += 5
    if unusual_case_flag(en):
        flags.append("en_single_capitalized_token")
        score += 1
    if unusual_case_flag(de):
        flags.append("de_single_capitalized_token")
        score += 1
    if len(tokenize(tr)) >= 12 and ("..." in en or "..." in de):
        flags.append("ellipsis_truncation")
        score += 4
    if en and de and normalize_search_text(en) == normalize_search_text(de):
        flags.append("en_de_identical")
        score += 2
    return score, flags


def harun_audit_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        if "machine" not in clean_cell(row.get("translation_source_en")).lower() and "machine" not in clean_cell(row.get("translation_source_de")).lower():
            continue
        definition_tr = clean_cell(row.get("definition_tr"))
        definition_en = clean_cell(row.get("definition_en"))
        definition_de = clean_cell(row.get("definition_de"))
        risk_score, flags = build_flags(definition_tr, definition_en, definition_de)
        audit_rows.append(
            {
                "audit_id": f"harun:{clean_cell(row.get('tm_key'))}",
                "source_dataset": "harun",
                "source_key": clean_cell(row.get("tm_key")),
                "source_row_idx": "",
                "entry": "",
                "entry_ids_preview": clean_cell(row.get("entry_ids_preview")),
                "definition_tr": definition_tr,
                "definition_en": definition_en,
                "definition_de": definition_de,
                "definition_en_source": clean_cell(row.get("translation_source_en")),
                "definition_de_source": clean_cell(row.get("translation_source_de")),
                "definition_en_review_status": clean_cell(row.get("mt_review_status")),
                "definition_de_review_status": clean_cell(row.get("mt_review_status")),
                "risk_score": str(risk_score),
                "flags": " | ".join(flags),
                "review_action": "",
                "reviewed_en": "",
                "reviewed_de": "",
                "review_notes": "",
            }
        )
    return audit_rows


def dezd_audit_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    audit_rows: list[dict[str, str]] = []
    for row in rows:
        if "machine" not in clean_cell(row.get("definition_full_en_source")).lower() and "machine" not in clean_cell(row.get("definition_full_de_source")).lower():
            continue
        definition_tr = clean_cell(row.get("definition_full_tr")) or clean_cell(row.get("definition"))
        definition_en = clean_cell(row.get("definition_full_en"))
        definition_de = clean_cell(row.get("definition_full_de"))
        risk_score, flags = build_flags(definition_tr, definition_en, definition_de)
        audit_rows.append(
            {
                "audit_id": f"dezd:{clean_cell(row.get('source_row_idx'))}:{clean_cell(row.get('entry'))}",
                "source_dataset": "dezd",
                "source_key": f"{clean_cell(row.get('source_row_idx'))}|{clean_cell(row.get('entry'))}",
                "source_row_idx": clean_cell(row.get("source_row_idx")),
                "entry": clean_cell(row.get("entry")),
                "entry_ids_preview": "",
                "definition_tr": definition_tr,
                "definition_en": definition_en,
                "definition_de": definition_de,
                "definition_en_source": clean_cell(row.get("definition_full_en_source")),
                "definition_de_source": clean_cell(row.get("definition_full_de_source")),
                "definition_en_review_status": clean_cell(row.get("definition_full_en_review_status")),
                "definition_de_review_status": clean_cell(row.get("definition_full_de_review_status")),
                "risk_score": str(risk_score),
                "flags": " | ".join(flags),
                "review_action": "",
                "reviewed_en": "",
                "reviewed_de": "",
                "review_notes": "",
            }
        )
    return audit_rows


def shard_rows(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    shard_count = max(1, shard_count)
    size = math.ceil(len(rows) / shard_count)
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audit_path = Path(args.audit_path)
    manifest_path = Path(args.manifest_path)
    summary_path = Path(args.summary_path)

    harun_rows = read_tsv(Path(args.harun_tm_path))
    dezd_rows = read_tsv(Path(args.dezd_path))
    audit_rows = harun_audit_rows(harun_rows) + dezd_audit_rows(dezd_rows)
    audit_rows.sort(
        key=lambda row: (
            -int(clean_cell(row.get("risk_score")) or "0"),
            clean_cell(row.get("source_dataset")),
            clean_cell(row.get("source_key")),
        )
    )

    write_tsv(audit_path, AUDIT_FIELDS, audit_rows)

    manifest_rows: list[dict[str, str]] = []
    for index, rows in enumerate(shard_rows(audit_rows, args.shard_count), start=1):
        shard_id = f"mt_{index:02d}"
        shard_path = output_dir / "shards" / f"{shard_id}.tsv"
        write_tsv(shard_path, AUDIT_FIELDS, rows)
        source_breakdown = Counter(clean_cell(row.get("source_dataset")) for row in rows)
        risk_values = [int(clean_cell(row.get("risk_score")) or "0") for row in rows]
        manifest_rows.append(
            {
                "shard_id": shard_id,
                "path": str(shard_path),
                "row_count": str(len(rows)),
                "risk_min": str(min(risk_values) if risk_values else 0),
                "risk_max": str(max(risk_values) if risk_values else 0),
                "source_breakdown": json.dumps(source_breakdown, ensure_ascii=False, sort_keys=True),
            }
        )
    write_tsv(manifest_path, MANIFEST_FIELDS, manifest_rows)

    summary = {
        "audit_row_count": len(audit_rows),
        "harun_row_count": sum(1 for row in audit_rows if row["source_dataset"] == "harun"),
        "dezd_row_count": sum(1 for row in audit_rows if row["source_dataset"] == "dezd"),
        "flag_counts": Counter(flag for row in audit_rows for flag in row["flags"].split(" | ") if flag),
        "risk_distribution": Counter(int(clean_cell(row.get("risk_score")) or "0") for row in audit_rows),
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
