#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from urllib import parse, request

from _dictionary_pipeline_common import SCRIPT_DIR, clean_cell, ensure_parent_dir


QUEUE_PATH_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review" / "open_source_import_review_queue.tsv"
FLAGS_PATH_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review" / "kaikki_qa_flags.tsv"

EXCLUDE_ISSUE_TYPES = {
    "parser_artifact",
    "structural_nonlemma",
    "redirect_like_entry",
    "mechanical_compound",
    "proper_noun_policy",
    "sensitive_lexis",
}

MANUAL_EXCLUDE_IDS = {
    "kaikki:0384:01",  # proverb gloss looks corrupted
    "kaikki:0834:01",  # past-tense form row with broken gloss; leave out
    "kaikki:1005:02",  # "to apart" is too broken to trust without guessing
}

FIXED_ENGLISH_OVERRIDES = {
    "kaikki:0067:01": "father (short form)",
    "kaikki:0100:01": "lake",
    "kaikki:0688:02": "persona; the mask or appearance one presents to the world",
    "kaikki:0964:01": "to make someone smile",
}

MANUAL_TR_OVERRIDES = {
    "kaikki:0308:01": "durma noktasına gelmek",
    "kaikki:0410:01": "lekelemek",
    "kaikki:0595:01": "burmak",
    "kaikki:0595:02": "sıkmak",
    "kaikki:0764:01": "taşımak",
    "kaikki:0838:01": "köpürmüş",
    "kaikki:1004:01": "tutuşturmak",
    "kaikki:1021:03": "dolaşma",
    "kaikki:1054:02": "ayakta durmak",
}

MANUAL_DE_OVERRIDES = {
    "kaikki:1021:03": "Umherstreifen",
    "kaikki:1028:02": "Gnosis",
}

TR_TEXT_NORMALIZATION = {
    "kÖY": "köy",
    "tANRI": "tanrı",
    "şEHİR": "şehir",
    "pATLICAN": "patlıcan",
    "şEFTALİ": "şeftali",
    "dİZ": "diz",
    "dUA": "dua",
    "pORSUK": "porsuk",
    "aYÇİÇEĞİ": "ayçiçeği",
    "dİL": "dil",
    "çOBAN": "çoban",
    "sİLAH": "silah",
    "kITLIK": "kıtlık",
    "kEKLİK": "keklik",
    "kISKANÇ": "kıskanç",
    "eVLİLİK DURUMU": "evlilik durumu",
    "sÜRGÜN": "sürgün",
    "gELİN": "gelin",
    "dIŞ KAPI": "dış kapı",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Kaikki curation decisions and fill Turkish/German manual translations in the review queue.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--flags-path", default=str(FLAGS_PATH_DEFAULT))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--backup-suffix", default=".pre_kaikki_curation_20260417.bak")
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


def windows_env_value(name: str) -> str:
    result = subprocess.run(
        ["cmd.exe", "/c", f"echo %{name}%"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    value = clean_cell(result.stdout)
    if not value or value == f"%{name}%":
        return ""
    return value


def resolve_api_key(explicit: str) -> str:
    if clean_cell(explicit):
        return clean_cell(explicit)
    for name in ("DEEPL_API_KEY", "DEEPL_AUTH_KEY"):
        value = clean_cell(os.environ.get(name))
        if value:
            return value
    for name in ("DEEPL_API_KEY", "DEEPL_AUTH_KEY"):
        value = windows_env_value(name)
        if value:
            return value
    raise RuntimeError("No DeepL API key found in environment")


def resolve_endpoint(explicit: str, api_key: str) -> str:
    if clean_cell(explicit):
        return clean_cell(explicit)
    if api_key.endswith(":fx"):
        return "https://api-free.deepl.com/v2/translate"
    return "https://api.deepl.com/v2/translate"


def deepl_translate(*, api_key: str, endpoint: str, texts: list[str], target_lang: str) -> list[str]:
    if not texts:
        return []
    data = parse.urlencode(
        [
            ("source_lang", "EN"),
            ("target_lang", target_lang),
            ("preserve_formatting", "1"),
            ("split_sentences", "nonewlines"),
            *[("text", text) for text in texts],
        ],
        doseq=True,
    ).encode("utf-8")
    req = request.Request(endpoint, data=data, method="POST")
    req.add_header("Authorization", f"DeepL-Auth-Key {api_key}")
    with request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    translations = payload.get("translations") or []
    if len(translations) != len(texts):
        raise RuntimeError(f"DeepL returned {len(translations)} translations for {len(texts)} texts")
    return [clean_cell(item.get("text")) for item in translations]


def balanced_parens(text: str) -> bool:
    return text.count("(") == text.count(")")


def choose_english_gloss(row: dict[str, str]) -> str:
    review_id = clean_cell(row.get("review_id"))
    if review_id in FIXED_ENGLISH_OVERRIDES:
        return FIXED_ENGLISH_OVERRIDES[review_id]
    raw_gloss = clean_cell(row.get("source_gloss_raw"))
    gloss_en = clean_cell(row.get("source_gloss_en"))
    if raw_gloss and balanced_parens(raw_gloss):
        return raw_gloss
    return gloss_en


def translation_input_for_row(row: dict[str, str]) -> tuple[str, str]:
    english_gloss = clean_cell(row.get("curated_definition_en"))
    entry_type = clean_cell(row.get("entry_type"))
    if entry_type == "verb" and english_gloss:
        base = english_gloss[3:] if english_gloss.lower().startswith("to ") else english_gloss
        return f"{base} (verb)", "verb"
    return english_gloss, "default"


def cleanup_translated_value(text: str, mode: str, target_lang: str) -> str:
    value = clean_cell(text)
    if mode != "verb":
        if target_lang == "TR":
            return TR_TEXT_NORMALIZATION.get(value, value)
        return value
    if target_lang == "TR":
        value = re.sub(r"\s*\(fiil\)\s*$", "", value, flags=re.I)
        value = TR_TEXT_NORMALIZATION.get(value, value)
    if target_lang == "DE":
        value = re.sub(r"\s*\(Verb\)\s*$", "", value, flags=re.I)
    return clean_cell(value)


def append_note(existing: str, note: str) -> str:
    current = clean_cell(existing)
    if not current:
        return note
    parts = current.split(" | ")
    if note in parts:
        return current
    return f"{current} | {note}"


def chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def load_flag_issues(path: Path) -> dict[str, set[str]]:
    _, rows = read_tsv(path)
    issues: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        review_id = clean_cell(row.get("review_id"))
        issue_type = clean_cell(row.get("issue_type"))
        if review_id and issue_type:
            issues[review_id].add(issue_type)
    return issues


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue_path)
    flags_path = Path(args.flags_path)

    api_key = resolve_api_key(args.api_key)
    endpoint = resolve_endpoint(args.endpoint, api_key)

    fieldnames, rows = read_tsv(queue_path)
    issues_by_review_id = load_flag_issues(flags_path)

    backup_path = backup_once(queue_path, args.backup_suffix)

    stats: Counter = Counter()
    kept_inputs: dict[str, tuple[str, str]] = {}

    for row in rows:
        if clean_cell(row.get("source_family")) != "kaikki":
            continue

        review_id = clean_cell(row.get("review_id"))
        issues = issues_by_review_id.get(review_id, set())
        should_exclude = bool(issues & EXCLUDE_ISSUE_TYPES) or review_id in MANUAL_EXCLUDE_IDS

        if should_exclude:
            row["curation_status"] = "reviewed"
            row["merge_decision"] = "exclude"
            row["curated_definition_en"] = ""
            row["curated_definition_tr"] = ""
            row["curated_definition_de"] = ""
            notes: list[str] = []
            if issues:
                notes.append("qa_flags=" + ",".join(sorted(issues)))
            if review_id in MANUAL_EXCLUDE_IDS:
                notes.append("manual_exclude=uncertain_or_corrupted")
            row["curated_notes"] = " ; ".join(notes) if notes else "excluded by QA pass"
            stats["excluded_rows"] += 1
            continue

        english_gloss = choose_english_gloss(row)
        if not english_gloss:
            row["curation_status"] = "reviewed"
            row["merge_decision"] = "exclude"
            row["curated_definition_en"] = ""
            row["curated_definition_tr"] = ""
            row["curated_definition_de"] = ""
            row["curated_notes"] = "manual_exclude=missing_usable_english_gloss"
            stats["excluded_rows"] += 1
            stats["excluded_missing_gloss"] += 1
            continue

        row["curation_status"] = "translated"
        row["merge_decision"] = "merge"
        row["curated_entry"] = clean_cell(row.get("curated_entry")) or clean_cell(row.get("headword"))
        row["curated_entry_type"] = clean_cell(row.get("curated_entry_type")) or clean_cell(row.get("entry_type"))
        row["curated_gender_number"] = clean_cell(row.get("curated_gender_number")) or clean_cell(row.get("gender_number"))
        row["curated_definition_en"] = english_gloss

        if review_id in FIXED_ENGLISH_OVERRIDES:
            row["curation_status"] = "fixed"
            row["curated_notes"] = append_note(row.get("curated_notes", ""), "english_gloss_normalized")
            stats["fixed_gloss_rows"] += 1

        kept_inputs[review_id] = translation_input_for_row(row)
        stats["kept_rows"] += 1

    unique_inputs = sorted(set(kept_inputs.values()))
    tr_cache: dict[tuple[str, str], str] = {}
    de_cache: dict[tuple[str, str], str] = {}

    for batch in chunks(unique_inputs, args.batch_size):
        batch_texts = [text for text, _mode in batch]
        batch_tr = deepl_translate(api_key=api_key, endpoint=endpoint, texts=batch_texts, target_lang="TR")
        batch_de = deepl_translate(api_key=api_key, endpoint=endpoint, texts=batch_texts, target_lang="DE")
        for (text, mode), tr_value, de_value in zip(batch, batch_tr, batch_de, strict=True):
            tr_cache[(text, mode)] = cleanup_translated_value(tr_value, mode, "TR")
            de_cache[(text, mode)] = cleanup_translated_value(de_value, mode, "DE")

    for row in rows:
        if clean_cell(row.get("source_family")) != "kaikki":
            continue
        if clean_cell(row.get("merge_decision")) != "merge":
            continue
        english_gloss = clean_cell(row.get("curated_definition_en"))
        if not english_gloss:
            continue
        translation_input, mode = kept_inputs[clean_cell(row.get("review_id"))]
        row["curated_definition_tr"] = tr_cache[(translation_input, mode)]
        row["curated_definition_de"] = de_cache[(translation_input, mode)]
        review_id = clean_cell(row.get("review_id"))
        if review_id in MANUAL_TR_OVERRIDES:
            row["curated_definition_tr"] = MANUAL_TR_OVERRIDES[review_id]
            row["curated_notes"] = append_note(row.get("curated_notes", ""), "curated_definition_tr_manual_override")
        if review_id in MANUAL_DE_OVERRIDES:
            row["curated_definition_de"] = MANUAL_DE_OVERRIDES[review_id]
            row["curated_notes"] = append_note(row.get("curated_notes", ""), "curated_definition_de_manual_override")
        row["curated_notes"] = append_note(row.get("curated_notes", ""), "curated_definition_tr_deepl_en_tr")
        row["curated_notes"] = append_note(row.get("curated_notes", ""), "curated_definition_de_deepl_en_de")
        stats["translated_rows"] += 1

    write_tsv(queue_path, fieldnames, rows)

    summary = {
        "queue_path": str(queue_path),
        "backup_path": str(backup_path),
        "api_endpoint": endpoint,
        "kept_rows": stats["kept_rows"],
        "excluded_rows": stats["excluded_rows"],
        "translated_rows": stats["translated_rows"],
        "fixed_gloss_rows": stats["fixed_gloss_rows"],
        "unique_translation_inputs": len(unique_inputs),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
