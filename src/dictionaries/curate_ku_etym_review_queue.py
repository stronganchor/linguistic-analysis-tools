#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from urllib import parse, request

from _dictionary_pipeline_common import SCRIPT_DIR, clean_cell, ensure_parent_dir, normalize_search_text


QUEUE_PATH_DEFAULT = SCRIPT_DIR / "tsv_output" / "open_source_review" / "open_source_import_review_queue.tsv"

QUOTE_TEXT_RE = re.compile(r'"([^"]*)"')
BOLD_TEXT_RE = re.compile(r"'''([^']+)'''")
HEADWORD_VARIANT_SPLIT_RE = re.compile(r"\s*,\s*")
TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:]+$")
VERBISH_HEADWORD_RE = re.compile(r"(?:yiş|yış|ayiş|ayış|itiş|itış|diş|dış|tiş|tış|nayiş|nayış)$", re.IGNORECASE)

EXCLUDE_REVIEW_IDS = {
    "ku_etym:D.raw:0026",  # -dar
    "ku_etym:S.raw:0042",  # -stan
}

MANUAL_GLOSS_OVERRIDES = {
    "ku_etym:C-cedilla.raw:0049": "forty",
}

MANUAL_VERB_REVIEW_IDS = {
    "ku_etym:A.raw:0018",
    "ku_etym:A.raw:0026",
    "ku_etym:G.raw:0020",
    "ku_etym:D.raw:0059",
    "ku_etym:G.raw:0041",
    "ku_etym:K.raw:0012",
    "ku_etym:K.raw:0068",
    "ku_etym:Z.raw:0011",
}

MANUAL_TR_OVERRIDES = {
    "ku_etym:A.raw:0026": "saymak",
    "ku_etym:C-cedilla.raw:0056": "öldürmek, vurmak",
    "ku_etym:E.raw:0030": "ben",
    "ku_etym:G.raw:0020": "yaşamak",
    "ku_etym:M.raw:0057": "ben",
}

MANUAL_DE_OVERRIDES = {
    "ku_etym:B.raw:0003": "Band",
    "ku_etym:C-cedilla.raw:0056": "erschlagen, töten",
    "ku_etym:D.raw:0073": "Hand",
    "ku_etym:D.raw:0108": "Smog",
    "ku_etym:E.raw:0030": "ich",
    "ku_etym:G.raw:0020": "leben",
    "ku_etym:M.raw:0034": "Mann",
    "ku_etym:M.raw:0057": "ich",
    "ku_etym:N.raw:0005": "Name",
    "ku_etym:S-cedilla.raw:0005": "Ruhe",
    "ku_etym:V.raw:0051": "Wolf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate ku.wiktionary etymology rows in the open-source review queue.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH_DEFAULT))
    parser.add_argument("--api-key", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--backup-suffix", default=".pre_ku_etym_curation_20260417.bak")
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


def strip_markup(value: str) -> str:
    text = clean_cell(value)
    text = re.sub(r"</?font[^>]*>", "", text)
    text = re.sub(r"</?sup[^>]*>", "", text)
    text = text.replace("'''", "").replace("''", "")
    return clean_cell(text)


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


def split_etym_headwords(value: str) -> list[str]:
    return unique_values([strip_markup(part) for part in HEADWORD_VARIANT_SPLIT_RE.split(strip_markup(value)) if strip_markup(part)])


def parse_etym_entry_line(line: str) -> tuple[list[str], str, str]:
    stripped = line.lstrip("*").strip()
    bold_matches = list(BOLD_TEXT_RE.finditer(stripped))
    headwords: list[str] = []
    for match in bold_matches:
        headwords.extend(split_etym_headwords(match.group(1)))
    headwords = unique_values(headwords)

    remainder = stripped[bold_matches[-1].end():].strip() if bold_matches else stripped
    if remainder.startswith(","):
        remainder = remainder[1:].strip()

    gloss = ""
    etymology = ""
    if remainder.startswith('"'):
        gloss_match = QUOTE_TEXT_RE.search(remainder)
        gloss = strip_markup(gloss_match.group(1) if gloss_match else "")
        etymology = remainder[gloss_match.end():].strip() if gloss_match else remainder
        if etymology.startswith(":"):
            etymology = etymology[1:].strip()
    elif remainder.startswith(":"):
        etymology = remainder[1:].strip()
        gloss_match = QUOTE_TEXT_RE.search(etymology)
        gloss = strip_markup(gloss_match.group(1) if gloss_match else "")
    else:
        etymology = remainder
    return headwords, normalize_gloss(gloss, ""), normalize_etymology(etymology)


def normalize_gloss(gloss: str, review_id: str) -> str:
    value = clean_cell(gloss)
    if review_id in MANUAL_GLOSS_OVERRIDES:
        value = MANUAL_GLOSS_OVERRIDES[review_id]
    value = TRAILING_PUNCT_RE.sub("", value)
    value = value.replace("fourty", "forty")
    return clean_cell(value)


def normalize_etymology(etymology: str) -> str:
    value = strip_markup(etymology)
    value = re.sub(r'^,\s*"[^"]+"\s*:\s*', "", value)
    value = value.lstrip(",;: ")
    if value.startswith("PIE base "):
        value = "from " + value
    return clean_cell(value)


def load_source_line(row: dict[str, str]) -> str:
    source_path = SCRIPT_DIR / clean_cell(row.get("source_path"))
    _file_name, line_number = clean_cell(row.get("source_locator")).split(":")
    return source_path.read_text(encoding="utf-8").splitlines()[int(line_number) - 1]


def split_variants(value: str) -> list[str]:
    parts: list[str] = []
    for pipe_part in clean_cell(value).split("|"):
        parts.extend(split_etym_headwords(pipe_part))
    return unique_values(parts)


def choose_target_title(row: dict[str, str], variants: list[str]) -> str:
    matched_titles = [clean_cell(part) for part in clean_cell(row.get("master_match_titles")).split("|") if clean_cell(part)]
    if matched_titles:
        return matched_titles[0]
    if variants:
        return variants[0]
    return clean_cell(row.get("headword"))


def alias_variants(variants: list[str], target_title: str, matched_titles: list[str]) -> list[str]:
    matched_searches = {normalize_search_text(title) for title in matched_titles}
    target_search = normalize_search_text(target_title)
    aliases: list[str] = []
    for variant in variants:
        variant_search = normalize_search_text(variant)
        if variant_search == target_search:
            continue
        if matched_searches and variant_search in matched_searches:
            continue
        aliases.append(variant)
    return unique_values(aliases)


def translation_input_for_row(review_id: str, headword: str, gloss: str) -> tuple[str, str]:
    english_gloss = clean_cell(gloss)
    if not english_gloss:
        return "", "default"
    is_verb = (
        english_gloss.lower().startswith("to ")
        or review_id in MANUAL_VERB_REVIEW_IDS
        or VERBISH_HEADWORD_RE.search(clean_cell(headword)) is not None
    )
    if is_verb:
        base = english_gloss[3:] if english_gloss.lower().startswith("to ") else english_gloss
        return f"{base} (verb)", "verb"
    return english_gloss, "default"


def cleanup_translated_value(text: str, mode: str, target_lang: str) -> str:
    value = clean_cell(text)
    if mode == "verb":
        if target_lang == "TR":
            value = re.sub(r"\s*\(fiil\)\s*$", "", value, flags=re.I)
        if target_lang == "DE":
            value = re.sub(r"\s*\(Verb\)\s*$", "", value, flags=re.I)
    return clean_cell(value)


def main() -> None:
    args = parse_args()
    queue_path = Path(args.queue_path)
    fieldnames, rows = read_tsv(queue_path)
    backup_path = backup_once(queue_path, args.backup_suffix)

    api_key = resolve_api_key(args.api_key)
    endpoint = resolve_endpoint(args.endpoint, api_key)

    stats: Counter = Counter()
    translation_inputs: dict[str, tuple[str, str]] = {}

    for row in rows:
        if clean_cell(row.get("source_family")) != "ku_wiktionary_etymology":
            continue

        review_id = clean_cell(row.get("review_id"))
        line = load_source_line(row)
        headwords, gloss, etymology = parse_etym_entry_line(line)
        if not gloss:
            gloss = normalize_gloss(clean_cell(row.get("source_gloss_en")), review_id)

        variants = split_variants(" | ".join(headwords) if headwords else clean_cell(row.get("headword_variants")))
        matched_titles = [clean_cell(part) for part in clean_cell(row.get("master_match_titles")).split("|") if clean_cell(part)]
        target_title = choose_target_title(row, variants)
        aliases = alias_variants(variants, target_title, matched_titles)

        row["headword"] = variants[0] if variants else clean_cell(row.get("headword"))
        row["headword_variants"] = " | ".join(variants)
        row["headword_search"] = normalize_search_text(row["headword"])
        row["source_gloss_en"] = gloss
        row["source_etymology"] = etymology
        row["source_gloss_raw"] = ""

        if review_id in EXCLUDE_REVIEW_IDS:
            row["curation_status"] = "reviewed"
            row["merge_decision"] = "exclude"
            row["curated_entry"] = ""
            row["curated_parent"] = ""
            row["curated_entry_type"] = ""
            row["curated_gender_number"] = ""
            row["curated_definition_tr"] = ""
            row["curated_definition_en"] = ""
            row["curated_definition_de"] = ""
            row["curated_etymology_note"] = ""
            row["curated_aliases"] = ""
            row["curated_notes"] = "manual_exclude=affix_like_etymology_row"
            stats["excluded_rows"] += 1
            continue

        merge_mode = "new_entry_with_etymology" if clean_cell(row.get("master_match_status")) == "no_headwords_in_master" else "etymology_enrichment"
        row["curation_status"] = "translated"
        row["merge_decision"] = "merge"
        row["curated_entry"] = target_title
        row["curated_parent"] = ""
        row["curated_entry_type"] = ""
        row["curated_gender_number"] = ""
        row["curated_dialect"] = ""
        row["curated_definition_en"] = gloss
        row["curated_etymology_note"] = etymology
        row["curated_aliases"] = " | ".join(aliases)
        row["curated_notes"] = append_note("", f"merge_mode={merge_mode}")
        row["curated_notes"] = append_note(row["curated_notes"], "source_fields_refreshed_from_raw_line")
        if review_id in MANUAL_GLOSS_OVERRIDES:
            row["curated_notes"] = append_note(row["curated_notes"], "english_gloss_normalized")

        translation_inputs[review_id] = translation_input_for_row(review_id, row["headword"], gloss)
        stats["merged_rows"] += 1
        stats[f"merge_mode:{merge_mode}"] += 1

    unique_inputs = sorted({value for value in translation_inputs.values() if value[0]})
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
        if clean_cell(row.get("source_family")) != "ku_wiktionary_etymology":
            continue
        if clean_cell(row.get("merge_decision")) != "merge":
            continue
        review_id = clean_cell(row.get("review_id"))
        translation_input, mode = translation_inputs[review_id]
        row["curated_definition_tr"] = tr_cache[(translation_input, mode)]
        row["curated_definition_de"] = de_cache[(translation_input, mode)]
        if review_id in MANUAL_TR_OVERRIDES:
            row["curated_definition_tr"] = MANUAL_TR_OVERRIDES[review_id]
            row["curated_notes"] = append_note(row["curated_notes"], "curated_definition_tr_manual_override")
        if review_id in MANUAL_DE_OVERRIDES:
            row["curated_definition_de"] = MANUAL_DE_OVERRIDES[review_id]
            row["curated_notes"] = append_note(row["curated_notes"], "curated_definition_de_manual_override")
        row["curated_notes"] = append_note(row["curated_notes"], "curated_definition_tr_deepl_en_tr")
        row["curated_notes"] = append_note(row["curated_notes"], "curated_definition_de_deepl_en_de")
        stats["translated_rows"] += 1

    write_tsv(queue_path, fieldnames, rows)

    summary = {
        "queue_path": str(queue_path),
        "backup_path": str(backup_path),
        "api_endpoint": endpoint,
        "merged_rows": stats["merged_rows"],
        "excluded_rows": stats["excluded_rows"],
        "translated_rows": stats["translated_rows"],
        "merge_mode_counts": {key.removeprefix("merge_mode:"): value for key, value in stats.items() if key.startswith("merge_mode:")},
        "unique_translation_inputs": len(unique_inputs),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
