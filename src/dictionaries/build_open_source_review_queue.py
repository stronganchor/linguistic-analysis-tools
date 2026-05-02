#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from _dictionary_pipeline_common import (
    SCRIPT_DIR,
    clean_cell,
    compact_text_list,
    normalize_search_text,
    write_json,
    write_tsv,
)


MASTER_SNAPSHOT_PATH = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-qa-pass-20260416.json"
OUTPUT_DIR = SCRIPT_DIR / "tsv_output" / "open_source_review"
OUTPUT_TSV_PATH = OUTPUT_DIR / "open_source_import_review_queue.tsv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "open_source_import_review_summary.json"

KAIKKI_PATH = SCRIPT_DIR / "open_sources_20260416" / "wikimedia" / "kaikki-zazaki" / "kaikki.org-dictionary-Zazaki.jsonl"
KU_ETYM_DIR = SCRIPT_DIR / "open_sources_20260416" / "wikimedia" / "ku-wiktionary" / "etimolojiya-peyven-zazaki"
KU_COMPARISON_PATH = SCRIPT_DIR / "open_sources_20260416" / "wikimedia" / "ku-wiktionary" / "pevek-berhevdana-farisi-kurmanci-zazaki-inglizi.raw.wikitext"

PERMISSION_STATUS = "cc_by_sa"

BOLD_TEXT_RE = re.compile(r"'''([^']+)'''")
QUOTE_TEXT_RE = re.compile(r'"([^"]*)"')
COLON_SPLIT_RE = re.compile(r"\s*:\s*")
HEADWORD_VARIANT_SPLIT_RE = re.compile(r"\s*,\s*")

STAGING_FIELDS = [
    "review_id",
    "queue_phase",
    "source_family",
    "source_label",
    "source_path",
    "source_locator",
    "source_item_id",
    "permission_status",
    "candidate_kind",
    "auto_priority",
    "headword",
    "headword_variants",
    "headword_search",
    "entry_type",
    "gender_number",
    "source_gloss_tr",
    "source_gloss_en",
    "source_gloss_raw",
    "source_gloss_de",
    "source_etymology",
    "source_context",
    "source_notes",
    "source_tags",
    "raw_source_excerpt",
    "master_match_status",
    "master_match_titles",
    "auto_flags",
    "curation_status",
    "merge_decision",
    "curated_entry",
    "curated_parent",
    "curated_entry_type",
    "curated_gender_number",
    "curated_dialect",
    "curated_definition_tr",
    "curated_definition_en",
    "curated_definition_de",
    "curated_etymology_note",
    "curated_aliases",
    "curated_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone open-source review queue TSV without touching the master dictionary.")
    parser.add_argument("--master-snapshot-path", default=str(MASTER_SNAPSHOT_PATH))
    parser.add_argument("--kaikki-path", default=str(KAIKKI_PATH))
    parser.add_argument("--ku-etym-dir", default=str(KU_ETYM_DIR))
    parser.add_argument("--ku-comparison-path", default=str(KU_COMPARISON_PATH))
    parser.add_argument("--output-path", default=str(OUTPUT_TSV_PATH))
    parser.add_argument("--summary-path", default=str(OUTPUT_SUMMARY_PATH))
    return parser.parse_args()


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


def join_values(values: list[str], limit: int = 8, joiner: str = " || ") -> str:
    return compact_text_list(unique_values(values), limit=limit, joiner=joiner)


def strip_markup(value: str) -> str:
    text = clean_cell(value)
    text = re.sub(r"</?font[^>]*>", "", text)
    text = re.sub(r"</?sup[^>]*>", "", text)
    text = text.replace("'''", "").replace("''", "")
    return clean_cell(text)


def split_etym_headwords(value: str) -> list[str]:
    return unique_values([strip_markup(part) for part in HEADWORD_VARIANT_SPLIT_RE.split(strip_markup(value)) if strip_markup(part)])


def title_index_from_snapshot(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    title_index: dict[str, list[str]] = {}
    for entry in payload.get("entries", []):
        title = clean_cell(entry.get("title"))
        if not title:
            continue
        title_index.setdefault(normalize_search_text(title), []).append(title)
    for key, titles in title_index.items():
        title_index[key] = unique_values(titles)
    return title_index


def match_titles(index: dict[str, list[str]], values: list[str]) -> list[str]:
    matched: list[str] = []
    for value in values:
        matched.extend(index.get(normalize_search_text(value), []))
    return unique_values(matched)


def candidate_kind_for_kaikki(word: str, pos: str, flags: list[str]) -> str:
    flag_set = set(flags)
    if "structural_pos" in flag_set or "symbolic_word" in flag_set or "nonalpha" in flag_set:
        return "skip_candidate"
    if "alt_of" in flag_set or "form_of" in flag_set or "has_variant_forms" in flag_set:
        return "variant_candidate"
    if pos in {"phrase", "proverb"} or "multiword" in flag_set:
        return "phrase_candidate"
    return "entry_candidate"


def priority_for_kaikki(candidate_kind: str) -> str:
    if candidate_kind == "entry_candidate":
        return "high"
    if candidate_kind in {"variant_candidate", "phrase_candidate"}:
        return "medium"
    return "low"


def extract_kaikki_rows(path: Path, master_index: dict[str, list[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    relative_path = path.relative_to(SCRIPT_DIR).as_posix()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            record = json.loads(raw_line)
            word = clean_cell(record.get("word"))
            if not word:
                continue
            if normalize_search_text(word) in master_index:
                continue

            forms = [clean_cell(item.get("form")) for item in record.get("forms", [])]
            form_tags = [clean_cell(tag) for item in record.get("forms", []) for tag in item.get("tags", [])]
            record_related = [clean_cell(item.get("word")) for item in record.get("related", [])]
            record_synonyms = [clean_cell(item.get("word")) for item in record.get("synonyms", [])]

            flags: list[str] = []
            pos = clean_cell(record.get("pos"))
            if pos in {"character", "suffix"}:
                flags.append("structural_pos")
            if any(tag == "alternative" for tag in form_tags):
                flags.append("has_variant_forms")
            if any(char in word for char in "/#"):
                flags.append("symbolic_word")
            if word.startswith("-") or word.endswith("-"):
                flags.append("affix_like")
            if " " in word:
                flags.append("multiword")
            if not re.search(r"[A-Za-zÇĞİÖŞÜçğıöşüÊÎÛÂêîûâ]", word):
                flags.append("nonalpha")

            source_context_parts: list[str] = []
            ipa_values = [clean_cell(item.get("ipa")) for item in record.get("sounds", [])]
            hyphenations = ["-".join(clean_cell(part) for part in item.get("parts", []) if clean_cell(part)) for item in record.get("hyphenations", [])]
            if ipa_values:
                source_context_parts.append(f"IPA={join_values(ipa_values, limit=4, joiner=' | ')}")
            if hyphenations:
                source_context_parts.append(f"hyphenation={join_values(hyphenations, limit=4, joiner=' | ')}")

            for sense_index, sense in enumerate(record.get("senses", []), 1):
                sense_tags = [clean_cell(tag) for tag in sense.get("tags", [])]
                sense_flags = list(flags)
                if "alt-of" in sense_tags or sense.get("alt_of"):
                    sense_flags.append("alt_of")
                if "form-of" in sense_tags or sense.get("form_of"):
                    sense_flags.append("form_of")

                candidate_kind = candidate_kind_for_kaikki(word, pos, sense_flags)
                glosses_en = [clean_cell(gloss) for gloss in sense.get("glosses", [])]
                raw_glosses = [clean_cell(gloss) for gloss in sense.get("raw_glosses", [])]
                synonyms = [clean_cell(item.get("word")) for item in sense.get("synonyms", [])]
                related_words = [clean_cell(item.get("word")) for item in sense.get("related", [])]
                alt_of = [clean_cell(item.get("word")) for item in sense.get("alt_of", [])]
                form_of = [clean_cell(item.get("word")) for item in sense.get("form_of", [])]

                source_notes_parts: list[str] = []
                if forms:
                    source_notes_parts.append(f"forms={join_values(forms, limit=8, joiner=' | ')}")
                if alt_of:
                    source_notes_parts.append(f"alt_of={join_values(alt_of, limit=8, joiner=' | ')}")
                if form_of:
                    source_notes_parts.append(f"form_of={join_values(form_of, limit=8, joiner=' | ')}")
                if record_synonyms or synonyms:
                    source_notes_parts.append(f"synonyms={join_values(record_synonyms + synonyms, limit=8, joiner=' | ')}")
                if record_related or related_words:
                    source_notes_parts.append(f"related={join_values(record_related + related_words, limit=8, joiner=' | ')}")

                tags = unique_values([pos] + sense_tags + form_tags)
                raw_excerpt = {
                    "word": word,
                    "pos": pos,
                    "sense_id": clean_cell(sense.get("id")),
                    "glosses": unique_values(glosses_en)[:4],
                    "raw_glosses": unique_values(raw_glosses)[:4],
                    "etymology_text": clean_cell(record.get("etymology_text")),
                }

                rows.append(
                    {
                        "review_id": f"kaikki:{line_number:04d}:{sense_index:02d}",
                        "queue_phase": "phase1_kaikki",
                        "source_family": "kaikki",
                        "source_label": "Kaikki / Wiktionary Zazaki",
                        "source_path": relative_path,
                        "source_locator": f"line:{line_number}",
                        "source_item_id": clean_cell(sense.get("id")) or f"sense:{sense_index}",
                        "permission_status": PERMISSION_STATUS,
                        "candidate_kind": candidate_kind,
                        "auto_priority": priority_for_kaikki(candidate_kind),
                        "headword": word,
                        "headword_variants": join_values(forms, limit=10, joiner=" | "),
                        "headword_search": normalize_search_text(word),
                        "entry_type": pos,
                        "gender_number": join_values([tag for tag in sense_tags if tag in {"masculine", "feminine", "common-gender", "neuter"}], limit=4, joiner=" | "),
                        "source_gloss_tr": "",
                        "source_gloss_en": join_values(glosses_en, limit=8),
                        "source_gloss_raw": join_values(raw_glosses, limit=8),
                        "source_gloss_de": "",
                        "source_etymology": clean_cell(record.get("etymology_text")),
                        "source_context": " || ".join(source_context_parts),
                        "source_notes": " || ".join(source_notes_parts),
                        "source_tags": join_values(tags, limit=12, joiner=" | "),
                        "raw_source_excerpt": json.dumps(raw_excerpt, ensure_ascii=False),
                        "master_match_status": "no_exact_title_match",
                        "master_match_titles": "",
                        "auto_flags": join_values(sense_flags, limit=12, joiner=" | "),
                        "curation_status": "pending",
                        "merge_decision": "",
                        "curated_entry": "",
                        "curated_parent": "",
                        "curated_entry_type": "",
                        "curated_gender_number": "",
                        "curated_dialect": "",
                        "curated_definition_tr": "",
                        "curated_definition_en": "",
                        "curated_definition_de": "",
                        "curated_etymology_note": "",
                        "curated_aliases": "",
                        "curated_notes": "",
                    }
                )
    return rows


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

    if remainder.startswith('"'):
        gloss_match = QUOTE_TEXT_RE.search(remainder)
        gloss = strip_markup(gloss_match.group(1) if gloss_match else "")
        remainder = remainder[gloss_match.end():].strip() if gloss_match else remainder
        if remainder.startswith(":"):
            remainder = remainder[1:].strip()
        return headwords, gloss, strip_markup(remainder)

    if remainder.startswith(":"):
        remainder = remainder[1:].strip()
        gloss_match = QUOTE_TEXT_RE.search(remainder)
        gloss = strip_markup(gloss_match.group(1) if gloss_match else "")
        return headwords, gloss, strip_markup(remainder)

    return headwords, "", strip_markup(remainder)


def build_etymology_row(
    *,
    relative_path: str,
    file_path: Path,
    line_number: int,
    line_text: str,
    note_lines: list[str],
    master_index: dict[str, list[str]],
) -> dict[str, str] | None:
    headwords, gloss, etymology = parse_etym_entry_line(line_text)
    if not headwords:
        return None

    matched_titles = match_titles(master_index, headwords)
    if len(matched_titles) == len(headwords):
        match_status = "all_headwords_in_master"
    elif matched_titles:
        match_status = "mixed_headword_cluster"
    else:
        match_status = "no_headwords_in_master"

    candidate_kind = "etymology_enrichment" if matched_titles else "entry_or_etymology_candidate"
    auto_priority = "medium" if matched_titles else "high"

    return {
        "review_id": f"ku_etym:{file_path.stem}:{line_number:04d}",
        "queue_phase": "phase2_ku_etymology",
        "source_family": "ku_wiktionary_etymology",
        "source_label": "ku.wiktionary etymology pages",
        "source_path": relative_path,
        "source_locator": f"{file_path.name}:{line_number}",
        "source_item_id": "",
        "permission_status": PERMISSION_STATUS,
        "candidate_kind": candidate_kind,
        "auto_priority": auto_priority,
        "headword": headwords[0],
        "headword_variants": " | ".join(headwords),
        "headword_search": normalize_search_text(headwords[0]),
        "entry_type": "",
        "gender_number": "",
        "source_gloss_tr": "",
        "source_gloss_en": gloss,
        "source_gloss_raw": "",
        "source_gloss_de": "",
        "source_etymology": etymology,
        "source_context": f"letter_bucket={file_path.stem}",
        "source_notes": " || ".join(note_lines),
        "source_tags": "etymology",
        "raw_source_excerpt": " || ".join([line_text] + note_lines),
        "master_match_status": match_status,
        "master_match_titles": " | ".join(matched_titles),
        "auto_flags": "multi_headword" if len(headwords) > 1 else "",
        "curation_status": "pending",
        "merge_decision": "",
        "curated_entry": "",
        "curated_parent": "",
        "curated_entry_type": "",
        "curated_gender_number": "",
        "curated_dialect": "",
        "curated_definition_tr": "",
        "curated_definition_en": "",
        "curated_definition_de": "",
        "curated_etymology_note": "",
        "curated_aliases": "",
        "curated_notes": "",
    }


def extract_ku_etymology_rows(directory: Path, master_index: dict[str, list[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    relative_dir = directory.relative_to(SCRIPT_DIR).as_posix()
    for file_path in sorted(directory.glob("*.raw.wikitext")):
        current_line_number = 0
        current_line_text = ""
        current_note_lines: list[str] = []

        def flush_current() -> None:
            nonlocal current_line_number, current_line_text, current_note_lines
            if not current_line_text:
                return
            row = build_etymology_row(
                relative_path=f"{relative_dir}/{file_path.name}",
                file_path=file_path,
                line_number=current_line_number,
                line_text=current_line_text,
                note_lines=current_note_lines,
                master_index=master_index,
            )
            if row is not None:
                rows.append(row)
            current_line_number = 0
            current_line_text = ""
            current_note_lines = []

        for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
            line = clean_cell(raw_line)
            if not line:
                continue
            if line.startswith("*"):
                flush_current()
                current_line_number = line_number
                current_line_text = line
            elif current_line_text and line.startswith(":"):
                current_note_lines.append(strip_markup(line.lstrip(":").strip()))
            else:
                continue
        flush_current()
    return rows


def parse_comparison_line(section: str, line: str) -> tuple[str, str, str, str] | None:
    parts = [clean_cell(part) for part in COLON_SPLIT_RE.split(line)]
    if section == "persian_kurdish_zazaki":
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2], ""
        return None
    if section == "kurdish_zazaki":
        if len(parts) == 3:
            return "", parts[0], parts[1], parts[2]
        if len(parts) == 2:
            return "", parts[0], parts[1], ""
        return None
    return None


def split_variants(value: str) -> list[str]:
    return unique_values([part for part in re.split(r"\s*/\s*", clean_cell(value)) if clean_cell(part)])


def candidate_kind_for_comparison(forms: list[str], matched_titles: list[str]) -> str:
    phrase_like = any(" " in form or any(char in form for char in "!?") for form in forms)
    if phrase_like:
        return "phrase_candidate"
    if matched_titles or len(forms) > 1:
        return "alias_candidate"
    return "entry_candidate"


def extract_ku_comparison_rows(path: Path, master_index: dict[str, list[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    relative_path = path.relative_to(SCRIPT_DIR).as_posix()
    section = "persian_kurdish_zazaki"
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = clean_cell(raw_line)
        if not line or line.startswith("{{") or line.startswith("[["):
            continue
        if line.startswith("=="):
            if "kurmancî û zazakî" in line:
                section = "kurdish_zazaki"
            continue
        if line.startswith("'''") or line.startswith("Persian"):
            continue

        parsed = parse_comparison_line(section, line)
        if parsed is None:
            continue
        persian, kurdish, zazaki, english = parsed
        zazaki_forms = split_variants(zazaki)
        if not zazaki_forms:
            continue

        matched_titles = match_titles(master_index, zazaki_forms)
        unmatched_count = len(zazaki_forms) - len(matched_titles)
        has_variant_cluster = len(zazaki_forms) > 1
        if not unmatched_count and not has_variant_cluster:
            continue

        if unmatched_count == len(zazaki_forms):
            match_status = "no_exact_title_match"
        elif unmatched_count == 0:
            match_status = "all_forms_in_master"
        else:
            match_status = "mixed_form_cluster"

        candidate_kind = candidate_kind_for_comparison(zazaki_forms, matched_titles)
        auto_priority = "medium" if unmatched_count else "low"
        context_parts: list[str] = []
        if persian:
            context_parts.append(f"Persian={persian}")
        if kurdish:
            context_parts.append(f"Kurdish={kurdish}")

        rows.append(
            {
                "review_id": f"ku_cmp:{line_number:04d}",
                "queue_phase": "phase3_ku_comparison",
                "source_family": "ku_wiktionary_comparison",
                "source_label": "ku.wiktionary comparison page",
                "source_path": relative_path,
                "source_locator": f"line:{line_number}",
                "source_item_id": "",
                "permission_status": PERMISSION_STATUS,
                "candidate_kind": candidate_kind,
                "auto_priority": auto_priority,
                "headword": zazaki_forms[0],
                "headword_variants": " | ".join(zazaki_forms),
                "headword_search": normalize_search_text(zazaki_forms[0]),
                "entry_type": "",
                "gender_number": "",
                "source_gloss_tr": "",
                "source_gloss_en": english,
                "source_gloss_raw": "",
                "source_gloss_de": "",
                "source_etymology": "",
                "source_context": " || ".join(context_parts),
                "source_notes": "",
                "source_tags": section,
                "raw_source_excerpt": line,
                "master_match_status": match_status,
                "master_match_titles": " | ".join(matched_titles),
                "auto_flags": "multi_variant" if has_variant_cluster else "",
                "curation_status": "pending",
                "merge_decision": "",
                "curated_entry": "",
                "curated_parent": "",
                "curated_entry_type": "",
                "curated_gender_number": "",
                "curated_dialect": "",
                "curated_definition_tr": "",
                "curated_definition_en": "",
                "curated_definition_de": "",
                "curated_etymology_note": "",
                "curated_aliases": "",
                "curated_notes": "",
            }
        )
    return rows


def build_summary(rows: list[dict[str, str]], args: argparse.Namespace) -> dict[str, object]:
    counters = {
        "queue_phase": Counter(),
        "source_family": Counter(),
        "candidate_kind": Counter(),
        "master_match_status": Counter(),
        "auto_priority": Counter(),
    }
    for row in rows:
        for key, counter in counters.items():
            counter[clean_cell(row.get(key))] += 1
    return {
        "generated_from": {
            "master_snapshot_path": str(Path(args.master_snapshot_path)),
            "kaikki_path": str(Path(args.kaikki_path)),
            "ku_etym_dir": str(Path(args.ku_etym_dir)),
            "ku_comparison_path": str(Path(args.ku_comparison_path)),
        },
        "row_count": len(rows),
        "counts": {key: dict(counter) for key, counter in counters.items()},
    }


def main() -> None:
    args = parse_args()
    master_index = title_index_from_snapshot(Path(args.master_snapshot_path))

    rows = []
    rows.extend(extract_kaikki_rows(Path(args.kaikki_path), master_index))
    rows.extend(extract_ku_etymology_rows(Path(args.ku_etym_dir), master_index))
    rows.extend(extract_ku_comparison_rows(Path(args.ku_comparison_path), master_index))

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    write_tsv(output_path, STAGING_FIELDS, rows)
    write_json(summary_path, build_summary(rows, args))
    print(json.dumps({"output_path": str(output_path), "row_count": len(rows), "summary_path": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
