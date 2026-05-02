#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, normalize_search_text, write_json, write_tsv
from run_ollama_master_translation_flag_review import build_flags


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-20260418-134935.json"
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418"

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ÿÇĞİÖŞÜçğıöşüêîûâ]+")
STRONG_TITLE_PUNCT = ",?!;:()\"“”"
EN_SENTENCE_START = {
    "he",
    "she",
    "they",
    "we",
    "you",
    "it",
    "my",
    "your",
    "our",
    "their",
    "his",
    "her",
    "when",
    "while",
    "if",
    "because",
    "as",
}
EN_ARTICLES = {"the", "a", "an"}
DE_SENTENCE_START = {
    "er",
    "sie",
    "wir",
    "ihr",
    "du",
    "mein",
    "dein",
    "sein",
    "unser",
    "wenn",
    "als",
    "weil",
    "da",
}
DE_ARTICLES = {"der", "die", "das", "ein", "eine", "einer", "einem", "einen"}
WEAK_PARENT_TOKENS = {
    "a",
    "ae",
    "aye",
    "awa",
    "o",
    "u",
    "ma",
    "mı",
    "mi",
    "şıma",
    "ema",
    "ez",
    "ke",
    "ra",
    "la",
}

KNOWN_CORRECTIONS = {
    "dict:caf7d49b775f317e1cd5fe8069517e30:0": {
        "issue_type": "retranslate",
        "priority": "high",
        "heuristic_flags": "known_bad_gloss | likely_oyle_not_ogle",
        "reason": (
            "Known bad gloss. `otır` is used elsewhere in the source for `öyle`, so `öğle değil mi?` "
            "is almost certainly a contamination. This should be reviewed against `la niye` and related "
            "`otır` entries."
        ),
        "suggested_tr": "öyle değil mi?",
        "suggested_en": "isn't that so?",
        "suggested_de": "nicht wahr?",
    },
    "dict:8adf95ff3a8622c99ba9e4dd59e0391e:0": {
        "issue_type": "retranslate",
        "priority": "high",
        "heuristic_flags": "known_live_page_issue | over_specific_polysemous_gloss",
        "reason": (
            "Mapped directly from live page `ll_dictionary_entry=68706` (`xali bıyayış`). The TR gloss is a "
            "semicolon-separated multi-sense entry (`tenhalaşmak; arılaşmak; boşalmak`), but EN/DE collapse it "
            "into the over-specific sense `ejaculate/ejakulieren`."
        ),
        "suggested_tr": "tenhalaşmak; arılaşmak; boşalmak",
        "suggested_en": "to become isolated; to become purified; to become empty",
        "suggested_de": "einsam werden; rein werden; leer werden",
    }
}

ISSUE_FIELDS = [
    "issue_type",
    "priority",
    "confidence_score",
    "entry_index",
    "sense_index",
    "import_key",
    "source_id",
    "source_row_idx",
    "title",
    "current_parent",
    "suggested_parent",
    "definition_tr",
    "definition_en",
    "definition_de",
    "suggested_tr",
    "suggested_en",
    "suggested_de",
    "heuristic_flags",
    "reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a flag report for suspicious dictionary entries.")
    parser.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--priority-limit", type=int, default=150)
    return parser.parse_args()


def decode_text(value: object) -> str:
    return clean_cell(html.unescape(clean_cell(value)))


def tokenize(value: str) -> list[str]:
    return TOKEN_RE.findall(decode_text(value).lower())


def token_count(value: str) -> int:
    return len(tokenize(value))


def title_has_strong_punct(title: str) -> bool:
    decoded = decode_text(title)
    return any(ch in decoded for ch in STRONG_TITLE_PUNCT) or "..." in decoded or "…" in decoded


def is_sentence_like_title(title: str) -> bool:
    tokens = tokenize(title)
    if len(tokens) >= 5:
        return True
    if len(tokens) >= 3 and title_has_strong_punct(title):
        return True
    if tokens and tokens[0] in WEAK_PARENT_TOKENS and len(tokens) >= 3:
        return True
    if decode_text(title).startswith(("...", "…")):
        return True
    return False


def is_short_generic_gloss(text: str) -> bool:
    decoded = decode_text(text)
    return token_count(decoded) <= 5 and "1)" not in decoded and ";" not in decoded and "|" not in decoded


def sentence_artifact_flags(text: str, lang: str) -> list[str]:
    decoded = decode_text(text)
    tokens = tokenize(decoded)
    if not tokens:
        return []
    flags: list[str] = []
    first = tokens[0]
    starts = EN_SENTENCE_START if lang == "en" else DE_SENTENCE_START
    articles = EN_ARTICLES if lang == "en" else DE_ARTICLES
    if first in starts:
        flags.append(f"{lang}_starts_with_sentence_word")
    if first in articles and len(tokens) >= 5:
        flags.append(f"{lang}_starts_with_article_long")
    if decoded.endswith("."):
        flags.append(f"{lang}_terminal_period")
    if lang == "en" and " in case of call" in decoded.lower():
        flags.append("en_parenthetical_literal")
    if lang == "de" and " im falle eines anrufs" in decoded.lower():
        flags.append("de_parenthetical_literal")
    if lang == "en" and "girl name" in decoded.lower():
        flags.append("en_name_label_literal")
    if lang == "de" and "mädchenname" in decoded.lower():
        flags.append("de_name_label_literal")
    if lang == "en" and "worshipped" in decoded.lower():
        flags.append("en_semantic_leap")
    if lang == "de" and "angebetet" in decoded.lower():
        flags.append("de_semantic_leap")
    return flags


def build_parent_index(entries: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    index: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        title = decode_text(entry.get("title"))
        title_norm = normalize_search_text(title)
        title_token_count = token_count(title)
        senses = entry.get("senses") or []
        for sense in senses:
            source_id = clean_cell(sense.get("source_id"))
            if not source_id or not title_norm:
                continue
            index[source_id][title_norm].append(
                {
                    "title": title,
                    "title_norm": title_norm,
                    "title_token_count": str(title_token_count),
                    "entry_type": clean_cell(sense.get("entry_type")),
                    "parent": decode_text(sense.get("parent")),
                    "source_row_idx": clean_cell(sense.get("source_row_idx")),
                    "import_key": clean_cell(entry.get("import_key")),
                }
            )
    for source_id in list(index):
        for title_norm in list(index[source_id]):
            index[source_id][title_norm].sort(
                key=lambda item: (
                    int(clean_cell(item.get("title_token_count")) or "999"),
                    decode_text(item.get("title")),
                    clean_cell(item.get("source_row_idx")),
                )
            )
    return index


def best_parent_candidate(*, source_id: str, title: str, title_norm: str, parent_index: dict[str, dict[str, list[dict[str, str]]]]) -> str:
    if not source_id or not title_norm:
        return ""
    title_tokens = title_norm.split()
    if len(title_tokens) < 2:
        return ""
    min_prefix = 2 if title_tokens[0] in WEAK_PARENT_TOKENS else 1
    max_prefix = min(4, len(title_tokens) - 1)
    for prefix_len in range(max_prefix, min_prefix - 1, -1):
        prefix_norm = " ".join(title_tokens[:prefix_len])
        for candidate in parent_index.get(source_id, {}).get(prefix_norm, []):
            if clean_cell(candidate.get("title_norm")) == title_norm:
                continue
            if clean_cell(candidate.get("entry_type")) != "word":
                continue
            candidate_token_count = int(clean_cell(candidate.get("title_token_count")) or "999")
            if candidate_token_count >= len(title_tokens):
                continue
            return decode_text(candidate.get("title"))
    return ""


def issue_priority(score: int) -> str:
    if score >= 10:
        return "high"
    if score >= 6:
        return "medium"
    return "low"


def build_translation_issue(
    *,
    entry_index: int,
    sense_index: int,
    import_key: str,
    source_id: str,
    source_row_idx: str,
    title: str,
    current_parent: str,
    tr: str,
    en: str,
    de: str,
) -> dict[str, str] | None:
    known = KNOWN_CORRECTIONS.get(f"{import_key}:{sense_index}")
    heuristic_flags: list[str] = []
    base_score, base_flags = build_flags(tr, en, de)
    heuristic_flags.extend(base_flags)
    score = 0

    short_generic = is_short_generic_gloss(tr)
    title_tokens = token_count(title)
    tr_tokens = token_count(tr)
    en_tokens = token_count(en)
    de_tokens = token_count(de)
    en_sentence_flags = sentence_artifact_flags(en, "en")
    de_sentence_flags = sentence_artifact_flags(de, "de")
    heuristic_flags.extend(en_sentence_flags)
    heuristic_flags.extend(de_sentence_flags)

    if any(flag in base_flags for flag in {"en_duplicate_tokens", "de_duplicate_tokens"}):
        score += 1
    if any(flag in base_flags for flag in {"en_too_short_for_source", "de_too_short_for_source"}):
        score += 3
    if any(flag in base_flags for flag in {"en_enumeration_mismatch", "de_enumeration_mismatch"}):
        score += 2
    if "en_de_identical" in base_flags:
        score += 1

    if short_generic and en_sentence_flags:
        heuristic_flags.append("en_sentence_for_short_gloss")
        score += 2
    if short_generic and de_sentence_flags:
        heuristic_flags.append("de_sentence_for_short_gloss")
        score += 2
    if title_tokens <= 3 and tr_tokens <= 5 and en_tokens >= 6:
        heuristic_flags.append("en_over_specific_for_short_entry")
        score += 2
    if title_tokens <= 3 and tr_tokens <= 5 and de_tokens >= 6:
        heuristic_flags.append("de_over_specific_for_short_entry")
        score += 2
    if short_generic and en and de and (en.endswith(".") or de.endswith(".")):
        heuristic_flags.append("sentence_punctuation_for_gloss")
        score += 1
    if "en_starts_with_sentence_word" in en_sentence_flags:
        score += 3
    if "de_starts_with_sentence_word" in de_sentence_flags:
        score += 3
    if "en_starts_with_article_long" in en_sentence_flags:
        score += 4
    if "de_starts_with_article_long" in de_sentence_flags:
        score += 4
    if "en_parenthetical_literal" in en_sentence_flags:
        score += 4
    if "de_parenthetical_literal" in de_sentence_flags:
        score += 4
    if "en_semantic_leap" in en_sentence_flags:
        score += 6
    if "de_semantic_leap" in de_sentence_flags:
        score += 6
    if "en_name_label_literal" in en_sentence_flags:
        score += 1
    if "de_name_label_literal" in de_sentence_flags:
        score += 1

    unique_flags = []
    seen_flags: set[str] = set()
    for flag in heuristic_flags:
        if flag and flag not in seen_flags:
            seen_flags.add(flag)
            unique_flags.append(flag)

    triggered = bool(known) or score >= 4 or base_score >= 8
    if not triggered:
        return None

    reason_parts: list[str] = []
    if known:
        reason_parts.append(clean_cell(known.get("reason")))
    else:
        if "en_sentence_for_short_gloss" in unique_flags or "de_sentence_for_short_gloss" in unique_flags:
            reason_parts.append("Short generic TR gloss is rendered as a full sentence in EN/DE.")
        if "en_over_specific_for_short_entry" in unique_flags or "de_over_specific_for_short_entry" in unique_flags:
            reason_parts.append("EN/DE translation appears more specific than the source gloss warrants.")
        if "en_parenthetical_literal" in unique_flags or "de_parenthetical_literal" in unique_flags:
            reason_parts.append("Literal rendering of parenthetical metadata should be retranslated idiomatically.")
        if "en_semantic_leap" in unique_flags or "de_semantic_leap" in unique_flags:
            reason_parts.append("Target translation shows a semantic leap not supported by the TR gloss.")
        if not reason_parts:
            reason_parts.append("Heuristic translation-risk pass flagged this entry for manual retranslation review.")

    result = {
        "issue_type": "retranslate",
        "priority": clean_cell(known.get("priority")) if known else issue_priority(score),
        "confidence_score": str(max(score, 12 if known else score)),
        "entry_index": str(entry_index),
        "sense_index": str(sense_index),
        "import_key": import_key,
        "source_id": source_id,
        "source_row_idx": source_row_idx,
        "title": title,
        "current_parent": current_parent,
        "suggested_parent": "",
        "definition_tr": tr,
        "definition_en": en,
        "definition_de": de,
        "suggested_tr": clean_cell(known.get("suggested_tr")) if known else "",
        "suggested_en": clean_cell(known.get("suggested_en")) if known else "",
        "suggested_de": clean_cell(known.get("suggested_de")) if known else "",
        "heuristic_flags": " | ".join(unique_flags) if not known else clean_cell(known.get("heuristic_flags")),
        "reason": " ".join(part for part in reason_parts if part),
    }
    return result


def build_reorganization_issue(
    *,
    entry_index: int,
    sense_index: int,
    import_key: str,
    source_id: str,
    source_row_idx: str,
    title: str,
    title_norm: str,
    current_parent: str,
    tr: str,
    en: str,
    de: str,
    entry_type: str,
    parent_index: dict[str, dict[str, list[dict[str, str]]]],
) -> dict[str, str] | None:
    if current_parent:
        return None
    if source_id not in {"dezd-kirmancki-dictionary", "palu-bingol-harun-turgut"}:
        return None
    if entry_type != "word":
        return None
    if not is_sentence_like_title(title):
        return None

    score = 0
    heuristic_flags: list[str] = ["sentence_like_title"]
    if title_has_strong_punct(title):
        score += 2
        heuristic_flags.append("title_has_clause_punctuation")
    title_tokens = token_count(title)
    if title_tokens >= 5:
        score += 2
        heuristic_flags.append("long_title")
    if tokenize(title) and tokenize(title)[0] in WEAK_PARENT_TOKENS:
        score += 2
        heuristic_flags.append("weak_function_word_start")
    if token_count(tr) >= 3:
        score += 1
        heuristic_flags.append("descriptive_gloss")
    if en.endswith(".") or de.endswith("."):
        score += 1
        heuristic_flags.append("sentence_like_translation")

    suggested_parent = best_parent_candidate(
        source_id=source_id,
        title=title,
        title_norm=title_norm,
        parent_index=parent_index,
    )
    if suggested_parent:
        score += 3
        heuristic_flags.append("matched_prefix_parent")

    if score < 5:
        return None

    if suggested_parent:
        reason = (
            "Standalone entry looks like a clause/example rather than a headword, and its opening tokens match "
            f"the likely parent `{suggested_parent}` from the same source."
        )
    else:
        reason = "Standalone entry looks more like an example clause or inflected phrase than a reusable headword."

    return {
        "issue_type": "reorganize_as_example",
        "priority": issue_priority(score),
        "confidence_score": str(score),
        "entry_index": str(entry_index),
        "sense_index": str(sense_index),
        "import_key": import_key,
        "source_id": source_id,
        "source_row_idx": source_row_idx,
        "title": title,
        "current_parent": current_parent,
        "suggested_parent": suggested_parent,
        "definition_tr": tr,
        "definition_en": en,
        "definition_de": de,
        "suggested_tr": "",
        "suggested_en": "",
        "suggested_de": "",
        "heuristic_flags": " | ".join(heuristic_flags),
        "reason": reason,
    }


def build_markdown_summary(
    *,
    snapshot_path: Path,
    translation_issues: list[dict[str, str]],
    reorg_issues: list[dict[str, str]],
    priority_issues: list[dict[str, str]],
) -> str:
    translation_by_source = Counter(clean_cell(row.get("source_id")) for row in translation_issues)
    reorg_by_source = Counter(clean_cell(row.get("source_id")) for row in reorg_issues)
    lines = [
        "# Dictionary Issue Flag Pass",
        "",
        f"- Snapshot: `{snapshot_path}`",
        f"- Retranslation candidates: **{len(translation_issues)}**",
        f"- Reorganization candidates: **{len(reorg_issues)}**",
        f"- Priority review rows: **{len(priority_issues)}**",
        "",
        "## Retranslation by source",
        "",
    ]
    for source_id, count in translation_by_source.most_common():
        lines.append(f"- `{source_id}`: {count}")
    lines.extend(["", "## Reorganization by source", ""])
    for source_id, count in reorg_by_source.most_common():
        lines.append(f"- `{source_id}`: {count}")
    lines.extend(["", "## Top Priority Samples", ""])
    for row in priority_issues[:20]:
        lines.append(
            f"- `{row['issue_type']}` `{row['title']}` ({row['source_id']}:{row['source_row_idx']}) -> {row['reason']}"
        )
    lines.append("")
    return "\n".join(lines)


def flag_set(row: dict[str, str]) -> set[str]:
    return {clean_cell(part) for part in clean_cell(row.get("heuristic_flags")).split("|") if clean_cell(part)}


def issue_sort_key(row: dict[str, str]) -> tuple[int, int, int, str, str]:
    flags = flag_set(row)
    critical_rank = 6
    if "known_bad_gloss" in flags or "known_live_page_issue" in flags:
        critical_rank = 0
    elif "en_semantic_leap" in flags or "de_semantic_leap" in flags:
        critical_rank = 1
    elif "en_parenthetical_literal" in flags or "de_parenthetical_literal" in flags:
        critical_rank = 2
    elif "en_starts_with_sentence_word" in flags or "de_starts_with_sentence_word" in flags:
        critical_rank = 3
    elif "en_starts_with_article_long" in flags or "de_starts_with_article_long" in flags:
        critical_rank = 4
    elif "matched_prefix_parent" in flags:
        critical_rank = 5
    return (
        critical_rank,
        {"high": 0, "medium": 1, "low": 2}.get(clean_cell(row.get("priority")), 3),
        -int(clean_cell(row.get("confidence_score")) or "0"),
        clean_cell(row.get("source_id")),
        clean_cell(row.get("title")),
    )


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    output_dir = Path(args.output_dir)

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    parent_index = build_parent_index(entries)

    translation_issues: list[dict[str, str]] = []
    reorg_issues: list[dict[str, str]] = []

    for entry_index, entry in enumerate(entries):
        import_key = clean_cell(entry.get("import_key"))
        title = decode_text(entry.get("title"))
        title_norm = normalize_search_text(title)
        senses = entry.get("senses") or []
        for sense_index, sense in enumerate(senses):
            source_id = clean_cell(sense.get("source_id"))
            source_row_idx = clean_cell(sense.get("source_row_idx"))
            current_parent = decode_text(sense.get("parent"))
            entry_type = clean_cell(sense.get("entry_type"))
            translations = sense.get("translations") or {}
            tr = decode_text(translations.get("tr") or sense.get("definition"))
            en = decode_text(translations.get("en"))
            de = decode_text(translations.get("de"))

            translation_issue = build_translation_issue(
                entry_index=entry_index,
                sense_index=sense_index,
                import_key=import_key,
                source_id=source_id,
                source_row_idx=source_row_idx,
                title=title,
                current_parent=current_parent,
                tr=tr,
                en=en,
                de=de,
            )
            if translation_issue:
                translation_issues.append(translation_issue)

            reorg_issue = build_reorganization_issue(
                entry_index=entry_index,
                sense_index=sense_index,
                import_key=import_key,
                source_id=source_id,
                source_row_idx=source_row_idx,
                title=title,
                title_norm=title_norm,
                current_parent=current_parent,
                tr=tr,
                en=en,
                de=de,
                entry_type=entry_type,
                parent_index=parent_index,
            )
            if reorg_issue:
                reorg_issues.append(reorg_issue)

    translation_issues.sort(key=issue_sort_key)
    reorg_issues.sort(key=issue_sort_key)

    must_include_ids = {
        (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index")), clean_cell(row.get("issue_type")))
        for row in translation_issues
        if {"known_bad_gloss", "known_live_page_issue", "en_semantic_leap", "de_semantic_leap"} & flag_set(row)
    }
    priority_issues: list[dict[str, str]] = []
    seen_priority_ids: set[tuple[str, str, str]] = set()
    for row in translation_issues + reorg_issues:
        row_id = (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index")), clean_cell(row.get("issue_type")))
        if row_id in must_include_ids and row_id not in seen_priority_ids:
            priority_issues.append(row)
            seen_priority_ids.add(row_id)
    for row in translation_issues + reorg_issues:
        row_id = (clean_cell(row.get("import_key")), clean_cell(row.get("sense_index")), clean_cell(row.get("issue_type")))
        if row_id in seen_priority_ids:
            continue
        priority_issues.append(row)
        seen_priority_ids.add(row_id)
        if len(priority_issues) >= args.priority_limit:
            break
    priority_issues = priority_issues[: args.priority_limit]

    ensure_parent_dir(output_dir / "placeholder")
    write_tsv(output_dir / "translation_retranslation_candidates.tsv", ISSUE_FIELDS, translation_issues)
    write_tsv(output_dir / "reorganization_candidates.tsv", ISSUE_FIELDS, reorg_issues)
    write_tsv(output_dir / "priority_review.tsv", ISSUE_FIELDS, priority_issues)

    summary = {
        "snapshot_path": str(snapshot_path),
        "entry_count": len(entries),
        "translation_candidate_count": len(translation_issues),
        "reorganization_candidate_count": len(reorg_issues),
        "priority_review_count": len(priority_issues),
        "translation_by_source": dict(Counter(clean_cell(row.get("source_id")) for row in translation_issues)),
        "reorganization_by_source": dict(Counter(clean_cell(row.get("source_id")) for row in reorg_issues)),
        "files": {
            "translation_retranslation_candidates_tsv": str(output_dir / "translation_retranslation_candidates.tsv"),
            "reorganization_candidates_tsv": str(output_dir / "reorganization_candidates.tsv"),
            "priority_review_tsv": str(output_dir / "priority_review.tsv"),
            "summary_json": str(output_dir / "summary.json"),
            "summary_md": str(output_dir / "summary.md"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        build_markdown_summary(
            snapshot_path=snapshot_path,
            translation_issues=translation_issues,
            reorg_issues=reorg_issues,
            priority_issues=priority_issues,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
