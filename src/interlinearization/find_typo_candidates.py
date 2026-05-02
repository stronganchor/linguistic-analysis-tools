#!/usr/bin/env python3
"""Find likely transcription typo candidates without changing source text."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from auto_interlinearize import ROOT, fold_key, normalize_key


OUTPUT_DIR = ROOT / "interlinearization/outputs"
SOURCE_DIR = ROOT / "interlinearization/sources/ak_categories"
OUT_TSV = OUTPUT_DIR / "ak_typo_candidates_refined_20260501.tsv"
OUT_MD = OUTPUT_DIR / "ak_typo_candidates_refined_20260501.md"
OUT_SHORTLIST_TSV = OUTPUT_DIR / "ak_typo_candidates_refined_20260501.shortlist.tsv"
OUT_SHORTLIST_MD = OUTPUT_DIR / "ak_typo_candidates_refined_20260501.shortlist.md"

VOWELS = set("aeêıioöuûüâ")


def clean_surface(value: str) -> str:
    key = normalize_key(value)
    key = key.strip(".,;:!?()[]{}\"'`")
    return key


def skeleton(value: str) -> str:
    value = clean_surface(value)
    value = value.replace("tş", "ç")
    value = value.replace("tsch", "ç")
    value = value.replace("êy", "ê")
    value = value.replace("ey", "e")
    value = value.replace("ıy", "ı")
    if len(value) > 1 and value[0] == "h" and value[1] in VOWELS:
        value = value[1:]
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("ı", "i")
    return value


def edits1(value: str) -> set[str]:
    letters = "abcçdefgğhıijklmnoöprsştuûüvyzqwxâê'"
    splits = [(value[:i], value[i:]) for i in range(len(value) + 1)]
    deletes = {left + right[1:] for left, right in splits if right}
    transposes = {left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1}
    replaces = {left + c + right[1:] for left, right in splits if right for c in letters}
    inserts = {left + c + right for left, right in splits for c in letters}
    return deletes | transposes | replaces | inserts


def load_line_meta() -> dict[tuple[str, str], dict[str, str]]:
    meta: dict[tuple[str, str], dict[str, str]] = {}
    for path in SOURCE_DIR.glob("ak_cat_*.tsv"):
        token_name = path.name.replace(".tsv", ".tokens.tsv")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                meta[(token_name, row["id"])] = row
    return meta


def load_tokens() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in OUTPUT_DIR.glob("ak_cat_*.tokens.tsv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["_file"] = path.name
                rows.append(row)
    return rows


def build_known(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    best_by_key: dict[str, dict[str, str]] = {}
    for row in rows:
        form = clean_surface(row.get("form", ""))
        if not form or " " in form or len(form) < 2:
            continue
        try:
            conf = float(row.get("confidence") or 0)
        except ValueError:
            conf = 0.0
        gloss = row.get("display_gloss") or row.get("gloss_en") or ""
        if conf < 0.88 or not gloss or gloss == "?":
            continue
        previous = best_by_key.get(form)
        if previous is None or conf > float(previous.get("confidence") or 0):
            best_by_key[form] = row
    return best_by_key


def reason_for(form: str, candidate: str) -> str:
    rules: list[str] = []
    if form.startswith("h") and len(form) > 1 and form[1] in VOWELS and not candidate.startswith("h"):
        rules.append("drop initial h- before vowel")
    if candidate.startswith("h") and len(candidate) > 1 and candidate[1] in VOWELS and not form.startswith("h"):
        rules.append("add initial h- before vowel")
    if "êy" in form or "êy" in candidate or "ey" in form or "ey" in candidate:
        rules.append("ê/êy smoothing")
    if "tş" in form or "tş" in candidate or "tsch" in form or "tsch" in candidate:
        rules.append("tş/ç affricate smoothing")
    if fold_key(form) == fold_key(candidate) and form != candidate:
        rules.append("diacritic or ı/i smoothing")
    if not rules:
        rules.append("one-edit match to approved form")
    return "; ".join(dict.fromkeys(rules))


def is_low_signal_vowel_variant(form: str, candidate: str, rule: str) -> bool:
    """Skip normal A.K. surface-vowel alternations unless another typo signal exists."""
    high_signal = ("tş/ç", "ê/êy", "initial h")
    if any(marker in rule for marker in high_signal):
        return False
    return fold_key(form) == fold_key(candidate)


def shortlist_rank(item: dict[str, str]) -> tuple[int, str]:
    rule = item["rule"]
    if "tş/ç" in rule:
        return (0, item["suggested_form"])
    if "ê/êy" in rule:
        return (1, item["suggested_form"])
    if "initial h" in rule:
        return (2, item["suggested_form"])
    return (9, item["suggested_form"])


def build_shortlist(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for item in candidates:
        rule = item["rule"]
        if item["priority"] != "high":
            continue
        if not any(marker in rule for marker in ("tş/ç", "ê/êy", "initial h")):
            continue
        grouped[(item["lesson_id"], item["line_id"], item["current_form"])].append(item)
    shortlist = [sorted(items, key=shortlist_rank)[0] for items in grouped.values()]
    shortlist.sort(key=lambda item: (shortlist_rank(item), item["category_name"], item["line_id"], item["current_form"]))
    return shortlist


def main() -> int:
    line_meta = load_line_meta()
    rows = load_tokens()
    known = build_known(rows)
    skeleton_index: dict[str, list[str]] = defaultdict(list)
    edit_index: dict[str, list[str]] = defaultdict(list)
    for key in known:
        skel = skeleton(key)
        skeleton_index[skel].append(key)
        for edited in edits1(skel):
            edit_index[edited].append(key)

    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        form = clean_surface(row.get("form", ""))
        if not form or " " in form or len(form) < 3:
            continue
        try:
            conf = float(row.get("confidence") or 0)
        except ValueError:
            conf = 0.0
        gloss = row.get("display_gloss") or row.get("gloss_en") or ""
        if conf >= 0.82 and gloss and gloss != "?":
            continue
        skel = skeleton(form)
        possible: list[tuple[str, str]] = []
        for candidate in skeleton_index.get(skel, []):
            if candidate != form:
                possible.append(("same after smoothing", candidate))
        if not possible and len(form) >= 4:
            for candidate in edit_index.get(skel, [])[:8]:
                if candidate != form:
                    possible.append(("one edit after smoothing", candidate))

        for match_type, candidate in possible[:3]:
            known_row = known[candidate]
            key = (row.get("lesson_id", ""), row.get("line_id", ""), form, candidate)
            if key in seen:
                continue
            seen.add(key)
            meta = line_meta.get((row.get("_file", ""), row.get("line_id", "")), {})
            candidate_conf = float(known_row.get("confidence") or 0)
            priority = "high" if match_type == "same after smoothing" else "medium"
            if priority == "medium" and candidate_conf < 0.92:
                continue
            rule = reason_for(form, candidate)
            if is_low_signal_vowel_variant(form, candidate, rule):
                continue
            candidates.append(
                {
                    "priority": priority,
                    "lesson_id": row.get("lesson_id", ""),
                    "line_id": row.get("line_id", ""),
                    "recording_type": meta.get("recording_type", ""),
                    "category_name": meta.get("category_name", ""),
                    "word_title": meta.get("word_title", ""),
                    "line_text": meta.get("text", ""),
                    "current_form": row.get("form", ""),
                    "suggested_form": candidate,
                    "rule": rule,
                    "match_type": match_type,
                    "current_gloss": gloss,
                    "suggested_gloss": known_row.get("display_gloss") or known_row.get("gloss_en") or "",
                    "suggested_lemma": known_row.get("lemma", ""),
                    "suggested_confidence": known_row.get("confidence", ""),
                    "suggested_source": known_row.get("source", ""),
                }
            )

    candidates.sort(key=lambda item: (0 if item["priority"] == "high" else 1, item["category_name"], item["line_id"], item["current_form"]))
    fields = [
        "priority",
        "lesson_id",
        "line_id",
        "recording_type",
        "category_name",
        "word_title",
        "line_text",
        "current_form",
        "suggested_form",
        "rule",
        "match_type",
        "current_gloss",
        "suggested_gloss",
        "suggested_lemma",
        "suggested_confidence",
        "suggested_source",
    ]
    with OUT_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(candidates)

    shortlist = build_shortlist(candidates)
    with OUT_SHORTLIST_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(shortlist)

    high = sum(1 for item in candidates if item["priority"] == "high")
    medium = sum(1 for item in candidates if item["priority"] == "medium")
    lines = [
        "# A.K. Refined Typo Candidate Pass - 2026-05-01",
        "",
        "Read-only report. No candidate in this file was applied automatically.",
        "",
        f"- Candidates: {len(candidates)}",
        f"- High priority: {high}",
        f"- Medium priority: {medium}",
        "",
        "The pass looks for low-confidence or unknown tokens that become known forms after limited smoothing: initial A.K. h- before vowel-initial stems, ê/êy alternation, tş/ç affricate spelling, and one edit after those normalizations. Pure diacritic/ı-i matches are now skipped because many are normal ezafe, oblique, plural, or vowel-raising forms rather than typos.",
        "",
        "## High Priority",
        "",
    ]
    for item in candidates:
        if item["priority"] != "high":
            continue
        lines.append(
            f"- `{item['current_form']}` -> `{item['suggested_form']}` "
            f"({item['rule']}); {item['category_name']} line {item['line_id']}: "
            f"{item['line_text']} [{item['word_title']}]"
        )
    lines.extend(["", "## Medium Priority", ""])
    for item in candidates:
        if item["priority"] != "medium":
            continue
        lines.append(
            f"- `{item['current_form']}` -> `{item['suggested_form']}` "
            f"({item['rule']}); {item['category_name']} line {item['line_id']}: "
            f"{item['line_text']} [{item['word_title']}]"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    shortlist_lines = [
        "# A.K. Refined Typo Candidate Shortlist - 2026-05-01",
        "",
        "Read-only shortlist of the highest-signal candidates from the full typo-candidate TSV. No candidate here was applied automatically.",
        "",
        f"- Shortlist candidates: {len(shortlist)}",
        "",
    ]
    for item in shortlist:
        shortlist_lines.append(
            f"- `{item['current_form']}` -> `{item['suggested_form']}` "
            f"({item['rule']}); {item['category_name']} line {item['line_id']}: "
            f"{item['line_text']} [{item['word_title']}]"
        )
    OUT_SHORTLIST_MD.write_text("\n".join(shortlist_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "candidates": len(candidates),
                "high": high,
                "medium": medium,
                "shortlist": len(shortlist),
                "tsv": str(OUT_TSV),
                "md": str(OUT_MD),
                "shortlist_tsv": str(OUT_SHORTLIST_TSV),
                "shortlist_md": str(OUT_SHORTLIST_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
