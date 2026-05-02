#!/usr/bin/env python3
"""First-pass Zazaki interlinearizer for approved LL Tools transcripts.

The model is intentionally simple and reviewable:
- approved seed analyses beat everything else
- exact dictionary matches beat approximate matches
- approximate matches keep confidence/source metadata
- unknowns are preserved for review rather than guessed silently
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICTIONARY = ROOT / "Dictionaries/tsv_output/lltools_import/lltools_dictionary_combined_public_reviewed_only.tsv"
DEFAULT_SEED = ROOT / "interlinearization/lexicon_seed.tsv"
DEFAULT_REVIEW_SEED = ROOT / "interlinearization/review_corrections.tsv"

WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['\u2019-])*", re.UNICODE)
PUNCT_RE = re.compile(r"\s+([,.;:!?])")

POS_BY_ENTRY_TYPE = {
    "noun": "N",
    "verb": "V",
    "word": "LEX",
    "suffix": "SUF",
    "prefix": "PFX",
    "adjective": "ADJ",
    "adverb": "ADV",
}

SUFFIX_RULES = [
    {"suffix": "iz", "gloss": "also", "pos": "CL", "label": "also suffix, equivalent to zi in other dialects", "confidence": 0.84},
    {"suffix": "z", "gloss": "also", "pos": "CL", "label": "also suffix, equivalent to zi in other dialects", "confidence": 0.82},
    {"suffix": "da", "gloss": "LOC", "pos": "CASE", "label": "locative suffix", "confidence": 0.84},
    {"suffix": "de", "gloss": "LOC", "pos": "CASE", "label": "locative suffix", "confidence": 0.8},
    {"suffix": "d", "gloss": "LOC", "pos": "CASE", "label": "reduced locative suffix", "confidence": 0.66},
    {"suffix": "t", "gloss": "LOC", "pos": "CASE", "label": "reduced locative suffix", "confidence": 0.64},
    {"suffix": "ûn", "gloss": "PL.OBL", "pos": "NUM", "label": "oblique plural suffix", "confidence": 0.82},
    {"suffix": "û", "gloss": "PL.OBL", "pos": "NUM", "label": "reduced oblique plural suffix", "confidence": 0.76},
    {"suffix": "ın", "gloss": "IPFV", "pos": "ASP", "label": "imperfective aspect suffix", "confidence": 0.8, "base_pos": {"V"}, "min_base_len": 1},
    {"suffix": "in", "gloss": "IPFV", "pos": "ASP", "label": "raised imperfective aspect suffix", "confidence": 0.78, "base_pos": {"V"}, "min_base_len": 1},
    {"suffix": "en", "gloss": "IPFV", "pos": "ASP", "label": "imperfective aspect suffix", "confidence": 0.8, "base_pos": {"V"}},
    {"suffix": "n", "gloss": "IPFV", "pos": "ASP", "label": "imperfective aspect suffix", "confidence": 0.8, "base_pos": {"V"}},
    {"suffix": "a", "gloss": "F.SG", "pos": "AGR", "label": "feminine singular suffix", "confidence": 0.82},
    {"suffix": "wo", "gloss": "and", "pos": "CONJ", "label": "w-buffered cliticized conjunction, from o/û", "confidence": 0.78, "min_base_len": 1},
    {"suffix": "o", "gloss": "M.SG.COP", "pos": "COP", "label": "masculine singular copula", "confidence": 0.72},
    {"suffix": "ê", "gloss": "PL/OBL/EZ", "pos": "AGR", "label": "plural, oblique, or absorbed ezafe/raised final vowel", "confidence": 0.74},
    {"suffix": "i", "gloss": "RAISED/OBL", "pos": "CASE", "label": "raised final ı or oblique ending", "confidence": 0.66},
    {"suffix": "ı", "gloss": "OBL", "pos": "CASE", "label": "oblique ending", "confidence": 0.66},
]

PREFIX_RULES = [
    {"prefix": "nı", "gloss": "NEG", "pos": "POL", "label": "negative prefix", "confidence": 0.82},
    {"prefix": "nu", "gloss": "NEG", "pos": "POL", "label": "negative prefix", "confidence": 0.8},
    {"prefix": "nê", "gloss": "NEG", "pos": "POL", "label": "negative prefix", "confidence": 0.8},
    {"prefix": "ne", "gloss": "NEG", "pos": "POL", "label": "negative prefix", "confidence": 0.8},
    {"prefix": "n", "gloss": "NEG", "pos": "POL", "label": "reduced negative prefix", "confidence": 0.68},
    {"prefix": "bı", "gloss": "SBJV", "pos": "MOOD", "label": "subjunctive prefix", "confidence": 0.82},
    {"prefix": "b", "gloss": "SBJV", "pos": "MOOD", "label": "reduced subjunctive prefix", "confidence": 0.68},
]


@dataclass(slots=True)
class Analysis:
    form: str
    lemma: str = ""
    gloss_en: str = ""
    gloss_tr: str = ""
    pos: str = ""
    confidence: float = 0.0
    source: str = "unknown"
    match_type: str = "unknown"
    entry_id: str = ""
    corrected_form: str = ""
    morphemes: list[dict[str, str]] = field(default_factory=list)
    review_note: str = ""

    def display_gloss(self) -> str:
        return self.gloss_en or "?"

    def as_dict(self) -> dict[str, object]:
        return {
            "form": self.form,
            "lemma": self.lemma,
            "gloss_en": self.gloss_en,
            "gloss_tr": self.gloss_tr,
            "display_gloss": self.display_gloss(),
            "pos": self.pos,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "match_type": self.match_type,
            "entry_id": self.entry_id,
            "corrected_form": self.corrected_form,
            "morphemes": self.morphemes,
            "review_note": self.review_note,
        }


def html_unescape(value: str) -> str:
    return html.unescape(value or "").replace("\u2019", "'").strip()


def normalize_key(value: str) -> str:
    value = html_unescape(value).lower()
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u0307", "")
    value = re.sub(r"^[\"'`]+|[\"'`]+$", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def fold_key(value: str) -> str:
    value = normalize_key(value)
    value = value.replace("ı", "i")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value


def concise_definition(value: str, max_len: int = 46) -> str:
    value = html_unescape(value)
    value = re.sub(r"\s+", " ", value)
    value = value.split("|", 1)[0].split(";", 1)[0].strip()
    value = re.sub(r"\s*[,;:]?\s*(örn\.?|örnek|example).*$", "", value, flags=re.I)
    if len(value) > max_len:
        value = value[: max_len - 1].rstrip(" ,.;:") + "..."
    return value


def is_single_token_entry(entry: str) -> bool:
    entry = normalize_key(entry)
    return bool(entry) and not any(ch.isspace() for ch in entry) and "," not in entry


def candidate_rank(row: dict[str, str]) -> tuple[int, int, int]:
    entry_type = normalize_key(row.get("entry_type", ""))
    source_tag = normalize_key(row.get("source_tag", ""))
    review = normalize_key(row.get("definition_en_review_status", ""))
    type_score = {"word": 5, "noun": 5, "verb": 4, "suffix": 2, "prefix": 2}.get(entry_type, 3)
    source_score = 2 if "harun" in source_tag else 1
    review_score = 2 if "needs_review" not in review else 0
    return (type_score, source_score, review_score)


def row_to_analysis(form: str, row: dict[str, str], confidence: float, match_type: str) -> Analysis:
    entry = html_unescape(row.get("entry_display") or row.get("entry") or "")
    entry_type = normalize_key(row.get("entry_type", ""))
    return Analysis(
        form=form,
        lemma=entry,
        gloss_en=concise_definition(row.get("definition_en", "")),
        gloss_tr=concise_definition(row.get("definition_tr", "")),
        pos=POS_BY_ENTRY_TYPE.get(entry_type, entry_type.upper()[:6] if entry_type else ""),
        confidence=confidence,
        source=row.get("source_tag") or row.get("source_dictionary") or "dictionary",
        match_type=match_type,
        entry_id=row.get("entry_id", ""),
    )


def load_dictionary(path: Path) -> tuple[dict[str, Analysis], dict[str, list[Analysis]]]:
    exact: dict[str, Analysis] = {}
    folded: dict[str, list[Analysis]] = {}
    rows_by_key: dict[str, list[dict[str, str]]] = {}

    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entry = html_unescape(row.get("entry_display") or row.get("entry") or "")
            if not is_single_token_entry(entry):
                continue
            key = normalize_key(entry)
            rows_by_key.setdefault(key, []).append(row)

    for key, rows in rows_by_key.items():
        best = sorted(rows, key=candidate_rank, reverse=True)[0]
        analysis = row_to_analysis(key, best, 0.92, "dictionary_exact")
        exact[key] = analysis
        folded.setdefault(fold_key(key), []).append(analysis)

    return exact, folded


def seed_row_to_analysis(row: dict[str, str]) -> Analysis | None:
    form = html_unescape(row.get("form", ""))
    if not form:
        return None
    confidence = float(row.get("confidence") or 0.97)
    analysis = Analysis(
        form=form,
        lemma=html_unescape(row.get("lemma", "")),
        gloss_en=html_unescape(row.get("gloss_en", "")),
        gloss_tr=html_unescape(row.get("gloss_tr", "")),
        pos=html_unescape(row.get("pos", "")),
        confidence=confidence,
        source=row.get("source") or "seed",
        match_type="seed_memory",
        corrected_form=html_unescape(row.get("corrected_form", "")),
        review_note=row.get("note", ""),
    )
    morphemes = parse_morphemes(row.get("morphemes", ""))
    if morphemes:
        analysis.morphemes = morphemes
    return analysis


def load_seed(path: Path) -> dict[str, Analysis]:
    seed: dict[str, Analysis] = {}
    if not path.exists():
        return seed
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            status = normalize_key(row.get("status", ""))
            if status and status not in {"approved", "seed"}:
                continue
            if normalize_key(row.get("scope", "")) in {"line", "lesson"}:
                continue
            analysis = seed_row_to_analysis(row)
            if not analysis:
                continue
            seed[normalize_key(analysis.form)] = analysis
    return seed


def csv_contains(value: str, needle: str) -> bool:
    return needle in {part.strip() for part in str(value).split(",") if part.strip()}


def load_context_overrides(path: Path, lesson_id: str) -> dict[str, list[tuple[set[str], Analysis]]]:
    overrides: dict[str, list[tuple[set[str], Analysis]]] = {}
    if not path.exists():
        return overrides
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            status = normalize_key(row.get("status", ""))
            if status and status not in {"approved", "seed"}:
                continue
            scope = normalize_key(row.get("scope", ""))
            if scope not in {"line", "lesson"}:
                continue
            if not csv_contains(row.get("lesson_ids", ""), lesson_id):
                continue
            analysis = seed_row_to_analysis(row)
            if not analysis:
                continue
            line_ids = {part.strip() for part in row.get("line_ids", "").split(",") if part.strip()}
            overrides.setdefault(normalize_key(analysis.form), []).append((line_ids, analysis))
    return overrides


def parse_morphemes(value: str) -> list[dict[str, str]]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    morphemes: list[dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        form = html_unescape(str(item.get("form", "")))
        gloss = html_unescape(str(item.get("gloss", "")))
        pos = html_unescape(str(item.get("pos", "")))
        if form or gloss or pos:
            morphemes.append({"form": form, "gloss": gloss, "pos": pos})
    return morphemes


def build_folded_index(index: dict[str, Analysis]) -> dict[str, list[Analysis]]:
    folded: dict[str, list[Analysis]] = {}
    for key, analysis in index.items():
        if " " in key:
            continue
        folded.setdefault(fold_key(key), []).append(analysis)
    return folded


def clone_for_form(analysis: Analysis, form: str, confidence: float | None = None, match_type: str | None = None) -> Analysis:
    return Analysis(
        form=form,
        lemma=analysis.lemma,
        gloss_en=analysis.gloss_en,
        gloss_tr=analysis.gloss_tr,
        pos=analysis.pos,
        confidence=analysis.confidence if confidence is None else confidence,
        source=analysis.source,
        match_type=analysis.match_type if match_type is None else match_type,
        entry_id=analysis.entry_id,
        corrected_form=analysis.corrected_form,
        morphemes=project_morphemes_to_form([dict(m) for m in analysis.morphemes], form),
        review_note=analysis.review_note,
    )


def project_morphemes_to_form(morphemes: list[dict[str, str]], form: str) -> list[dict[str, str]]:
    if not morphemes or " " in normalize_key(form):
        return morphemes
    joined = "".join(str(m.get("form", "")) for m in morphemes)
    if not joined or normalize_key(joined) == normalize_key(form):
        return morphemes
    if fold_key(joined) != fold_key(form) or len(joined) != len(form):
        return morphemes
    projected: list[dict[str, str]] = []
    cursor = 0
    for morpheme in morphemes:
        length = len(str(morpheme.get("form", "")))
        next_morpheme = dict(morpheme)
        next_morpheme["form"] = form[cursor : cursor + length]
        projected.append(next_morpheme)
        cursor += length
    return projected if cursor == len(form) else morphemes


def lower_raised_final(key: str) -> str | None:
    if key.endswith("â"):
        return key[:-1] + "a"
    if key.endswith("ê"):
        return key[:-1] + "e"
    if key.endswith("i"):
        return key[:-1] + "ı"
    return None


def limited_variants(values: Iterable[str], limit: int = 80) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
            if len(out) >= limit:
                break
    return out


def orthography_variants(key: str) -> list[str]:
    variants: set[str] = set()
    lowered = lower_raised_final(key)
    if lowered:
        variants.add(lowered)

    dental_pairs = [
        ("tp", ("tw", "tu")),
        ("db", ("dw", "du")),
        ("tw", ("tp", "tu")),
        ("dw", ("db", "du")),
        ("tu", ("tp", "tw")),
        ("du", ("db", "dw")),
    ]
    for source, replacements in dental_pairs:
        if source in key:
            for replacement in replacements:
                variants.add(key.replace(source, replacement))

    vowel_sequences = ["we", "wa", "wı"]
    vowel_replacements = ["o", "ö", "û", "ü", "u"]
    current = {key}
    for sequence in vowel_sequences:
        next_values = set(current)
        for value in current:
            if sequence not in value:
                continue
            for replacement in vowel_replacements:
                next_values.add(value.replace(sequence, replacement, 1))
        current = set(limited_variants(next_values))
    variants.update(current - {key})

    return limited_variants(sorted(variants))


def lookup_known(
    form: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
    *,
    allow_variants: bool = True,
) -> Analysis | None:
    key = normalize_key(form)
    if not key or " " in key:
        return None
    if key in seed:
        return clone_for_form(seed[key], form)
    if key in exact:
        return clone_for_form(exact[key], form)

    seed_match = choose_unique_folded(seed_folded.get(fold_key(key), []))
    if seed_match:
        return clone_for_form(seed_match, form, min(seed_match.confidence, 0.88), "seed_folded_variant")

    dictionary_match = choose_unique_folded(folded.get(fold_key(key), []))
    if dictionary_match:
        return clone_for_form(dictionary_match, form, 0.78, "dictionary_folded")

    if allow_variants:
        for variant in orthography_variants(key):
            variant_match = lookup_known(variant, seed, seed_folded, exact, folded, allow_variants=False)
            if variant_match:
                variant_match.form = form
                variant_match.confidence = min(variant_match.confidence, 0.72)
                variant_match.match_type = "orthography_variant"
                variant_match.review_note = append_note(
                    variant_match.review_note,
                    f"Matched by dialect/orthography variant {form} -> {variant}.",
                )
                return variant_match
    return None


def append_note(existing: str, addition: str) -> str:
    existing = existing.strip()
    return f"{existing} {addition}".strip() if existing else addition


def choose_unique_folded(matches: list[Analysis]) -> Analysis | None:
    if not matches:
        return None
    lemmas = {normalize_key(m.lemma) for m in matches}
    glosses = {normalize_key(m.display_gloss()) for m in matches}
    if len(lemmas) == 1 or len(glosses) == 1:
        return matches[0]
    return None


def format_morphemes(token: dict[str, object]) -> str:
    morphemes = token.get("morphemes") or []
    if not morphemes:
        return ""
    return "-".join(str(m.get("form", "")) for m in morphemes if m.get("form"))


def display_form(token: dict[str, object]) -> str:
    return str(token.get("corrected_form") or token.get("form") or "")


def word_display_form(token: dict[str, object]) -> str:
    return f"{token.get('prefix_punct', '')}{display_form(token)}{token.get('suffix_punct', '')}"


def format_morpheme_glosses(token: dict[str, object]) -> str:
    morphemes = token.get("morphemes") or []
    if not morphemes:
        return ""
    return "-".join(str(m.get("gloss", "")) for m in morphemes if m.get("gloss"))


def make_suffix_analysis(token: str, base_analysis: Analysis, suffix_form: str, rule: dict[str, object], confidence: float | None = None) -> Analysis:
    base_form = token[: len(token) - len(suffix_form)]
    analysis = clone_for_form(base_analysis, token, confidence or min(base_analysis.confidence, float(rule["confidence"])), "suffix_morphology")
    suffix_gloss = str(rule["gloss"])
    suffix_pos = str(rule["pos"])
    suffix_label = str(rule["label"])
    if suffix_form == "o" and base_analysis.pos == "V":
        suffix_gloss = "1SG"
        suffix_pos = "AGR"
        suffix_label = "A.K. verbal -o ending; not masculine here"
    analysis.gloss_en = f"{base_analysis.display_gloss()}-{suffix_gloss}"
    analysis.gloss_tr = base_analysis.gloss_tr
    analysis.source = base_analysis.source
    if base_analysis.morphemes:
        analysis.morphemes = [dict(m) for m in base_analysis.morphemes]
    else:
        analysis.morphemes = [{"form": base_form, "gloss": base_analysis.display_gloss(), "pos": base_analysis.pos}]
    analysis.morphemes.append({"form": suffix_form, "gloss": suffix_gloss, "pos": suffix_pos})
    analysis.review_note = append_note(analysis.review_note, suffix_label)
    return analysis


def morph_surface(analysis: Analysis, token: str) -> str:
    return analysis.corrected_form or token


def use_original_surface(analysis: Analysis, original_form: str) -> Analysis:
    if analysis.form != original_form:
        analysis.corrected_form = analysis.corrected_form or analysis.form
        analysis.form = original_form
    return analysis


def try_suffix(
    token: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> Analysis | None:
    key = normalize_key(token)
    folded_token = fold_key(token)
    for rule in SUFFIX_RULES:
        suffix = str(rule["suffix"])
        min_base_len = int(rule.get("min_base_len", 3))
        if not key.endswith(suffix) or len(key) - len(suffix) < min_base_len:
            continue
        base = key[: -len(suffix)]
        base_analysis = lookup_known(base, seed, seed_folded, exact, folded)
        if not base_analysis:
            folded_base = folded_token[: -len(fold_key(suffix))]
            base_analysis = lookup_known(folded_base, seed, seed_folded, exact, folded)
        if not base_analysis:
            base_analysis = try_suffix(base, seed, seed_folded, exact, folded)
        if not base_analysis:
            base_analysis = try_prefix(base, seed, seed_folded, exact, folded)
        if base_analysis:
            if not base_analysis.morphemes:
                suffix_base = try_suffix(base, seed, seed_folded, exact, folded)
                if suffix_base and suffix_base.lemma == base_analysis.lemma:
                    base_analysis = suffix_base
                else:
                    prefix_base = try_prefix(base, seed, seed_folded, exact, folded)
                    if prefix_base and prefix_base.lemma == base_analysis.lemma:
                        base_analysis = prefix_base
            allowed_pos = rule.get("base_pos")
            if allowed_pos and base_analysis.pos not in allowed_pos:
                continue
            return make_suffix_analysis(token, base_analysis, token[len(token) - len(suffix) :], rule)
    return None


def make_prefix_analysis(token: str, base_analysis: Analysis, prefix_form: str, rule: dict[str, object]) -> Analysis:
    base_form = token[len(prefix_form) :]
    analysis = clone_for_form(base_analysis, token, min(base_analysis.confidence, float(rule["confidence"])), "prefix_morphology")
    prefix_gloss = str(rule["gloss"])
    analysis.gloss_en = f"{prefix_gloss}-{base_analysis.display_gloss()}"
    base_morphemes = [dict(m) for m in base_analysis.morphemes]
    analysis.morphemes = [{"form": prefix_form, "gloss": prefix_gloss, "pos": str(rule["pos"])}]
    if base_morphemes:
        analysis.morphemes.extend(base_morphemes)
    else:
        analysis.morphemes.append({"form": base_form, "gloss": base_analysis.display_gloss(), "pos": base_analysis.pos})
    analysis.review_note = append_note(analysis.review_note, str(rule["label"]))
    return analysis


def try_prefix(
    token: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> Analysis | None:
    key = normalize_key(token)
    for rule in PREFIX_RULES:
        prefix = str(rule["prefix"])
        if not key.startswith(prefix) or len(key) <= len(prefix) + 2:
            continue
        base = key[len(prefix) :]
        base_analysis = lookup_known(base, seed, seed_folded, exact, folded)
        if not base_analysis:
            base_analysis = try_suffix(base, seed, seed_folded, exact, folded)
        if base_analysis:
            if not base_analysis.morphemes:
                suffix_base = try_suffix(base, seed, seed_folded, exact, folded)
                if suffix_base and suffix_base.lemma == base_analysis.lemma:
                    base_analysis = suffix_base
            return make_prefix_analysis(token, base_analysis, token[: len(prefix)], rule)
    return None


def enrich_exact_morphology(
    analysis: Analysis,
    token: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> Analysis:
    if analysis.morphemes:
        return analysis

    surface = morph_surface(analysis, token)

    suffix_match = try_suffix(surface, seed, seed_folded, exact, folded)
    if suffix_match and suffix_match.lemma == analysis.lemma:
        suffix_match = use_original_surface(suffix_match, token)
        suffix_match.corrected_form = analysis.corrected_form
        suffix_match.source = analysis.source
        suffix_match.confidence = max(suffix_match.confidence, min(analysis.confidence, 0.9))
        suffix_match.match_type = analysis.match_type
        suffix_match.review_note = append_note(suffix_match.review_note, analysis.review_note)
        return suffix_match

    prefix_match = try_prefix(surface, seed, seed_folded, exact, folded)
    if prefix_match and prefix_match.lemma == analysis.lemma:
        prefix_match = use_original_surface(prefix_match, token)
        prefix_match.corrected_form = analysis.corrected_form
        prefix_match.source = analysis.source
        prefix_match.confidence = max(prefix_match.confidence, min(analysis.confidence, 0.88))
        prefix_match.match_type = analysis.match_type
        prefix_match.review_note = append_note(prefix_match.review_note, analysis.review_note)
        return prefix_match

    normalized = normalize_key(surface)
    if analysis.pos == "V" and normalized.endswith("n") and len(normalized) > 3:
        stem = surface[:-1]
        analysis.morphemes = [
            {"form": stem, "gloss": analysis.display_gloss(), "pos": analysis.pos},
            {"form": surface[-1], "gloss": "IPFV", "pos": "ASP"},
        ]
        analysis.review_note = append_note(analysis.review_note, "Final -n interpreted as imperfective aspect.")

    if not analysis.morphemes and len(normalized) > 3 and analysis.pos not in {"POST", "PART", "INTJ", "PRON", "DET", "CONJ"}:
        lowered = lower_raised_final(normalized)
        if lowered:
            base_analysis = lookup_known(lowered, seed, seed_folded, exact, folded, allow_variants=False)
            if base_analysis and base_analysis.lemma == analysis.lemma:
                analysis.morphemes = [
                    {"form": surface[:-1] + lowered[-1], "gloss": base_analysis.display_gloss(), "pos": base_analysis.pos},
                    {"form": surface[-1], "gloss": "RAISE/EZ", "pos": "PHON"},
                ]
                analysis.review_note = append_note(analysis.review_note, "Final vowel raising may mark plural/ezafe absorption.")

    if not analysis.morphemes and analysis.pos == "V":
        for rule in PREFIX_RULES:
            prefix = str(rule["prefix"])
            if prefix == "b":
                continue
            if normalized.startswith(prefix) and len(normalized) > len(prefix) + 2:
                prefix_form = surface[: len(prefix)]
                base_form = surface[len(prefix) :]
                prefix_gloss = str(rule["gloss"])
                analysis.morphemes = [
                    {"form": prefix_form, "gloss": prefix_gloss, "pos": str(rule["pos"])},
                    {"form": base_form, "gloss": analysis.display_gloss(), "pos": analysis.pos},
                ]
                analysis.review_note = append_note(analysis.review_note, str(rule["label"]))
                break
    return analysis


def analyze_token(
    token: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> Analysis:
    direct_match = lookup_known(token, seed, seed_folded, exact, folded, allow_variants=False)
    if direct_match:
        return enrich_exact_morphology(direct_match, token, seed, seed_folded, exact, folded)

    suffix_match = try_suffix(token, seed, seed_folded, exact, folded)
    if suffix_match:
        return suffix_match

    prefix_match = try_prefix(token, seed, seed_folded, exact, folded)
    if prefix_match:
        return prefix_match

    variant_match = lookup_known(token, seed, seed_folded, exact, folded, allow_variants=True)
    if variant_match:
        return enrich_exact_morphology(variant_match, token, seed, seed_folded, exact, folded)

    return Analysis(form=token, confidence=0.0, review_note="No seed or dictionary match.")


def context_override_for(
    token: str,
    line_id: str,
    context_overrides: dict[str, list[tuple[set[str], Analysis]]] | None,
) -> Analysis | None:
    if not context_overrides:
        return None
    for line_ids, analysis in context_overrides.get(normalize_key(token), []):
        if line_ids and line_id not in line_ids:
            continue
        return clone_for_form(analysis, token, match_type="context_review_seed")
    return None


def tokenize_line(text: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(html_unescape(text))]


def split_interword_punctuation(segment: str) -> tuple[str, str]:
    if not segment:
        return "", ""
    if not re.search(r"\s", segment):
        return re.sub(r"\s+", "", segment), ""

    first_space = re.search(r"\s", segment)
    last_space = None
    for match in re.finditer(r"\s+", segment):
        last_space = match
    if not first_space or not last_space:
        return re.sub(r"\s+", "", segment), ""

    before_space = re.sub(r"\s+", "", segment[: first_space.start()])
    after_space = re.sub(r"\s+", "", segment[last_space.end() :])
    between = re.sub(r"\s+", "", segment[first_space.end() : last_space.start()])
    if between and not before_space and not after_space:
        before_space = between
    elif between:
        before_space += between
    return before_space, after_space


def token_punctuation(text: str) -> list[dict[str, str]]:
    text = html_unescape(text)
    matches = list(WORD_RE.finditer(text))
    if not matches:
        return []

    parts = [{"prefix": "", "suffix": ""} for _ in matches]
    leading = re.sub(r"\s+", "", text[: matches[0].start()])
    if leading:
        parts[0]["prefix"] = leading

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        left, right = split_interword_punctuation(text[match.end() : end])
        parts[index]["suffix"] += left
        if right and index + 1 < len(matches):
            parts[index + 1]["prefix"] += right
        elif right:
            parts[index]["suffix"] += right
    return parts


def load_phrase_seed(seed: dict[str, Analysis]) -> dict[tuple[str, ...], Analysis]:
    phrases: dict[tuple[str, ...], Analysis] = {}
    for key, analysis in seed.items():
        parts = tuple(part for part in key.split(" ") if part)
        if len(parts) > 1:
            phrases[parts] = analysis
    return phrases


def token_key_options(
    token: str,
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> set[str]:
    options = {normalize_key(token)}
    suffix_match = try_suffix(token, seed, seed_folded, exact, folded)
    if suffix_match and suffix_match.morphemes:
        base_form = str(suffix_match.morphemes[0].get("form", ""))
        if base_form:
            options.add(normalize_key(base_form))
            lowered = lower_raised_final(normalize_key(base_form))
            if lowered:
                options.add(lowered)
    direct_match = lookup_known(token, seed, seed_folded, exact, folded)
    if direct_match and direct_match.lemma and " " not in direct_match.lemma:
        options.add(normalize_key(direct_match.lemma))
    options.update(orthography_variants(normalize_key(token)))
    return {option for option in options if option}


def phrase_matches(
    tokens: list[str],
    phrases: dict[tuple[str, ...], Analysis],
    seed: dict[str, Analysis],
    seed_folded: dict[str, list[Analysis]],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    if not tokens or not phrases:
        return matches

    normalized_options = [token_key_options(token, seed, seed_folded, exact, folded) for token in tokens]
    phrase_lengths = sorted({len(parts) for parts in phrases}, reverse=True)
    occupied: set[int] = set()

    for start in range(len(normalized_options)):
        if start in occupied:
            continue
        for length in phrase_lengths:
            end = start + length
            if end > len(normalized_options) or any(index in occupied for index in range(start, end)):
                continue
            analysis = None
            matched_key: tuple[str, ...] | None = None
            for phrase_key, candidate in phrases.items():
                if len(phrase_key) != length:
                    continue
                if all(phrase_key[offset] in normalized_options[start + offset] for offset in range(length)):
                    analysis = candidate
                    matched_key = phrase_key
                    break
            if not analysis:
                continue
            for index in range(start, end):
                occupied.add(index)
            match = analysis.as_dict()
            match.update(
                {
                    "start_token": start + 1,
                    "end_token": end,
                    "forms": tokens[start:end],
                    "phrase_key": " ".join(matched_key or ()),
                    "match_type": "phrase_memory",
                }
            )
            matches.append(match)
            break
    return matches


def read_alignment_tsv(path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    rows: list[dict[str, object]] = []
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = html_unescape(row.get("text_full") or row.get("text") or "")
            if not text:
                continue
            rows.append(
                {
                    "id": row.get("id") or str(len(rows) + 1),
                    "text": text,
                    "start_sec": row.get("start_sec", ""),
                    "end_sec": row.get("end_sec", ""),
                    "audio_windows": row.get("audio_windows", ""),
                    "audio_posix": row.get("audio_posix", ""),
                }
            )
            if row.get("audio_windows"):
                metadata["audio_windows"] = row["audio_windows"]
            if row.get("audio_posix"):
                metadata["audio_posix"] = row["audio_posix"]
    return rows, metadata


def read_plain_text(path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    chunks = [part.strip() for part in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")) if part.strip()]
    return [{"id": str(i + 1), "text": chunk, "start_sec": "", "end_sec": ""} for i, chunk in enumerate(chunks)], {}


def interlinearize(
    rows: list[dict[str, object]],
    seed: dict[str, Analysis],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
    context_overrides: dict[str, list[tuple[set[str], Analysis]]] | None = None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seed_folded = build_folded_index(seed)
    phrases = load_phrase_seed(seed)
    for row in rows:
        tokens = tokenize_line(str(row["text"]))
        line_id = str(row.get("id", ""))
        analyses = [
            context_override_for(token, line_id, context_overrides)
            or analyze_token(token, seed, seed_folded, exact, folded)
            for token in tokens
        ]
        punctuation = token_punctuation(str(row["text"]))
        token_dicts = [a.as_dict() for a in analyses]
        for token, punct in zip(token_dicts, punctuation):
            token["prefix_punct"] = punct["prefix"]
            token["suffix_punct"] = punct["suffix"]
        scores = [a.confidence for a in analyses]
        line = dict(row)
        line["tokens"] = token_dicts
        line["phrase_matches"] = phrase_matches(tokens, phrases, seed, seed_folded, exact, folded)
        line["token_count"] = len(tokens)
        line["matched_count"] = sum(1 for a in analyses if a.confidence > 0)
        line["high_confidence_count"] = sum(1 for a in analyses if a.confidence >= 0.84)
        line["mean_confidence"] = round(statistics.mean(scores), 3) if scores else 0.0
        result.append(line)
    return result


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_tokens_tsv(path: Path, lesson_id: str, lines: Iterable[dict[str, object]]) -> None:
    fields = [
        "lesson_id",
        "line_id",
        "token_index",
        "form",
        "lemma",
        "display_gloss",
        "gloss_en",
        "gloss_tr",
        "pos",
        "confidence",
        "match_type",
        "source",
        "entry_id",
        "corrected_form",
        "prefix_punct",
        "suffix_punct",
        "morphemes",
        "review_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for line in lines:
            for idx, token in enumerate(line["tokens"], start=1):
                row = {field: token.get(field, "") for field in fields}
                row["morphemes"] = json.dumps(token.get("morphemes") or [], ensure_ascii=False)
                row.update({"lesson_id": lesson_id, "line_id": line["id"], "token_index": idx})
                writer.writerow(row)


def write_review_tsv(path: Path, lesson_id: str, lines: Iterable[dict[str, object]]) -> None:
    counts: dict[str, dict[str, object]] = {}
    for line in lines:
        for token in line["tokens"]:
            if token["confidence"] >= 0.84:
                continue
            key = normalize_key(str(token["form"]))
            rec = counts.setdefault(
                key,
                {
                    "form": token["form"],
                    "count": 0,
                    "best_guess_lemma": token["lemma"],
                    "best_guess_gloss": token["display_gloss"],
                    "match_type": token["match_type"],
                    "review_note": token["review_note"],
                    "lines": [],
                },
            )
            rec["count"] = int(rec["count"]) + 1
            rec["lines"].append(str(line["id"]))
    fields = ["lesson_id", "form", "count", "best_guess_lemma", "best_guess_gloss", "match_type", "review_note", "lines"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for rec in sorted(counts.values(), key=lambda item: (-int(item["count"]), normalize_key(str(item["form"])))):
            writer.writerow({**rec, "lesson_id": lesson_id, "lines": ",".join(rec["lines"])})


def pad_cells(values: list[str]) -> str:
    if not values:
        return ""
    widths = [max(3, min(24, len(v))) for v in values]
    return "  ".join(v.ljust(widths[i]) for i, v in enumerate(values))


def phrase_row_values(tokens: list[dict[str, object]], phrase_matches: list[dict[str, object]]) -> list[str]:
    values = [""] * len(tokens)
    occupied: set[int] = set()
    for phrase in sorted(phrase_matches, key=lambda item: int(item.get("start_token") or 0)):
        try:
            start = int(phrase.get("start_token") or 0) - 1
            end = int(phrase.get("end_token") or 0)
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or start >= len(tokens):
            continue
        end = min(end, len(tokens))
        span = set(range(start, end))
        if occupied & span:
            continue
        lemma = str(phrase.get("lemma") or "").strip()
        gloss = str(phrase.get("display_gloss") or phrase.get("gloss_en") or "?").strip()
        label = f"{lemma} = {gloss}" if lemma and lemma != "?" else gloss
        values[start] = f"{values[start]}; {label}" if values[start] else label
        occupied.update(span)
    return values


def interlinear_rows(
    tokens: list[dict[str, object]],
    phrase_matches: list[dict[str, object]] | None = None,
) -> list[tuple[str, list[str]]]:
    rows = [
        ("WORD", [word_display_form(t) for t in tokens]),
        ("MORPH", [format_morphemes(t) or display_form(t) for t in tokens]),
        ("LEMMA", [str(t["lemma"] or "?") for t in tokens]),
        ("GLOSS", [format_morpheme_glosses(t) or str(t["display_gloss"] or "?") for t in tokens]),
        ("POS", [str(t["pos"] or "?") for t in tokens]),
        ("CONF", [str(t["confidence"]) for t in tokens]),
    ]
    phrase_values = phrase_row_values(tokens, phrase_matches or [])
    if any(phrase_values):
        rows.append(("PHRASE", phrase_values))
    return rows


def format_interlinear_rows(
    tokens: list[dict[str, object]],
    phrase_matches: list[dict[str, object]] | None = None,
) -> list[str]:
    rows = interlinear_rows(tokens, phrase_matches)
    if not rows:
        return []
    label_width = max(len(label) for label, _ in rows)
    token_count = max(len(values) for _, values in rows)
    widths = [
        max(3, *(len(values[index]) for _, values in rows if index < len(values)))
        for index in range(token_count)
    ]
    out: list[str] = []
    for label, values in rows:
        cells = [values[index] if index < len(values) else "" for index in range(token_count)]
        out.append(f"{label.ljust(label_width)}  " + "  ".join(cells[index].ljust(widths[index]) for index in range(token_count)).rstrip())
    return out


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = payload["lines"]
    summary = payload["summary"]
    out = [
        f"# {payload['title']}",
        "",
        f"- Lesson ID: `{payload['lesson_id']}`",
        f"- Tokens: {summary['tokens']}",
        f"- Matched tokens: {summary['matched_tokens']} ({summary['matched_pct']})",
        f"- High-confidence tokens: {summary['high_confidence_tokens']} ({summary['high_confidence_pct']})",
        f"- Mean confidence: {summary['mean_confidence']}",
        "",
    ]
    for line in lines:
        tokens = line["tokens"]
        out.append(f"## Line {line['id']}")
        if line.get("start_sec") and line.get("end_sec"):
            out.append(f"`{line['start_sec']}s-{line['end_sec']}s`")
        out.append("")
        out.append(str(line["text"]))
        out.append("")
        out.append("```text")
        out.extend(format_interlinear_rows(tokens, line.get("phrase_matches", [])))
        out.append("```")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def write_flex_sfm(path: Path, payload: dict[str, object]) -> None:
    out = [r"\_sh v3.0 520 Text", f"\\id {payload['title']}"]
    for line in payload["lines"]:
        tokens = line["tokens"]
        ref = f"{payload['lesson_id']}.{str(line['id']).zfill(3)}"
        out.append(f"\\ref {ref}")
        out.append("\\tx " + PUNCT_RE.sub(r"\1", str(line["text"])))
        out.append("\\mb " + " ".join((format_morphemes(t) or display_form(t)).replace(" ", ".") for t in tokens))
        out.append("\\ge " + " ".join((format_morpheme_glosses(t) or str(t["display_gloss"])).replace(" ", ".") for t in tokens))
        out.append("\\ps " + " ".join(str(t["pos"] or "UNK") for t in tokens))
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def summarize(lines: list[dict[str, object]]) -> dict[str, object]:
    total = sum(int(line["token_count"]) for line in lines)
    matched = sum(int(line["matched_count"]) for line in lines)
    high = sum(int(line["high_confidence_count"]) for line in lines)
    scores = [float(token["confidence"]) for line in lines for token in line["tokens"]]
    return {
        "lines": len(lines),
        "tokens": total,
        "matched_tokens": matched,
        "matched_pct": f"{(matched / total):.1%}" if total else "0.0%",
        "high_confidence_tokens": high,
        "high_confidence_pct": f"{(high / total):.1%}" if total else "0.0%",
        "mean_confidence": round(statistics.mean(scores), 3) if scores else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build first-pass Zazaki interlinear outputs.")
    parser.add_argument("--alignment-tsv", type=Path, help="LL Tools line_alignment.tsv-style input.")
    parser.add_argument("--text", type=Path, help="Plain UTF-8 text input split on blank lines.")
    parser.add_argument("--lesson-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--review-seed", type=Path, default=DEFAULT_REVIEW_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "interlinearization/outputs")
    args = parser.parse_args()

    if bool(args.alignment_tsv) == bool(args.text):
        parser.error("Provide exactly one of --alignment-tsv or --text.")

    seed = load_seed(args.seed)
    seed.update(load_seed(args.review_seed))
    context_overrides = load_context_overrides(args.review_seed, args.lesson_id)
    exact, folded = load_dictionary(args.dictionary)
    rows, metadata = read_alignment_tsv(args.alignment_tsv) if args.alignment_tsv else read_plain_text(args.text)
    lines = interlinearize(rows, seed, exact, folded, context_overrides)

    payload = {
        "schema": "zazaki_interlinear.v1",
        "lesson_id": args.lesson_id,
        "title": args.title,
        "source": str(args.alignment_tsv or args.text),
        "dictionary": str(args.dictionary),
        "seed": str(args.seed),
        "metadata": metadata,
        "summary": summarize(lines),
        "lines": lines,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / args.lesson_id
    write_json(base.with_suffix(".interlinear.json"), payload)
    write_markdown(base.with_suffix(".interlinear.md"), payload)
    write_tokens_tsv(base.with_suffix(".tokens.tsv"), args.lesson_id, lines)
    write_review_tsv(base.with_suffix(".review_candidates.tsv"), args.lesson_id, lines)
    write_flex_sfm(base.with_suffix(".flex.sfm"), payload)

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {base.with_suffix('.interlinear.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
