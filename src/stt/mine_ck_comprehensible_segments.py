#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prepare_lltools_ipa_dataset import normalize_plain_text
from review_lltools_ipa_comparison import ipa_to_canonical, orthography_to_canonical


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = ROOT_DIR / "Speech to Text IPA" / "transcription_outputs" / "ck_corpus_dual_ctc_20260411-131711"
DEFAULT_AK_TEXT_MANIFEST = ROOT_DIR / "Speech to Text IPA" / "training_data" / "genc-palu-recording_text-20260409-071931-simple-v002-whisper-medium" / "manifest_text.tsv"
DEFAULT_AK_IPA_MANIFEST = ROOT_DIR / "Speech to Text IPA" / "training_data" / "genc-palu-recording_ipa-20260410-065107-whisper-medium-v001" / "manifest_ipa.tsv"
CORRECTED_SEGMENTS_DIR = ROOT_DIR / "audio_corpus" / "corrected_segments"


def default_dictionary_tsv() -> Path:
    candidates = [
        ROOT_DIR / "Dictionaries" / "tsv_output" / "Combined - Harun Turgut + DEZD.tsv",
        ROOT_DIR / "Dictionaries" / "tsv_output" / "Palu - Bingöl Harun Turgut - corrected.tsv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


DEFAULT_DICT_TSV = default_dictionary_tsv()

CORRECTED_SOURCE_BASENAMES = {
    "1. Exploring a mine.wav",
    "Mesele kurık lir.wav",
    "Sabri neweş.wav",
}

EXCLUDED_TRAINING_SOURCE_IDS = {
    "full_orho_s_story",
}

EXCLUDED_TRAINING_SOURCE_TITLES = {
    "orho's story",
}

MANUAL_FUNCTION_WORDS = {
    "a", "ama", "ani", "aw", "aya", "e", "ee", "eja", "eya", "ez", "ha", "he", "hema", "her",
    "hın", "hiri", "ho", "ı", "iki", "im", "ina", "inke", "it", "itiya", "jo", "ka", "ma", "mı",
    "mi", "na", "ni", "nı", "non", "o", "ow", "pe", "qe", "ra", "ro", "sa", "ser", "se", "ta",
    "tı", "u", "va", "ve", "von", "wa", "we", "wı", "wo", "ya", "yo", "yen", "yı", "zaf", "şın",
}


def normalize_surface(value: str) -> str:
    text = normalize_plain_text(value)
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("|", " ")
    for mark in [".", "?", "!", ",", ":", ";", "(", ")", '"', "“", "”", "’", "'"]:
        text = text.replace(mark, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def surface_tokens(value: str) -> list[str]:
    return [part for part in normalize_surface(value).split() if part]


def orth_canonical(value: str) -> str:
    return orthography_to_canonical(normalize_surface(value))


def orth_canonical_tokens(value: str) -> list[str]:
    return [part for part in orth_canonical(value).split() if part]


def ipa_canonical(value: str) -> str:
    return ipa_to_canonical(value or "")


def ipa_canonical_tokens(value: str) -> list[str]:
    return [part for part in ipa_canonical(value).split() if part]


def agreement_score(orth_tokens: list[str], orth_can: str, ipa_tokens: list[str], ipa_can: str) -> float:
    return 0.55 * normalized_similarity(orth_can, ipa_can) + 0.45 * token_jaccard(orth_tokens, ipa_tokens)


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, start=1):
        current = [index_a]
        for index_b, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[index_b] + 1, current[index_b - 1] + 1, previous[index_b - 1] + cost))
        previous = current
    return previous[-1]


def normalized_similarity(a: str, b: str) -> float:
    aa = compact(a)
    bb = compact(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return max(0.0, 1.0 - (levenshtein_distance(aa, bb) / max(len(aa), len(bb))))


def token_jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


@dataclass
class CorrectedSegment:
    audio_basename: str
    start_sec: float
    end_sec: float
    text: str
    text_canonical: str


def parse_corrected_segment_file(path: Path) -> list[CorrectedSegment]:
    rows: list[CorrectedSegment] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    audio_basename = ""
    if lines:
        match = re.search(r"file:///.*?/([^/]+\.wav)", lines[0])
        if match:
            audio_basename = match.group(1)
    for raw_line in lines[2:]:
        line = raw_line.strip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        text = normalize_surface(parts[6])
        if not text:
            continue
        rows.append(
            CorrectedSegment(
                audio_basename=audio_basename,
                start_sec=float(parts[3]),
                end_sec=float(parts[5]),
                text=text,
                text_canonical=orth_canonical(text),
            )
        )
    return rows


def load_corrected_segments() -> dict[str, list[CorrectedSegment]]:
    by_audio: dict[str, list[CorrectedSegment]] = defaultdict(list)
    for path in sorted(CORRECTED_SEGMENTS_DIR.glob("*.tsv")):
        for row in parse_corrected_segment_file(path):
            if row.audio_basename:
                by_audio[row.audio_basename].append(row)
    return by_audio


def build_ak_resources(text_manifest: Path, ipa_manifest: Path) -> dict[str, Any]:
    text_rows = read_tsv_rows(text_manifest)
    ipa_rows = read_tsv_rows(ipa_manifest)

    orth_token_freq: Counter[str] = Counter()
    orth_phrase_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    ipa_phrase_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    token_to_titles: dict[str, Counter[str]] = defaultdict(Counter)

    for row in text_rows:
        surface = normalize_surface(row.get("target_text") or row.get("recording_text") or "")
        if not surface:
            continue
        canonical = orth_canonical(surface)
        orth_phrase_map[canonical].append(
            {
                "word_title": row.get("word_title") or "",
                "recording_type": row.get("recording_type") or "",
                "surface": surface,
            }
        )
        for token in orth_canonical_tokens(surface):
            orth_token_freq[token] += 1
            token_to_titles[token][row.get("word_title") or surface] += 1

    for row in ipa_rows:
        target_ipa = str(row.get("target_ipa") or "").strip()
        if not target_ipa:
            continue
        canonical = ipa_canonical(target_ipa)
        ipa_phrase_map[canonical].append(
            {
                "word_title": row.get("word_title") or "",
                "recording_type": row.get("recording_type") or "",
                "target_ipa": target_ipa,
            }
        )

    auto_function_words = {
        token
        for token, freq in orth_token_freq.items()
        if freq >= 20 and len(token) <= 4
    }
    function_words = set(MANUAL_FUNCTION_WORDS) | auto_function_words
    return {
        "orth_token_freq": orth_token_freq,
        "orth_phrase_map": orth_phrase_map,
        "ipa_phrase_map": ipa_phrase_map,
        "token_to_titles": token_to_titles,
        "function_words": function_words,
    }


def build_dictionary_resources(dict_tsv: Path) -> dict[str, Any]:
    rows = read_tsv_rows(dict_tsv)
    all_token_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    content_token_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    phrase_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    clean_phrase_map: dict[str, list[dict[str, str]]] = defaultdict(list)
    token_best_weight: dict[str, float] = defaultdict(float)

    def info_weight(row: dict[str, str]) -> tuple[float, str]:
        entry_type = str(row.get("entry_type") or "")
        parent = str(row.get("parent") or "").strip()
        needs_review = str(row.get("needs_review") or "").strip()
        if entry_type == "header":
            return 0.0, "header"
        if entry_type in {"prefix", "suffix"}:
            return 0.25, "affix"
        if needs_review in {"out_of_order", "num_in_entry", "num_in_def"} or needs_review.startswith("huge_"):
            return 0.35, "review_low"
        if needs_review == "1":
            return (0.75, "review_phrase") if parent else (0.55, "review_single")
        return 1.0, "clean"

    for row in rows:
        entry = normalize_surface(row.get("entry") or "")
        if not entry:
            continue
        entry_type = str(row.get("entry_type") or "")
        weight, dict_tier = info_weight(row)
        info = {
            "entry": entry,
            "definition": normalize_surface(row.get("definition") or ""),
            "entry_type": entry_type,
            "dict_weight": weight,
            "dict_tier": dict_tier,
        }
        phrase_map[orth_canonical(entry)].append(info)
        if weight >= 0.75:
            clean_phrase_map[orth_canonical(entry)].append(info)
        for token in orth_canonical_tokens(entry):
            all_token_map[token].append(info)
            token_best_weight[token] = max(token_best_weight[token], weight)
            if entry_type not in {"header", "prefix", "suffix"}:
                content_token_map[token].append(info)

    return {
        "all_token_map": all_token_map,
        "content_token_map": content_token_map,
        "phrase_map": phrase_map,
        "clean_phrase_map": clean_phrase_map,
        "token_best_weight": dict(token_best_weight),
    }


def build_ck_corrected_resources(corrected_by_audio: dict[str, list[CorrectedSegment]]) -> dict[str, Any]:
    token_map: dict[str, Counter[str]] = defaultdict(Counter)
    phrase_map: dict[str, list[str]] = defaultdict(list)
    for segments in corrected_by_audio.values():
        for segment in segments:
            phrase_map[segment.text_canonical].append(segment.text)
            for token in orth_canonical_tokens(segment.text):
                token_map[token][segment.text] += 1
    return {
        "token_map": token_map,
        "phrase_map": phrase_map,
    }


def is_junk_segment(orth_text: str, ipa_text: str, orth_tokens: list[str], ipa_tokens: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    orth_compact = compact(orth_text)
    ipa_compact = compact(ipa_text)
    if not orth_compact:
        reasons.append("empty_orth")
    if not ipa_compact:
        reasons.append("empty_ipa")
    if orth_compact and re.fullmatch(r"(.)\1{2,}", orth_compact):
        reasons.append("orth_repeated_char")
    if ipa_compact and re.fullmatch(r"(.)\1{2,}", ipa_compact):
        reasons.append("ipa_repeated_char")
    if orth_compact in {"nnn", "mmm", "aaa"}:
        reasons.append("orth_placeholder")
    if ipa_compact in {"nnn", "mmm", "aaa"}:
        reasons.append("ipa_placeholder")
    if orth_tokens and all(len(token) <= 1 for token in orth_tokens):
        reasons.append("orth_all_single_char")
    if ipa_tokens and all(len(token) <= 1 for token in ipa_tokens):
        reasons.append("ipa_all_single_char")
    return (len(reasons) > 0, reasons)


def best_corrected_overlap(
    row: dict[str, str],
    corrected_by_audio: dict[str, list[CorrectedSegment]],
) -> tuple[float, str]:
    audio_basename = Path(str(row.get("source_audio_posix") or "")).name
    corrected_rows = corrected_by_audio.get(audio_basename, [])
    if not corrected_rows:
        return 0.0, ""
    start = float(row.get("start_sec") or 0.0)
    end = float(row.get("end_sec") or 0.0)
    candidates = []
    for corrected in corrected_rows:
        overlap = overlap_seconds(start, end, corrected.start_sec, corrected.end_sec)
        if overlap > 0.15:
            candidates.append((corrected, overlap))
    if not candidates:
        return 0.0, ""
    candidates.sort(key=lambda item: item[0].start_sec)
    joined_text = " ".join(item[0].text for item in candidates)
    predicted = orth_canonical(str(row.get("predicted_orthography") or ""))
    similarity = normalized_similarity(predicted, orth_canonical(joined_text))
    return similarity, joined_text


def top_counter_items(counter: Counter[str], limit: int = 3) -> str:
    return " | ".join(item for item, _count in counter.most_common(limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-comprehensibility C.K. segments from the dual IPA+orthography transcription run."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument(
        "--input-tsv",
        default="",
        help="Optional explicit transcription TSV. If omitted, prefer ck_triple_transcriptions.tsv, then ck_dual_transcriptions.tsv in --run-dir.",
    )
    parser.add_argument("--ak-text-manifest", default=str(DEFAULT_AK_TEXT_MANIFEST))
    parser.add_argument("--ak-ipa-manifest", default=str(DEFAULT_AK_IPA_MANIFEST))
    parser.add_argument("--dictionary-tsv", default=str(DEFAULT_DICT_TSV))
    parser.add_argument("--min-duration-sec", type=float, default=0.35)
    parser.add_argument("--max-duration-sec", type=float, default=8.0)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if args.input_tsv:
        dual_path = Path(args.input_tsv).expanduser().resolve()
    else:
        triple_path = run_dir / "ck_triple_transcriptions.tsv"
        dual_path = triple_path if triple_path.exists() else (run_dir / "ck_dual_transcriptions.tsv")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (run_dir / "mining")
    output_dir.mkdir(parents=True, exist_ok=True)

    ak_resources = build_ak_resources(
        text_manifest=Path(args.ak_text_manifest).expanduser().resolve(),
        ipa_manifest=Path(args.ak_ipa_manifest).expanduser().resolve(),
    )
    dict_resources = build_dictionary_resources(Path(args.dictionary_tsv).expanduser().resolve())
    corrected_by_audio = load_corrected_segments()
    ck_corrected_resources = build_ck_corrected_resources(corrected_by_audio)

    rows = read_tsv_rows(dual_path)
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        source_title_norm = normalize_surface(str(row.get("source_title") or ""))
        if source_id in EXCLUDED_TRAINING_SOURCE_IDS or source_title_norm in EXCLUDED_TRAINING_SOURCE_TITLES:
            continue

        duration = float(row.get("duration_sec") or 0.0)
        orth_text = normalize_surface(str(row.get("predicted_orthography") or ""))
        ipa_text = str(row.get("predicted_ipa") or "").strip()
        whisper_ipa_text = str(row.get("whisper_predicted_ipa_repaired") or row.get("whisper_predicted_ipa") or "").strip()
        orth_can = str(row.get("predicted_orthography_canonical") or orth_canonical(orth_text))
        ipa_can = str(row.get("predicted_ipa_canonical") or ipa_canonical(ipa_text))
        whisper_ipa_can = str(row.get("whisper_predicted_ipa_canonical") or ipa_canonical(whisper_ipa_text))
        orth_tokens = [tok for tok in orth_can.split() if tok]
        ipa_tokens = [tok for tok in ipa_can.split() if tok]
        whisper_ipa_tokens = [tok for tok in whisper_ipa_can.split() if tok]

        junk, junk_reasons = is_junk_segment(orth_text, ipa_text, orth_tokens, ipa_tokens)
        whisper_junk = False
        whisper_junk_reasons: list[str] = []
        if whisper_ipa_text:
            whisper_junk, whisper_junk_reasons = is_junk_segment(orth_text, whisper_ipa_text, orth_tokens, whisper_ipa_tokens)
        if duration < args.min_duration_sec or duration > args.max_duration_sec:
            junk_reasons.append("duration_outside_review_band")
            if whisper_ipa_text:
                whisper_junk_reasons.append("duration_outside_review_band")

        dict_hits = [tok for tok in orth_tokens if tok in dict_resources["all_token_map"]]
        dict_content_hits = [tok for tok in orth_tokens if tok in dict_resources["content_token_map"]]
        ak_token_hits = [tok for tok in orth_tokens if tok in ak_resources["orth_token_freq"]]
        ck_corrected_hits = [tok for tok in orth_tokens if tok in ck_corrected_resources["token_map"]]

        token_support_scores = []
        for tok in orth_tokens:
            dict_weight = float(dict_resources["token_best_weight"].get(tok, 0.0))
            ak_weight = 1.0 if tok in ak_resources["orth_token_freq"] else 0.0
            ck_weight = 1.0 if tok in ck_corrected_resources["token_map"] else 0.0
            token_support_scores.append(max(dict_weight, ak_weight, ck_weight))
        lexical_coverage = (sum(token_support_scores) / len(orth_tokens)) if orth_tokens else 0.0

        content_tokens = [tok for tok in orth_tokens if tok not in ak_resources["function_words"] and len(tok) >= 3]
        content_support_count = sum(
            1
            for tok in content_tokens
            if float(dict_resources["token_best_weight"].get(tok, 0.0)) >= 0.75 or tok in ck_corrected_resources["token_map"]
        )
        content_support = min(1.0, content_support_count / 2.0)

        mms_alignment = agreement_score(orth_tokens, orth_can, ipa_tokens, ipa_can)
        whisper_alignment = agreement_score(orth_tokens, orth_can, whisper_ipa_tokens, whisper_ipa_can) if whisper_ipa_text else 0.0
        agreement = max(mms_alignment, whisper_alignment)
        witness_consensus = (
            max(normalized_similarity(ipa_can, whisper_ipa_can), token_jaccard(ipa_tokens, whisper_ipa_tokens))
            if whisper_ipa_text
            else 0.0
        )
        exact_ak_orth = orth_can in ak_resources["orth_phrase_map"]
        exact_ak_ipa = ipa_can in ak_resources["ipa_phrase_map"]
        exact_ak_whisper = whisper_ipa_can in ak_resources["ipa_phrase_map"] if whisper_ipa_text else False
        exact_dict_phrase_weight = max(
            (float(info["dict_weight"]) for info in dict_resources["phrase_map"].get(orth_can, [])),
            default=0.0,
        )
        exact_dict_phrase = exact_dict_phrase_weight >= 0.75
        exact_ck_phrase = orth_can in ck_corrected_resources["phrase_map"]
        exact_support = max(
            1.0 if exact_ak_orth else 0.0,
            1.0 if exact_ak_ipa else 0.0,
            1.0 if exact_ak_whisper else 0.0,
            exact_dict_phrase_weight,
            1.0 if exact_ck_phrase else 0.0,
        )

        corrected_overlap_sim, corrected_overlap_text = best_corrected_overlap(row, corrected_by_audio)
        duration_score = 1.0 if 0.8 <= duration <= 4.5 else (0.75 if 0.5 <= duration <= 6.5 else 0.35)

        effective_junk = junk and (not whisper_ipa_text or whisper_junk)
        active_junk_reasons = whisper_junk_reasons if (whisper_ipa_text and not whisper_junk and junk) else junk_reasons

        junk_penalty = 0.18 * len([reason for reason in active_junk_reasons if reason != "duration_outside_review_band"])
        if "duration_outside_review_band" in active_junk_reasons:
            junk_penalty += 0.08

        raw_score = (
            0.24 * agreement
            + 0.08 * witness_consensus
            + 0.22 * lexical_coverage
            + 0.14 * content_support
            + 0.18 * exact_support
            + 0.08 * duration_score
            + 0.08 * corrected_overlap_sim
        ) - junk_penalty
        if len(orth_tokens) >= 2:
            raw_score += 0.05
        else:
            raw_score -= 0.08
        if len(orth_tokens) >= 2 and exact_support >= 0.75:
            raw_score += 0.05
        if whisper_ipa_text and witness_consensus >= 0.82:
            raw_score += 0.04
        score = max(0.0, min(1.0, raw_score))

        prefer_whisper = False
        if whisper_ipa_text and not whisper_junk:
            if exact_ak_whisper and not exact_ak_ipa:
                prefer_whisper = True
            elif whisper_alignment > (mms_alignment + 0.05):
                prefer_whisper = True
            elif whisper_alignment >= mms_alignment and witness_consensus >= 0.88:
                prefer_whisper = True

        assistant_guess_ipa = whisper_ipa_text if prefer_whisper else ipa_text
        assistant_guess_source = "whisper" if prefer_whisper else "new_mms"

        is_corrected_source = Path(str(row.get("source_audio_posix") or "")).name in CORRECTED_SOURCE_BASENAMES

        matched_ak_titles = Counter()
        for tok in orth_tokens:
            matched_ak_titles.update(ak_resources["token_to_titles"].get(tok, Counter()))
        matched_dict_entries = []
        seen_dict = set()
        for tok in orth_tokens:
            for info in dict_resources["content_token_map"].get(tok, []):
                key = (info["entry"], info["definition"])
                if key in seen_dict:
                    continue
                seen_dict.add(key)
                matched_dict_entries.append(f"{info['entry']}={info['definition']}")
                if len(matched_dict_entries) >= 5:
                    break
            if len(matched_dict_entries) >= 5:
                break

        phrase_candidate = (
            (not effective_junk)
            and (len(orth_tokens) >= 2)
            and (0.8 <= duration <= min(args.max_duration_sec, 6.5))
            and (score >= 0.70)
            and (lexical_coverage >= 0.78)
            and (agreement >= 0.62)
            and (
                content_support_count > 0
                or exact_support >= 0.75
                or corrected_overlap_sim >= 0.72
                or witness_consensus >= 0.72
            )
        )
        phrase_priority = (
            (not effective_junk)
            and (len(orth_tokens) >= 2)
            and (0.8 <= duration <= min(args.max_duration_sec, 6.5))
            and (score >= 0.76)
            and (lexical_coverage >= 0.85)
            and (agreement >= 0.70)
            and (
                content_support_count >= 1
                or exact_support >= 1.0
                or corrected_overlap_sim >= 0.82
                or witness_consensus >= 0.82
            )
        )
        single_word_candidate = (
            (not effective_junk)
            and (len(orth_tokens) == 1)
            and (0.45 <= duration <= 2.5)
            and (score >= 0.74)
            and (lexical_coverage >= 0.95)
            and (agreement >= 0.75)
            and (exact_support >= 0.75 or witness_consensus >= 0.9)
        )

        if phrase_priority:
            tier = "A"
        elif phrase_candidate:
            tier = "B"
        elif single_word_candidate:
            tier = "W"
        else:
            tier = ""

        if tier == "A":
            candidate_kind = "phrase_priority"
        elif tier == "B":
            candidate_kind = "phrase_candidate"
        elif tier == "W":
            candidate_kind = "single_word"
        else:
            candidate_kind = ""

        scored_rows.append(
            {
                **row,
                "score": round(score * 100.0, 2),
                "tier": tier,
                "agreement_score": round(agreement, 4),
                "mms_alignment_score": round(mms_alignment, 4),
                "whisper_alignment_score": round(whisper_alignment, 4),
                "witness_consensus": round(witness_consensus, 4),
                "lexical_coverage": round(lexical_coverage, 4),
                "content_support_count": content_support_count,
                "token_count": len(orth_tokens),
                "dict_hit_count": len(set(dict_hits)),
                "ak_hit_count": len(set(ak_token_hits)),
                "ck_corrected_hit_count": len(set(ck_corrected_hits)),
                "exact_dict_phrase_weight": round(exact_dict_phrase_weight, 2),
                "exact_ak_orth": int(exact_ak_orth),
                "exact_ak_ipa": int(exact_ak_ipa),
                "exact_ak_whisper": int(exact_ak_whisper),
                "exact_dict_phrase": int(exact_dict_phrase),
                "exact_ck_phrase": int(exact_ck_phrase),
                "corrected_overlap_similarity": round(corrected_overlap_sim, 4),
                "corrected_overlap_text": corrected_overlap_text,
                "is_corrected_source": int(is_corrected_source),
                "candidate_kind": candidate_kind,
                "assistant_guess_ipa": assistant_guess_ipa,
                "assistant_guess_source": assistant_guess_source,
                "whisper_predicted_ipa": str(row.get("whisper_predicted_ipa") or ""),
                "whisper_predicted_ipa_repaired": whisper_ipa_text,
                "whisper_predicted_ipa_canonical": whisper_ipa_can,
                "matched_ak_titles": top_counter_items(matched_ak_titles, limit=5),
                "matched_dict_entries": " | ".join(matched_dict_entries),
                "junk_reasons": ",".join(junk_reasons),
                "whisper_junk_reasons": ",".join(whisper_junk_reasons),
            }
        )

    scored_rows.sort(
        key=lambda row: (
            -float(row["score"]),
            str(row["source_group"]),
            str(row["source_title"]),
            float(row["start_sec"]),
        )
    )

    all_scored_path = output_dir / "ck_comprehensible_scored.tsv"
    candidate_path = output_dir / "ck_comprehensible_candidates.tsv"
    candidate_novel_path = output_dir / "ck_comprehensible_candidates_novel.tsv"
    phrase_candidate_path = output_dir / "ck_comprehensible_phrase_candidates.tsv"
    phrase_candidate_novel_path = output_dir / "ck_comprehensible_phrase_candidates_novel.tsv"
    single_word_candidate_path = output_dir / "ck_comprehensible_single_word_candidates.tsv"
    single_word_candidate_novel_path = output_dir / "ck_comprehensible_single_word_candidates_novel.tsv"
    corrected_calibration_path = output_dir / "ck_corrected_source_calibration.tsv"
    summary_path = output_dir / "summary.json"

    fieldnames = list(scored_rows[0].keys()) if scored_rows else []
    if fieldnames:
        with all_scored_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(scored_rows)

        candidates = [row for row in scored_rows if row["tier"] in {"A", "B", "W"}]
        with candidate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(candidates)

        candidates_novel = [row for row in candidates if int(row["is_corrected_source"]) == 0]
        with candidate_novel_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(candidates_novel)

        phrase_candidates = [row for row in candidates if row["candidate_kind"] in {"phrase_priority", "phrase_candidate"}]
        with phrase_candidate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(phrase_candidates)

        phrase_candidates_novel = [row for row in phrase_candidates if int(row["is_corrected_source"]) == 0]
        with phrase_candidate_novel_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(phrase_candidates_novel)

        single_word_candidates = [row for row in candidates if row["candidate_kind"] == "single_word"]
        with single_word_candidate_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(single_word_candidates)

        single_word_candidates_novel = [row for row in single_word_candidates if int(row["is_corrected_source"]) == 0]
        with single_word_candidate_novel_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(single_word_candidates_novel)

        corrected_rows = [row for row in scored_rows if int(row["is_corrected_source"]) == 1]
        with corrected_calibration_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(corrected_rows)

    summary = {
        "run_dir": str(run_dir),
        "input_rows": len(rows),
        "scored_rows": len(scored_rows),
        "candidate_rows": len([row for row in scored_rows if row["tier"] in {"A", "B", "W"}]),
        "candidate_novel_rows": len([row for row in scored_rows if row["tier"] in {"A", "B", "W"} and int(row["is_corrected_source"]) == 0]),
        "phrase_candidate_rows": len([row for row in scored_rows if row["candidate_kind"] in {"phrase_priority", "phrase_candidate"}]),
        "phrase_candidate_novel_rows": len([row for row in scored_rows if row["candidate_kind"] in {"phrase_priority", "phrase_candidate"} and int(row["is_corrected_source"]) == 0]),
        "single_word_candidate_rows": len([row for row in scored_rows if row["candidate_kind"] == "single_word"]),
        "single_word_candidate_novel_rows": len([row for row in scored_rows if row["candidate_kind"] == "single_word" and int(row["is_corrected_source"]) == 0]),
        "tier_counts": dict(Counter(row["tier"] or "none" for row in scored_rows)),
        "candidate_kind_counts": dict(Counter(row["candidate_kind"] or "none" for row in scored_rows)),
        "source_group_candidate_counts": dict(
            Counter(row["source_group"] for row in scored_rows if row["candidate_kind"] in {"phrase_priority", "phrase_candidate"})
        ),
        "top_novel_phrase_examples": [
            {
                "segment_id": row["segment_id"],
                "source_title": row["source_title"],
                "start_sec": row["start_sec"],
                "duration_sec": row["duration_sec"],
                "predicted_orthography": row["predicted_orthography"],
                "predicted_ipa": row["predicted_ipa"],
                "score": row["score"],
                "tier": row["tier"],
                "matched_ak_titles": row["matched_ak_titles"],
                "matched_dict_entries": row["matched_dict_entries"],
            }
            for row in scored_rows
            if row["candidate_kind"] in {"phrase_priority", "phrase_candidate"} and int(row["is_corrected_source"]) == 0
        ][:25],
        "top_novel_single_word_examples": [
            {
                "segment_id": row["segment_id"],
                "source_title": row["source_title"],
                "start_sec": row["start_sec"],
                "duration_sec": row["duration_sec"],
                "predicted_orthography": row["predicted_orthography"],
                "predicted_ipa": row["predicted_ipa"],
                "score": row["score"],
                "tier": row["tier"],
                "matched_ak_titles": row["matched_ak_titles"],
                "matched_dict_entries": row["matched_dict_entries"],
            }
            for row in scored_rows
            if row["candidate_kind"] == "single_word" and int(row["is_corrected_source"]) == 0
        ][:25],
        "outputs": {
            "all_scored_tsv": str(all_scored_path),
            "candidates_tsv": str(candidate_path),
            "candidates_novel_tsv": str(candidate_novel_path),
            "phrase_candidates_tsv": str(phrase_candidate_path),
            "phrase_candidates_novel_tsv": str(phrase_candidate_novel_path),
            "single_word_candidates_tsv": str(single_word_candidate_path),
            "single_word_candidates_novel_tsv": str(single_word_candidate_novel_path),
            "corrected_calibration_tsv": str(corrected_calibration_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
