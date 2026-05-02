#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "Manidar/DEZD-Dictionary-Kirmancki"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "external-data" / "dezd-dictionary-kirmancki-20260414"


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def simplify_orthography(value: str) -> str:
    text = value
    replacements = {
        "Â": "E",
        "â": "e",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.lower()
    text = re.sub(r"[.,!?;:()\"]+", " ", text)
    text = normalize_spaces(text)
    return text


def clean_entry_text(value: str) -> str:
    text = unicodedata.normalize("NFC", html.unescape(value or ""))
    text = text.replace("\t", " ")
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = normalize_spaces(text)
    return text


def looks_like_variant(value: str) -> bool:
    if not value:
        return False
    if len(value) > 32:
        return False
    if any(char.isdigit() for char in value):
        return False
    if value.count(" ") > 2:
        return False
    if "-" in value or '"' in value:
        return False
    return True


def extract_candidate_strings(raw_value: str) -> list[str]:
    cleaned = clean_entry_text(raw_value)
    if not cleaned:
        return []

    candidates = [re.sub(r"\([^)]*\)", " ", cleaned)]
    for match in re.finditer(r"\(([^)]*)\)", cleaned):
        variant = normalize_spaces(match.group(1))
        if looks_like_variant(variant):
            candidates.append(variant)
    return [normalize_spaces(item) for item in candidates if normalize_spaces(item)]


def extract_unigrams(raw_value: str) -> set[str]:
    unigrams: set[str] = set()
    for candidate in extract_candidate_strings(raw_value):
        simplified = simplify_orthography(candidate)
        simplified = re.sub(r"[^a-zçğıöşüêîû' ]+", " ", simplified)
        simplified = normalize_spaces(simplified)
        if not simplified:
            continue
        for token in simplified.split():
            if not token:
                continue
            if len(token) == 1 and token not in {"a", "e", "o", "u"}:
                continue
            unigrams.add(token)
    return unigrams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simplified-orthography unigram list from DEZD Kirmancki.")
    parser.add_argument("--dataset", default=DATASET_ID, help="HF dataset id")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)

    all_unigrams: set[str] = set()
    row_count = 0
    rows_with_terms = 0
    sample_terms: list[dict[str, object]] = []

    for row in dataset:
        row_count += 1
        raw_value = str(row.get("Kırmancki") or "")
        terms = sorted(extract_unigrams(raw_value))
        if not terms:
            continue
        rows_with_terms += 1
        all_unigrams.update(terms)
        if len(sample_terms) < 25:
            sample_terms.append(
                {
                    "raw": raw_value,
                    "terms": terms,
                }
            )

    unigram_path = output_dir / "dezd_simple_unigrams.txt"
    unigram_path.write_text("\n".join(sorted(all_unigrams)) + "\n", encoding="utf-8")

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "row_count": row_count,
        "rows_with_terms": rows_with_terms,
        "unique_unigrams": len(all_unigrams),
        "output_path": str(unigram_path),
        "sample_terms": sample_terms,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
