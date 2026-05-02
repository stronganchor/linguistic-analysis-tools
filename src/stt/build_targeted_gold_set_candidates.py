#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from serve_gold_set_review import (
    ReviewStore,
    default_ipa_metadata_path,
    default_text_metadata_path,
    iso_now,
    normalize_space,
    path_windows,
)


IPA_VOWELS = [
    "i",
    "y",
    "ɨ",
    "ʉ",
    "ɯ",
    "u",
    "ɪ",
    "ʏ",
    "ʊ",
    "e",
    "ø",
    "ɘ",
    "ɵ",
    "ɤ",
    "o",
    "ə",
    "ɛ",
    "œ",
    "ɜ",
    "ɞ",
    "ʌ",
    "ɔ",
    "æ",
    "ɐ",
    "a",
    "ɶ",
    "ɑ",
    "ɒ",
]
VOWEL_PATTERN = re.compile("|".join(sorted(map(re.escape, IPA_VOWELS), key=len, reverse=True)))


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_confirmed_path() -> Path:
    candidates = [
        script_dir() / "gold-set-review-ak-isolation" / "gold-set-confirmed.jsonl",
        Path.home() / "Downloads" / "gold-set-confirmed.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_output_dir() -> Path:
    return script_dir() / "gold-set-review-ak-isolation-targeted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a targeted shortlist of A.K. isolation recordings that best "
            "fills the vowel gaps in an existing confirmed gold-set seed."
        )
    )
    parser.add_argument(
        "--confirmed-jsonl",
        nargs="+",
        default=[str(default_confirmed_path())],
        help="One or more confirmed gold-set JSONL files to merge before building the next shortlist.",
    )
    parser.add_argument("--ipa-metadata-path", default=str(default_ipa_metadata_path()))
    parser.add_argument("--text-metadata-path", default=str(default_text_metadata_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--speaker-user-id", type=int, default=8)
    parser.add_argument("--recording-type", default="isolation")
    parser.add_argument("--max-items", type=int, default=72)
    parser.add_argument(
        "--target-total-per-vowel",
        type=int,
        default=12,
        help="Desired total token count per vowel before it stops being prioritized.",
    )
    parser.add_argument(
        "--target-single-per-vowel",
        type=int,
        default=4,
        help="Desired count of clean single-vowel words per vowel before it stops being prioritized.",
    )
    parser.add_argument(
        "--preferred-only",
        action="store_true",
        help="Restrict the shortlist to the preferred tier only.",
    )
    parser.add_argument(
        "--ignore-vowels",
        nargs="*",
        default=[],
        help="IPA vowels to ignore when computing coverage targets, for example: --ignore-vowels ʊ",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_vowels(ipa_text: str) -> list[str]:
    return VOWEL_PATTERN.findall(normalize_space(ipa_text))


def word_key(value: str) -> str:
    return normalize_space(value).casefold()


def parse_ignore_vowels(raw_values: list[str]) -> set[str]:
    ignored: set[str] = set()
    for raw in raw_values:
        for part in re.split(r"[\s,]+", str(raw or "").strip()):
            if part:
                ignored.add(part)
    return ignored


def coverage_counts(rows: list[dict[str, Any]], ignored_vowels: set[str] | None = None) -> tuple[Counter[str], Counter[str]]:
    ignored_vowels = ignored_vowels or set()
    total = Counter()
    single = Counter()
    for row in rows:
        vowels = [
            vowel
            for vowel in extract_vowels(str(row.get("recording_ipa") or ""))
            if vowel not in ignored_vowels
        ]
        total.update(vowels)
        if len(vowels) == 1:
            single[vowels[0]] += 1
    return total, single


def merge_confirmed_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_recordings: set[int] = set()
    for path in paths:
        for row in read_jsonl(path):
            recording_id = int(row.get("recording_id") or 0)
            if not recording_id or recording_id in seen_recordings:
                continue
            seen_recordings.add(recording_id)
            merged.append(row)
    return merged


def load_candidate_pool(args: argparse.Namespace) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="gold-set-target-builder-") as tmp:
        store = ReviewStore(
            ipa_metadata_path=Path(args.ipa_metadata_path).expanduser().resolve(),
            text_metadata_path=Path(args.text_metadata_path).expanduser().resolve(),
            output_dir=Path(tmp),
            candidate_jsonl_path=None,
            speaker_user_id=int(args.speaker_user_id),
            recording_type=str(args.recording_type),
            include_multiword=False,
            include_missing_orthography=False,
            max_items=0,
        )
        return [dict(item) for item in store.items]


def candidate_priority(
    item: dict[str, Any],
    total_counts: Counter[str],
    single_counts: Counter[str],
    total_goal: int,
    single_goal: int,
    confirmed_word_keys: set[str],
    confirmed_orth_keys: set[str],
    selected_word_keys: set[str],
    selected_orth_keys: set[str],
    selected_signatures: set[str],
) -> tuple[float, str, str, str]:
    vowels = item["_vowels"]
    if not vowels:
        return (-1e9, "", "skip", "No recognizable vowels in IPA.")

    remaining_total = {
        vowel: max(0, total_goal - int(total_counts.get(vowel, 0)))
        for vowel in set(vowels)
    }
    remaining_single = {
        vowel: max(0, single_goal - int(single_counts.get(vowel, 0)))
        for vowel in set(vowels)
    }

    score = float(int(item.get("candidate_score") or 0) * 2)
    if item.get("candidate_tier") == "preferred":
        score += 10.0
    elif item.get("candidate_tier") == "strong":
        score += 2.0
    else:
        score -= 20.0

    for vowel in vowels:
        deficit = remaining_total.get(vowel, 0)
        if deficit > 0:
            score += 14.0 + min(deficit, 6) * 5.0
        else:
            score += 1.0

    if len(vowels) == 1:
        vowel = vowels[0]
        single_deficit = remaining_single.get(vowel, 0)
        score += 24.0 + min(single_deficit, 6) * 18.0
        bucket = "single-vowel gap fill" if single_deficit > 0 else "single-vowel backup"
    elif any(remaining_total.get(vowel, 0) > 0 for vowel in vowels):
        score += 6.0
        score -= max(0, len(vowels) - 2) * 6.0
        bucket = "vowel coverage gap fill"
    else:
        score -= max(0, len(vowels) - 1) * 8.0
        bucket = "backup"

    if item["_signature_key"] in selected_signatures:
        score -= 200.0
    if item["_word_key"] in confirmed_word_keys:
        score -= 18.0
    if item["_orth_key"] in confirmed_orth_keys:
        score -= 14.0
    if item["_word_key"] in selected_word_keys:
        score -= 10.0
    if item["_orth_key"] in selected_orth_keys:
        score -= 8.0

    target_vowels = sorted(
        {
            vowel
            for vowel in vowels
            if remaining_total.get(vowel, 0) > 0 or remaining_single.get(vowel, 0) > 0
        },
        key=lambda vowel: (
            -remaining_single.get(vowel, 0),
            -remaining_total.get(vowel, 0),
            vowel,
        ),
    )
    if not target_vowels:
        target_vowels = sorted(set(vowels))
    target_vowels_text = ", ".join(target_vowels)

    if len(vowels) == 1:
        vowel = vowels[0]
        reason = (
            f"single-vowel {vowel} candidate; current gold set has "
            f"{single_counts.get(vowel, 0)}/{single_goal} clean single-vowel items "
            f"and {total_counts.get(vowel, 0)}/{total_goal} total tokens for {vowel}"
        )
    else:
        deficits = ", ".join(
            f"{vowel} {total_counts.get(vowel, 0)}/{total_goal}"
            for vowel in target_vowels
        )
        reason = f"adds undercovered vowels {target_vowels_text}; current totals: {deficits}"
    if item.get("candidate_tier") != "preferred":
        reason += f"; included from {item.get('candidate_tier')} tier because it helps a weak vowel"

    return score, target_vowels_text, bucket, reason


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "selection_rank",
        "priority_score",
        "selection_bucket",
        "target_vowels",
        "recording_id",
        "word_id",
        "speaker_user_id",
        "speaker_name",
        "recording_type",
        "word_title",
        "word_translation",
        "orthography_text",
        "orthography_source",
        "recording_translation",
        "recording_ipa",
        "vowel_sequence",
        "audio_relpath",
        "audio_posix",
        "audio_filename",
        "audio_ext",
        "ipa_vowel_count",
        "candidate_score",
        "candidate_tier",
        "target_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(column, "") for column in columns])


def main() -> None:
    args = parse_args()
    confirmed_paths = [Path(value).expanduser().resolve() for value in args.confirmed_jsonl]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for confirmed_path in confirmed_paths:
        if not confirmed_path.is_file():
            raise SystemExit(f"Confirmed gold-set JSONL not found: {confirmed_path}")

    ignored_vowels = parse_ignore_vowels(list(args.ignore_vowels))

    confirmed_rows = merge_confirmed_rows(confirmed_paths)
    confirmed_ids = {int(row.get("recording_id") or 0) for row in confirmed_rows}
    confirmed_word_keys = {
        word_key(str(row.get("word_title") or ""))
        for row in confirmed_rows
        if word_key(str(row.get("word_title") or ""))
    }
    confirmed_orth_keys = {
        word_key(str(row.get("orthography_text") or ""))
        for row in confirmed_rows
        if word_key(str(row.get("orthography_text") or ""))
    }

    total_counts, single_counts = coverage_counts(confirmed_rows, ignored_vowels=ignored_vowels)
    candidate_pool = load_candidate_pool(args)

    allowed_tiers = {"preferred"} if args.preferred_only else {"preferred", "strong"}
    filtered_pool: list[dict[str, Any]] = []
    for item in candidate_pool:
        if int(item.get("recording_id") or 0) in confirmed_ids:
            continue
        if str(item.get("candidate_tier") or "") not in allowed_tiers:
            continue
        vowels = [
            vowel
            for vowel in extract_vowels(str(item.get("recording_ipa") or ""))
            if vowel not in ignored_vowels
        ]
        if not vowels:
            continue
        item = dict(item)
        item["_vowels"] = vowels
        item["_word_key"] = word_key(str(item.get("word_title") or ""))
        item["_orth_key"] = word_key(str(item.get("orthography_text") or ""))
        item["_signature_key"] = (
            item["_orth_key"],
            normalize_space(str(item.get("recording_ipa") or "")).casefold(),
        )
        filtered_pool.append(item)

    selected: list[dict[str, Any]] = []
    selected_signatures: set[tuple[str, str]] = set()
    selected_word_keys: set[str] = set()
    selected_orth_keys: set[str] = set()
    remaining_pool = filtered_pool[:]

    while remaining_pool and len(selected) < max(0, int(args.max_items or 0)):
        scored_items: list[tuple[float, str, str, str, dict[str, Any]]] = []
        for item in remaining_pool:
            score, target_vowels_text, bucket, reason = candidate_priority(
                item=item,
                total_counts=total_counts,
                single_counts=single_counts,
                total_goal=int(args.target_total_per_vowel),
                single_goal=int(args.target_single_per_vowel),
                confirmed_word_keys=confirmed_word_keys,
                confirmed_orth_keys=confirmed_orth_keys,
                selected_word_keys=selected_word_keys,
                selected_orth_keys=selected_orth_keys,
                selected_signatures=selected_signatures,
            )
            scored_items.append((score, target_vowels_text, bucket, reason, item))

        scored_items.sort(
            key=lambda entry: (
                -entry[0],
                entry[2] != "single-vowel gap fill",
                entry[4].get("candidate_tier") != "preferred",
                len(entry[4]["_vowels"]),
                -int(entry[4].get("candidate_score") or 0),
                str(entry[4].get("word_title") or "").casefold(),
                int(entry[4].get("recording_id") or 0),
            )
        )
        best_score, target_vowels_text, bucket, reason, best_item = scored_items[0]
        if best_score < -100:
            break

        enriched = dict(best_item)
        for private_key in ("_vowels", "_word_key", "_orth_key", "_signature_key"):
            enriched.pop(private_key, None)
        enriched["selection_rank"] = len(selected) + 1
        enriched["priority_score"] = round(best_score, 2)
        enriched["selection_bucket"] = bucket
        enriched["target_vowels"] = target_vowels_text
        enriched["vowel_sequence"] = " ".join(best_item["_vowels"])
        enriched["target_reason"] = reason
        selected.append(enriched)

        total_counts.update(best_item["_vowels"])
        if len(best_item["_vowels"]) == 1:
            single_counts[best_item["_vowels"][0]] += 1
        selected_signatures.add(best_item["_signature_key"])
        if best_item["_word_key"]:
            selected_word_keys.add(best_item["_word_key"])
        if best_item["_orth_key"]:
            selected_orth_keys.add(best_item["_orth_key"])
        remaining_pool = [
            item for item in remaining_pool if int(item.get("recording_id") or 0) != int(best_item["recording_id"])
        ]

    candidates_path = output_dir / "gold-set-candidates.jsonl"
    candidates_tsv_path = output_dir / "gold-set-candidates.tsv"
    merged_confirmed_path = output_dir / "gold-set-confirmed-merged.jsonl"
    summary_path = output_dir / "gold-set-targeted-summary.json"
    write_jsonl(candidates_path, selected)
    write_tsv(candidates_tsv_path, selected)
    write_jsonl(merged_confirmed_path, confirmed_rows)

    confirmed_total, confirmed_single = coverage_counts(confirmed_rows, ignored_vowels=ignored_vowels)
    selected_total, selected_single = coverage_counts(selected, ignored_vowels=ignored_vowels)
    projected_total = Counter(confirmed_total)
    projected_total.update(selected_total)
    projected_single = Counter(confirmed_single)
    projected_single.update(selected_single)
    summary = {
        "generated_at": iso_now(),
        "confirmed_jsonl_paths": [path_windows(path) for path in confirmed_paths],
        "merged_confirmed_jsonl": path_windows(merged_confirmed_path),
        "ipa_metadata_path": path_windows(Path(args.ipa_metadata_path).expanduser().resolve()),
        "text_metadata_path": path_windows(Path(args.text_metadata_path).expanduser().resolve()),
        "output_dir": path_windows(output_dir),
        "candidate_jsonl": path_windows(candidates_path),
        "candidate_tsv": path_windows(candidates_tsv_path),
        "speaker_user_id": int(args.speaker_user_id),
        "recording_type": str(args.recording_type),
        "allowed_tiers": sorted(allowed_tiers),
        "ignored_vowels": sorted(ignored_vowels),
        "max_items": int(args.max_items or 0),
        "target_total_per_vowel": int(args.target_total_per_vowel),
        "target_single_per_vowel": int(args.target_single_per_vowel),
        "confirmed_rows": len(confirmed_rows),
        "shortlist_rows": len(selected),
        "current_total_by_vowel": dict(sorted(confirmed_total.items())),
        "current_single_by_vowel": dict(sorted(confirmed_single.items())),
        "shortlist_total_by_vowel": dict(sorted(selected_total.items())),
        "shortlist_single_by_vowel": dict(sorted(selected_single.items())),
        "projected_total_by_vowel": dict(sorted(projected_total.items())),
        "projected_single_by_vowel": dict(sorted(projected_single.items())),
        "selection_buckets": dict(Counter(str(row.get("selection_bucket") or "") for row in selected)),
        "candidate_tiers": dict(Counter(str(row.get("candidate_tier") or "") for row in selected)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for confirmed_path in confirmed_paths:
        print(f"Confirmed seed: {path_windows(confirmed_path)}")
    print(f"Merged confirmed: {path_windows(merged_confirmed_path)}")
    print(f"Shortlist JSONL: {path_windows(candidates_path)}")
    print(f"Shortlist TSV: {path_windows(candidates_tsv_path)}")
    print(f"Summary: {path_windows(summary_path)}")
    print(f"Shortlist items: {len(selected)}")


if __name__ == "__main__":
    main()
