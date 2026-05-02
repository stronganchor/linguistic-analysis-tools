#!/usr/bin/env python3
"""Build category-level A.K. interlinear texts from LL Tools STT metadata."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from auto_interlinearize import (
    DEFAULT_DICTIONARY,
    DEFAULT_REVIEW_SEED,
    DEFAULT_SEED,
    ROOT,
    Analysis,
    interlinearize,
    load_dictionary,
    load_seed,
    summarize,
    tokenize_line,
    write_flex_sfm,
    write_json,
    write_markdown,
    write_review_tsv,
    write_tokens_tsv,
)


DEFAULT_METADATA = ROOT / "zazacaogren/live-ipa-recording-sync-20260428/metadata.csv"
DEFAULT_SOURCE_DIR = ROOT / "interlinearization/sources/ak_categories"
DEFAULT_OUTPUT_DIR = ROOT / "interlinearization/outputs"
DEFAULT_MANIFEST_TSV = ROOT / "interlinearization/sources/ak_category_texts_manifest.tsv"
DEFAULT_SUMMARY_JSON = ROOT / "interlinearization/outputs/ak_category_texts.summary.json"

SOURCE_FIELDS = [
    "id",
    "text",
    "text_projected",
    "token_count",
    "word_audio_id",
    "recording_id",
    "recording_type",
    "word_id",
    "word_title",
    "word_translation",
    "recording_translation",
    "recording_ipa",
    "review_status",
    "needs_review",
    "category_slug",
    "category_name",
    "audio",
    "audio_url",
    "duration_sec",
]


def html_unescape(value: str) -> str:
    return html.unescape(value or "").replace("\u2019", "'").strip()


def slugify(value: str) -> str:
    value = html_unescape(value).lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "uncategorized"


def safe_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def source_row(row: dict[str, str]) -> dict[str, str]:
    text = html_unescape(row.get("recording_text", ""))
    word_audio_id = row.get("word_audio_id") or row.get("recording_id")
    return {
        "id": word_audio_id,
        "text": text,
        "text_projected": " ".join(tokenize_line(text)),
        "token_count": str(len(tokenize_line(text))),
        "word_audio_id": word_audio_id,
        "recording_id": row.get("recording_id", ""),
        "recording_type": row.get("recording_type", ""),
        "word_id": row.get("word_id", ""),
        "word_title": html_unescape(row.get("word_title", "")),
        "word_translation": html_unescape(row.get("word_translation", "")),
        "recording_translation": html_unescape(row.get("recording_translation", "")),
        "recording_ipa": html_unescape(row.get("recording_ipa", "")),
        "review_status": row.get("review-status", ""),
        "needs_review": row.get("needs_review", ""),
        "category_slug": row.get("category_slug", ""),
        "category_name": html_unescape(row.get("category_name", "")),
        "audio": row.get("audio", ""),
        "audio_url": row.get("audio_url", ""),
        "duration_sec": row.get("duration_seconds", ""),
    }


def read_ak_rows(path: Path, speaker_user_id: str, reviewed_only: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("speaker_user_id", "")).strip() != speaker_user_id:
                continue
            if not html_unescape(row.get("recording_text", "")):
                continue
            if reviewed_only and (row.get("review-status") != "reviewed" or row.get("needs_review") == "1"):
                continue
            rows.append(row)
    return rows


def group_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("category_slug") or slugify(row.get("category_name", ""))].append(row)
    return dict(grouped)


def sort_key(row: dict[str, str]) -> tuple[int, int, str, int]:
    return (
        safe_int(row.get("word_id", "")),
        {"isolation": 0, "introduction": 1, "question": 2}.get(row.get("recording_type", ""), 9),
        row.get("recording_type", ""),
        safe_int(row.get("word_audio_id") or row.get("recording_id", "")),
    )


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SOURCE_FIELDS})


def write_outputs(
    lesson_id: str,
    title: str,
    source_path: Path,
    rows: list[dict[str, str]],
    seed: dict[str, Analysis],
    exact: dict[str, Analysis],
    folded: dict[str, list[Analysis]],
    output_dir: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    interlinear_rows = interlinearize(rows, seed, exact, folded)
    payload = {
        "schema": "zazaki_interlinear.v1",
        "lesson_id": lesson_id,
        "title": title,
        "source": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
        "dictionary": str(DEFAULT_DICTIONARY),
        "seed": str(DEFAULT_SEED),
        "metadata": metadata,
        "summary": summarize(interlinear_rows),
        "lines": interlinear_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / lesson_id
    write_json(base.with_suffix(".interlinear.json"), payload)
    write_markdown(base.with_suffix(".interlinear.md"), payload)
    write_tokens_tsv(base.with_suffix(".tokens.tsv"), lesson_id, interlinear_rows)
    write_review_tsv(base.with_suffix(".review_candidates.tsv"), lesson_id, interlinear_rows)
    write_flex_sfm(base.with_suffix(".flex.sfm"), payload)
    return payload["summary"]


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "lesson_id",
        "title",
        "category_slug",
        "category_name",
        "source_tsv",
        "lines",
        "tokens",
        "matched_tokens",
        "matched_pct",
        "high_confidence_tokens",
        "high_confidence_pct",
        "mean_confidence",
        "reviewed_lines",
        "needs_review_lines",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build A.K. category interlinear texts from metadata.csv.")
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--speaker-user-id", default="8")
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-tsv", type=Path, default=DEFAULT_MANIFEST_TSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--review-seed", type=Path, default=DEFAULT_REVIEW_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_ak_rows(args.metadata_csv, args.speaker_user_id, args.reviewed_only)
    grouped = group_rows(rows)

    seed = load_seed(args.seed)
    seed.update(load_seed(args.review_seed))
    exact, folded = load_dictionary(args.dictionary)

    manifest_rows: list[dict[str, object]] = []
    all_scores: list[float] = []
    totals = {
        "categories": 0,
        "lines": 0,
        "tokens": 0,
        "matched_tokens": 0,
        "high_confidence_tokens": 0,
        "reviewed_lines": 0,
        "needs_review_lines": 0,
    }

    for category_slug, category_rows in sorted(grouped.items(), key=lambda item: (html_unescape(item[1][0].get("category_name", "")), item[0])):
        category_rows = sorted(category_rows, key=sort_key)
        category_name = html_unescape(category_rows[0].get("category_name", "")) or category_slug
        lesson_id = f"ak_cat_{slugify(category_slug)}"
        title = f"A.K. Category - {category_name}"
        source_path = args.source_dir / f"{lesson_id}.tsv"
        source_rows = [source_row(row) for row in category_rows]
        write_source(source_path, source_rows)

        rows_for_interlinear = [
            {
                "id": row["id"],
                "text": row["text"],
                "start_sec": "",
                "end_sec": "",
                "audio_windows": "",
                "audio_posix": row["audio"],
            }
            for row in source_rows
        ]
        reviewed_lines = sum(1 for row in source_rows if row.get("review_status") == "reviewed" and row.get("needs_review") != "1")
        needs_review_lines = len(source_rows) - reviewed_lines
        summary = write_outputs(
            lesson_id,
            title,
            source_path,
            rows_for_interlinear,
            seed,
            exact,
            folded,
            args.output_dir,
            {
                "source_metadata_csv": str(args.metadata_csv),
                "category_slug": category_slug,
                "category_name": category_name,
                "speaker_user_id": args.speaker_user_id,
                "reviewed_only": args.reviewed_only,
                "reviewed_lines": reviewed_lines,
                "needs_review_lines": needs_review_lines,
            },
        )

        totals["categories"] += 1
        totals["lines"] += int(summary["lines"])
        totals["tokens"] += int(summary["tokens"])
        totals["matched_tokens"] += int(summary["matched_tokens"])
        totals["high_confidence_tokens"] += int(summary["high_confidence_tokens"])
        totals["reviewed_lines"] += reviewed_lines
        totals["needs_review_lines"] += needs_review_lines
        all_scores.extend([float(summary["mean_confidence"])] * int(summary["tokens"]))

        manifest_rows.append(
            {
                "lesson_id": lesson_id,
                "title": title,
                "category_slug": category_slug,
                "category_name": category_name,
                "source_tsv": str(source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else source_path),
                "reviewed_lines": reviewed_lines,
                "needs_review_lines": needs_review_lines,
                **summary,
            }
        )

    write_manifest(args.manifest_tsv, manifest_rows)

    aggregate = {
        **totals,
        "matched_pct": f"{(totals['matched_tokens'] / totals['tokens']):.1%}" if totals["tokens"] else "0.0%",
        "high_confidence_pct": f"{(totals['high_confidence_tokens'] / totals['tokens']):.1%}" if totals["tokens"] else "0.0%",
        "mean_confidence": round(statistics.mean(all_scores), 3) if all_scores else 0.0,
        "metadata_csv": str(args.metadata_csv),
        "manifest_tsv": str(args.manifest_tsv),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "reviewed_only": args.reviewed_only,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
