#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
REVIEW_ROOT_DEFAULT = SCRIPT_DIR / "translation_qa_20260416"
OUTPUT_PATH_DEFAULT = REVIEW_ROOT_DEFAULT / "mt_translation_gold_benchmark.tsv"
SUMMARY_PATH_DEFAULT = REVIEW_ROOT_DEFAULT / "mt_translation_gold_benchmark_summary.json"
VALID_ACTIONS = {"ok", "fix_en", "fix_de", "fix_both"}
EXCLUDED_PARTS = {"ollama_auto_review", "shards", "_refresh", "tmp_full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stable gold benchmark from manually reviewed MT QA rows.")
    parser.add_argument("--review-root", default=str(REVIEW_ROOT_DEFAULT))
    parser.add_argument("--output-path", default=str(OUTPUT_PATH_DEFAULT))
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH_DEFAULT))
    parser.add_argument("--corrected-per-source", type=int, default=12)
    parser.add_argument("--ok-per-source", type=int, default=3)
    parser.add_argument("--max-definition-tr-chars", type=int, default=0)
    parser.add_argument("--max-definition-en-chars", type=int, default=0)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def review_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*.tsv"):
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        if path.name == OUTPUT_PATH_DEFAULT.name:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: (item.stat().st_mtime, str(item)))


def parse_risk(row: dict[str, str]) -> int:
    value = clean_cell(row.get("risk_score"))
    try:
        return int(value)
    except ValueError:
        return 0


def row_sort_key(row: dict[str, str]) -> tuple[int, str, str]:
    return (-parse_risk(row), clean_cell(row.get("audit_id")), clean_cell(row.get("definition_tr")))


def within_length_limits(row: dict[str, str], args: argparse.Namespace) -> bool:
    definition_tr = clean_cell(row.get("definition_tr"))
    definition_en = clean_cell(row.get("definition_en"))
    if args.max_definition_tr_chars > 0 and len(definition_tr) > args.max_definition_tr_chars:
        return False
    if args.max_definition_en_chars > 0 and len(definition_en) > args.max_definition_en_chars:
        return False
    return True


def load_gold_rows(root: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    latest_by_id: dict[str, dict[str, str]] = {}
    source_path_by_id: dict[str, str] = {}
    for path in review_paths(root):
        rows = read_tsv(path)
        if not rows or "review_action" not in rows[0]:
            continue
        for row in rows:
            audit_id = clean_cell(row.get("audit_id"))
            action = clean_cell(row.get("review_action")).lower()
            if not audit_id or action not in VALID_ACTIONS:
                continue
            if not within_length_limits(row, args):
                continue
            tagged = dict(row)
            tagged["gold_source_path"] = str(path)
            latest_by_id[audit_id] = tagged
            source_path_by_id[audit_id] = str(path)
    gold_rows = list(latest_by_id.values())
    for row in gold_rows:
        row["gold_source_path"] = source_path_by_id[clean_cell(row.get("audit_id"))]
    return gold_rows


def choose_rows(rows: list[dict[str, str]], corrected_per_source: int, ok_per_source: int) -> list[dict[str, str]]:
    selected_ids: set[str] = set()
    selected: list[dict[str, str]] = []
    by_source_and_bucket: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        source = clean_cell(row.get("source_dataset")) or "unknown"
        action = clean_cell(row.get("review_action")).lower()
        bucket = "ok" if action == "ok" else "corrected"
        by_source_and_bucket[(source, bucket)].append(row)

    for rows_for_bucket in by_source_and_bucket.values():
        rows_for_bucket.sort(key=row_sort_key)

    for source in sorted({clean_cell(row.get("source_dataset")) or "unknown" for row in rows}):
        for row in by_source_and_bucket.get((source, "corrected"), [])[: max(0, corrected_per_source)]:
            audit_id = clean_cell(row.get("audit_id"))
            if audit_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(audit_id)
        for row in by_source_and_bucket.get((source, "ok"), [])[: max(0, ok_per_source)]:
            audit_id = clean_cell(row.get("audit_id"))
            if audit_id in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(audit_id)

    target_total = max(0, corrected_per_source) * 2 + max(0, ok_per_source) * 2
    if len(selected) < target_total:
        remaining = [row for row in sorted(rows, key=row_sort_key) if clean_cell(row.get("audit_id")) not in selected_ids]
        for row in remaining[: target_total - len(selected)]:
            selected.append(row)
            selected_ids.add(clean_cell(row.get("audit_id")))

    return sorted(selected, key=lambda row: (clean_cell(row.get("source_dataset")), row_sort_key(row)))


def main() -> None:
    args = parse_args()
    review_root = Path(args.review_root)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    gold_rows = load_gold_rows(review_root, args)
    chosen_rows = choose_rows(gold_rows, args.corrected_per_source, args.ok_per_source)

    fieldnames = list(chosen_rows[0].keys()) if chosen_rows else []
    ensure_parent_dir(output_path)
    write_tsv(output_path, fieldnames, chosen_rows)

    action_counts = Counter(clean_cell(row.get("review_action")).lower() for row in chosen_rows)
    source_counts = Counter(clean_cell(row.get("source_dataset")) for row in chosen_rows)
    summary = {
        "review_root": str(review_root),
        "gold_row_count": len(gold_rows),
        "benchmark_row_count": len(chosen_rows),
        "action_counts": dict(action_counts),
        "source_counts": dict(source_counts),
        "max_definition_tr_chars": args.max_definition_tr_chars,
        "max_definition_en_chars": args.max_definition_en_chars,
        "output_path": str(output_path),
    }
    ensure_parent_dir(summary_path)
    write_json(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
