#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_PATH = SCRIPT_DIR / "translation_qa_20260416" / "mt_translation_gold_benchmark.tsv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "translation_qa_20260416" / "benchmark_scores"
VALID_ACTIONS = {"ok", "fix_en", "fix_de", "fix_both", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score an Ollama MT QA run against the gold benchmark.")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def normalize_gloss(value: str) -> str:
    text = unicodedata.normalize("NFKC", clean_cell(value)).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,;:/()])\s*", r"\1", text)
    return text.strip(" .;,")


def load_predictions(run_dir: Path) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for path in sorted((run_dir / "batches").glob("*.tsv")):
        for row in read_tsv(path):
            audit_id = clean_cell(row.get("audit_id"))
            action = clean_cell(row.get("review_action")).lower()
            if audit_id and action in VALID_ACTIONS:
                predictions[audit_id] = row
    return predictions


def expected_en_action(action: str) -> str:
    return "fix" if action in {"fix_en", "fix_both"} else "keep"


def expected_de_action(action: str) -> str:
    return "fix" if action in {"fix_de", "fix_both"} else "keep"


def field_match(side: str, gold: dict[str, str], pred: dict[str, str] | None) -> tuple[bool, str]:
    gold_action = clean_cell(gold.get("review_action")).lower()
    expected = expected_en_action(gold_action) if side == "en" else expected_de_action(gold_action)
    gold_value = normalize_gloss(gold.get("reviewed_en" if side == "en" else "reviewed_de"))
    if pred is None:
        return False, "missing_prediction"
    pred_action = clean_cell(pred.get("review_action")).lower()
    pred_value = normalize_gloss(pred.get("reviewed_en" if side == "en" else "reviewed_de"))
    if expected == "keep":
        allowed_actions = {"ok", "fix_de"} if side == "en" else {"ok", "fix_en"}
        if pred_action in allowed_actions and not pred_value:
            return True, ""
        return False, "unexpected_fix_or_output"
    if pred_action not in {"fix_both", f"fix_{side}"}:
        return False, "missing_required_fix"
    if pred_value != gold_value:
        return False, "value_mismatch"
    return True, ""


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_path)
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)

    gold_rows = read_tsv(benchmark_path)
    gold_by_id = {clean_cell(row.get("audit_id")): row for row in gold_rows}
    predictions = load_predictions(run_dir)

    run_summary_path = run_dir / "summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}

    mismatches: list[dict[str, str]] = []
    counters = Counter()
    by_source = defaultdict(Counter)
    by_action = defaultdict(Counter)

    for audit_id, gold in gold_by_id.items():
        counters["gold_rows"] += 1
        source = clean_cell(gold.get("source_dataset")) or "unknown"
        gold_action = clean_cell(gold.get("review_action")).lower()
        pred = predictions.get(audit_id)
        pred_action = clean_cell(pred.get("review_action")).lower() if pred else ""

        action_match = pred is not None and pred_action == gold_action
        en_match, en_reason = field_match("en", gold, pred)
        de_match, de_reason = field_match("de", gold, pred)
        row_exact = action_match and en_match and de_match

        counters["predicted_rows"] += int(pred is not None)
        counters["action_match"] += int(action_match)
        counters["en_match"] += int(en_match)
        counters["de_match"] += int(de_match)
        counters["row_exact"] += int(row_exact)
        by_source[source]["rows"] += 1
        by_source[source]["row_exact"] += int(row_exact)
        by_action[gold_action]["rows"] += 1
        by_action[gold_action]["row_exact"] += int(row_exact)

        if not row_exact:
            mismatches.append(
                {
                    "audit_id": audit_id,
                    "source_dataset": source,
                    "gold_action": gold_action,
                    "pred_action": pred_action or "missing",
                    "en_issue": en_reason,
                    "de_issue": de_reason,
                    "definition_tr": clean_cell(gold.get("definition_tr")),
                    "gold_en": clean_cell(gold.get("reviewed_en")),
                    "pred_en": clean_cell(pred.get("reviewed_en")) if pred else "",
                    "gold_de": clean_cell(gold.get("reviewed_de")),
                    "pred_de": clean_cell(pred.get("reviewed_de")) if pred else "",
                }
            )

    benchmark_name = run_dir.name
    report_dir = output_dir / benchmark_name
    ensure_parent_dir(report_dir / "placeholder")
    mismatch_fields = list(mismatches[0].keys()) if mismatches else [
        "audit_id",
        "source_dataset",
        "gold_action",
        "pred_action",
        "en_issue",
        "de_issue",
        "definition_tr",
        "gold_en",
        "pred_en",
        "gold_de",
        "pred_de",
    ]
    write_tsv(report_dir / "mismatches.tsv", mismatch_fields, mismatches)

    summary = {
        "run_dir": str(run_dir),
        "model": run_summary.get("model", ""),
        "duration_seconds": run_summary.get("duration_seconds"),
        "successful_batch_count": run_summary.get("successful_batch_count"),
        "failed_batch_count": run_summary.get("failed_batch_count"),
        "gold_rows": counters["gold_rows"],
        "predicted_rows": counters["predicted_rows"],
        "action_match_rate": round(counters["action_match"] / max(1, counters["gold_rows"]), 4),
        "en_match_rate": round(counters["en_match"] / max(1, counters["gold_rows"]), 4),
        "de_match_rate": round(counters["de_match"] / max(1, counters["gold_rows"]), 4),
        "row_exact_rate": round(counters["row_exact"] / max(1, counters["gold_rows"]), 4),
        "by_source": {
            key: {
                "rows": value["rows"],
                "row_exact_rate": round(value["row_exact"] / max(1, value["rows"]), 4),
            }
            for key, value in sorted(by_source.items())
        },
        "by_action": {
            key: {
                "rows": value["rows"],
                "row_exact_rate": round(value["row_exact"] / max(1, value["rows"]), 4),
            }
            for key, value in sorted(by_action.items())
        },
        "mismatch_count": len(mismatches),
        "mismatches_path": str(report_dir / "mismatches.tsv"),
    }
    write_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
