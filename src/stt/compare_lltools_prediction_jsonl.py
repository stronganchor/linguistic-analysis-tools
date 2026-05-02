#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from genc_palu_dialect_rules import summarize_genc_palu_dialect_repairs
from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join two LL Tools prediction JSONL files by recording_id and write a comparison JSONL."
    )
    parser.add_argument("--left", required=True, help="First prediction JSONL file.")
    parser.add_argument("--right", required=True, help="Second prediction JSONL file.")
    parser.add_argument("--left-name", default="left", help="Label used for the first model.")
    parser.add_argument("--right-name", default="right", help="Label used for the second model.")
    parser.add_argument("--output-path", required=True, help="Comparison JSONL path.")
    parser.add_argument(
        "--only-differences",
        action="store_true",
        help="Only write rows whose recording_ipa differs between the two inputs.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            recording_id = str(row.get("recording_id") or "")
            if not recording_id:
                continue
            rows[recording_id] = row
    return rows


def main() -> None:
    args = parse_args()
    left_path = Path(args.left).expanduser().resolve()
    right_path = Path(args.right).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    left_rows = load_rows(left_path)
    right_rows = load_rows(right_path)

    common_ids = sorted(set(left_rows) & set(right_rows), key=lambda value: int(value))
    left_only_ids = sorted(set(left_rows) - set(right_rows), key=lambda value: int(value))
    right_only_ids = sorted(set(right_rows) - set(left_rows), key=lambda value: int(value))

    comparison_rows: list[dict] = []
    same_count = 0
    different_count = 0

    for recording_id in common_ids:
        left_row = left_rows[recording_id]
        right_row = right_rows[recording_id]
        left_prediction = str(left_row.get("recording_ipa") or "")
        right_prediction = str(right_row.get("recording_ipa") or "")
        word_title = str(left_row.get("word_title") or right_row.get("word_title") or "")
        word_translation = str(left_row.get("word_translation") or right_row.get("word_translation") or "")
        left_rule_summary = summarize_ipa_rule_issues(left_prediction)
        right_rule_summary = summarize_ipa_rule_issues(right_prediction)
        left_dialect_summary = summarize_genc_palu_dialect_repairs(
            left_prediction,
            word_title=word_title,
            word_translation=word_translation,
        )
        right_dialect_summary = summarize_genc_palu_dialect_repairs(
            right_prediction,
            word_title=word_title,
            word_translation=word_translation,
        )
        exact_match = left_prediction == right_prediction
        if exact_match:
            same_count += 1
        else:
            different_count += 1
        if args.only_differences and exact_match:
            continue

        merged_row = {
            "word_id": left_row.get("word_id") or right_row.get("word_id"),
            "recording_id": left_row.get("recording_id") or right_row.get("recording_id"),
            "word_slug": left_row.get("word_slug") or right_row.get("word_slug"),
            "recording_slug": left_row.get("recording_slug") or right_row.get("recording_slug"),
            "recording_type": left_row.get("recording_type") or right_row.get("recording_type"),
            "speaker_user_id": left_row.get("speaker_user_id") or right_row.get("speaker_user_id"),
            "source_bundle": left_row.get("source_bundle") or right_row.get("source_bundle"),
            "category_slug": left_row.get("category_slug") or right_row.get("category_slug"),
            "category_name": left_row.get("category_name") or right_row.get("category_name"),
            "word_title": word_title,
            "word_translation": word_translation,
            f"{args.left_name}_recording_ipa": left_prediction,
            f"{args.right_name}_recording_ipa": right_prediction,
            f"{args.left_name}_rule_issue_count": int(left_rule_summary["count"]),
            f"{args.left_name}_rule_issue_types": left_rule_summary["types"],
            f"{args.left_name}_rule_repaired": basic_repair_ipa_text(left_prediction),
            f"{args.left_name}_dialect_rule_changed": bool(left_dialect_summary["changed"]),
            f"{args.left_name}_dialect_rule_types": left_dialect_summary["types"],
            f"{args.left_name}_dialect_rule_repaired": left_dialect_summary["repaired"],
            f"{args.right_name}_rule_issue_count": int(right_rule_summary["count"]),
            f"{args.right_name}_rule_issue_types": right_rule_summary["types"],
            f"{args.right_name}_rule_repaired": basic_repair_ipa_text(right_prediction),
            f"{args.right_name}_dialect_rule_changed": bool(right_dialect_summary["changed"]),
            f"{args.right_name}_dialect_rule_types": right_dialect_summary["types"],
            f"{args.right_name}_dialect_rule_repaired": right_dialect_summary["repaired"],
            "exact_match": exact_match,
        }
        comparison_rows.append(merged_row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in comparison_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_name": args.left_name,
        "right_name": args.right_name,
        "output_jsonl": str(output_path),
        "common_recordings": len(common_ids),
        "exact_matches": same_count,
        "differences": different_count,
        "left_only_recordings": len(left_only_ids),
        "right_only_recordings": len(right_only_ids),
        "written_rows": len(comparison_rows),
        "rows_with_any_rule_issues": sum(
            1
            for row in comparison_rows
            if int(row.get(f"{args.left_name}_rule_issue_count") or 0) > 0
            or int(row.get(f"{args.right_name}_rule_issue_count") or 0) > 0
        ),
        "by_category_slug": dict(
            Counter(str(row.get("category_slug") or "") for row in comparison_rows)
        ),
        "difference_by_category_slug": dict(
            Counter(
                str(row.get("category_slug") or "")
                for row in comparison_rows
                if not row.get("exact_match")
            )
        ),
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote comparison JSONL: {output_path}")
    print(f"Wrote summary:          {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
