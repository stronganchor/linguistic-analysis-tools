#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified review sheet with four model columns for unresolved recordings."
    )
    parser.add_argument("--input-tsv", required=True, help="Base unresolved-recordings TSV.")
    parser.add_argument("--whisper-ipa-jsonl", required=True, help="Full-bundle Whisper IPA second-opinion JSONL.")
    parser.add_argument("--mms-ipa-tsv", required=True, help="MMS IPA prediction TSV for missing-IPA rows.")
    parser.add_argument("--mms-orth-tsv", required=True, help="Orthography MMS prediction TSV.")
    parser.add_argument("--whisper-orth-tsv", required=True, help="Orthography Whisper prediction TSV.")
    parser.add_argument("--output-prefix", required=True, help="Output prefix for merged files.")
    return parser.parse_args()


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_index(rows: list[dict], key: str) -> dict[str, dict]:
    return {str(row[key]).strip(): row for row in rows if str(row.get(key) or "").strip()}


def main() -> None:
    args = parse_args()
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    whisper_ipa_jsonl = Path(args.whisper_ipa_jsonl).expanduser().resolve()
    mms_ipa_tsv = Path(args.mms_ipa_tsv).expanduser().resolve()
    mms_orth_tsv = Path(args.mms_orth_tsv).expanduser().resolve()
    whisper_orth_tsv = Path(args.whisper_orth_tsv).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    base_rows = load_tsv(input_tsv)
    whisper_ipa_rows = load_jsonl(whisper_ipa_jsonl)
    mms_ipa_rows = load_tsv(mms_ipa_tsv)
    mms_orth_rows = load_tsv(mms_orth_tsv)
    whisper_orth_rows = load_tsv(whisper_orth_tsv)

    whisper_ipa_index = build_index(whisper_ipa_rows, "recording_id")
    mms_ipa_index = build_index(mms_ipa_rows, "recording_id")
    mms_orth_index = build_index(mms_orth_rows, "recording_id")
    whisper_orth_index = build_index(whisper_orth_rows, "recording_id")

    merged_rows: list[dict[str, str]] = []
    for row in base_rows:
        rid = str(row["recording_id"]).strip()
        whisper_ipa = whisper_ipa_index.get(rid, {})
        mms_ipa = mms_ipa_index.get(rid, {})
        mms_orth = mms_orth_index.get(rid, {})
        whisper_orth = whisper_orth_index.get(rid, {})

        current_ipa = str(row.get("recording_ipa") or "").strip()
        if current_ipa:
            mms_ipa_value = current_ipa
            mms_ipa_source = "current_metadata_recording_ipa"
        else:
            mms_ipa_value = str(mms_ipa.get("predicted_ipa") or "").strip()
            mms_ipa_source = "new_mms_v003_cpu_batch" if mms_ipa_value else ""

        merged = dict(row)
        merged["mms_ipa_v003"] = mms_ipa_value
        merged["mms_ipa_source"] = mms_ipa_source
        merged["whisper_medium_ipa_v005_1254"] = str(
            whisper_ipa.get("ipa_rule_repaired")
            or whisper_ipa.get("predicted_text")
            or whisper_ipa.get("recording_ipa")
            or ""
        ).strip()
        merged["whisper_medium_ipa_issue_types"] = "|".join(whisper_ipa.get("ipa_rule_issue_types") or [])
        merged["mms_orth_simple_1282"] = str(mms_orth.get("predicted_text") or "").strip()
        merged["whisper_medium_orth_2138"] = str(whisper_orth.get("predicted_text") or "").strip()
        merged_rows.append(merged)

    base_fieldnames = list(base_rows[0].keys()) if base_rows else []
    model_fieldnames = [
        "mms_ipa_v003",
        "mms_ipa_source",
        "whisper_medium_ipa_v005_1254",
        "whisper_medium_ipa_issue_types",
        "mms_orth_simple_1282",
        "whisper_medium_orth_2138",
    ]
    fieldnames = base_fieldnames + [name for name in model_fieldnames if name not in base_fieldnames]

    tsv_path = output_prefix.with_suffix(".tsv")
    jsonl_path = output_prefix.with_suffix(".jsonl")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": len(base_rows),
        "output_rows": len(merged_rows),
        "rows_with_mms_ipa": sum(1 for row in merged_rows if row["mms_ipa_v003"]),
        "rows_with_whisper_ipa": sum(1 for row in merged_rows if row["whisper_medium_ipa_v005_1254"]),
        "rows_with_mms_orth": sum(1 for row in merged_rows if row["mms_orth_simple_1282"]),
        "rows_with_whisper_orth": sum(1 for row in merged_rows if row["whisper_medium_orth_2138"]),
        "outputs": {
            "tsv": str(tsv_path),
            "jsonl": str(jsonl_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
