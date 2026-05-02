#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "tsv_output" / "Combined - Harun Turgut + DEZD.tsv"
DEFAULT_INPUTS = [
    SCRIPT_DIR / "tsv_output" / "Palu - Bingöl Harun Turgut - corrected.tsv",
    SCRIPT_DIR / "tsv_output" / "DEZD - Kirmancki Dictionary.tsv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple dictionary TSVs into one combined cross-reference TSV.")
    parser.add_argument(
        "--input-tsv",
        action="append",
        dest="input_tsvs",
        default=[],
        help="Input TSV path. Repeat for multiple files.",
    )
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(item).expanduser().resolve() for item in (args.input_tsvs or [str(path) for path in DEFAULT_INPUTS])]
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    fieldname_set: set[str] = set()
    seen: set[tuple[str, str, str, str]] = set()
    input_counts: dict[str, int] = {}

    for input_path in input_paths:
        count = 0
        with input_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in fieldname_set:
                        fieldname_set.add(name)
                        fieldnames.append(name)
            for row in reader:
                row = dict(row)
                if "source_dictionary" not in row or not str(row.get("source_dictionary") or "").strip():
                    row["source_dictionary"] = input_path.stem
                    if "source_dictionary" not in fieldname_set:
                        fieldname_set.add("source_dictionary")
                        fieldnames.append("source_dictionary")
                key = (
                    str(row.get("entry") or "").casefold(),
                    str(row.get("definition") or "").casefold(),
                    str(row.get("parent") or "").casefold(),
                    str(row.get("source_dictionary") or "").casefold(),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                count += 1
        input_counts[str(input_path)] = count

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output_path": str(output_path),
        "rows_written": len(rows),
        "input_counts": input_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
