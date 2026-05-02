#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TSV_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418" / "translation_retranslation_candidates.tsv"
OUTPUT_DIR_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "issue_flag_pass_20260418" / "retranslation_round_01"

ROUND1_ROWS = [
    ("dict:caf7d49b775f317e1cd5fe8069517e30", "0"),
    ("dict:8adf95ff3a8622c99ba9e4dd59e0391e", "0"),
    ("dict:e9c495ba9c161e50f701f5f26c06e18b", "0"),
    ("dict:19bb515883e745b9329acf855706ac73", "0"),
    ("dict:8a4f1d7a25fba9ee081f01902dc15790", "0"),
    ("dict:42374d67ecb474c55c6d5aef2d7d7638", "0"),
    ("dict:21916f3f7b4e2dd763320b4836a4a9d5", "0"),
    ("dict:571cc6144943b953b23e58bdc2ea3dff", "0"),
    ("dict:7fb4cab12fdac08280d4a5bfc8ce563a", "0"),
    ("dict:427a816caa06f4d8aec4fafa9ae153bd", "0"),
    ("dict:1e08411c69a747e5a1995cd1dc539b0f", "0"),
    ("dict:8a8a0c91d602830e49822ef5ecf9fd8d", "0"),
    ("dict:a80b3e7c30b508b5d9de3fb1a4f7e111", "0"),
    ("dict:2bcf09137d3948507fa39519d0dfc9b2", "0"),
    ("dict:8faf4ba8a486d28f561880756de1fa8b", "0"),
    ("dict:e6e3dd747fe2097986e778d2b84d742b", "0"),
    ("dict:302ad89cb77b675957f80bf285b39d5a", "0"),
    ("dict:5e30ec30b307f419e75aae3738d5c569", "0"),
    ("dict:2c2e06fb755ac9dc71fbb6ae50b1dffe", "0"),
    ("dict:58d505166380a5e74c4330ecdf4c1fd6", "0"),
    ("dict:71a6a01f8b03cd6ce2fbb1d500e5b5d7", "0"),
    ("dict:fe1e3a8925f9e8eaa5648ed7c7ec27bc", "0"),
    ("dict:a00f5c9ead346d25512d74f15258b199", "0"),
    ("dict:8154f3900887eb58319085cccbc6674b", "0"),
    ("dict:3e8e7e5126a1e745612b8e40377b3f8b", "0"),
    ("dict:3089619d551538d40ad1f3e2a885bd4e", "0"),
    ("dict:ea21198cde9ee5b0c052a711868c5da1", "0"),
    ("dict:13f00a4c16d4730b86d79d6d49ed3ef1", "0"),
    ("dict:6bc109e91b0714667eebac6cb1fae272", "0"),
    ("dict:e70bd6f1d5c94b0b04047c8e51397f40", "0"),
]

SHARD_FIELDS = [
    "round_id",
    "shard_id",
    "row_id",
    "issue_type",
    "priority",
    "confidence_score",
    "import_key",
    "sense_index",
    "source_id",
    "source_row_idx",
    "title",
    "definition_tr",
    "definition_en",
    "definition_de",
    "suggested_en_seed",
    "suggested_de_seed",
    "heuristic_flags",
    "reason",
    "review_action",
    "reviewed_en",
    "reviewed_de",
    "review_notes",
]

MANIFEST_FIELDS = ["shard_id", "path", "row_count", "titles"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round-1 retranslation shards from high-confidence flagged rows.")
    parser.add_argument("--input-path", default=str(INPUT_TSV_DEFAULT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR_DEFAULT))
    parser.add_argument("--shard-count", type=int, default=10)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def chunk_rows(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    if not rows:
        return []
    shard_count = max(1, min(shard_count, len(rows)))
    shard_size = math.ceil(len(rows) / shard_count)
    return [rows[index : index + shard_size] for index in range(0, len(rows), shard_size)]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    rows = read_tsv(input_path)
    by_id = {(clean_cell(row.get("import_key")), clean_cell(row.get("sense_index"))): row for row in rows}

    selected_rows: list[dict[str, str]] = []
    missing: list[str] = []
    for import_key, sense_index in ROUND1_ROWS:
        row = by_id.get((import_key, sense_index))
        if not row:
            missing.append(f"{import_key}:{sense_index}")
            continue
        selected_rows.append(row)

    round_rows: list[dict[str, str]] = []
    for index, row in enumerate(selected_rows, start=1):
        round_rows.append(
            {
                "round_id": "round_01",
                "shard_id": "",
                "row_id": f"round01_{index:03d}",
                "issue_type": clean_cell(row.get("issue_type")),
                "priority": clean_cell(row.get("priority")),
                "confidence_score": clean_cell(row.get("confidence_score")),
                "import_key": clean_cell(row.get("import_key")),
                "sense_index": clean_cell(row.get("sense_index")),
                "source_id": clean_cell(row.get("source_id")),
                "source_row_idx": clean_cell(row.get("source_row_idx")),
                "title": clean_cell(row.get("title")),
                "definition_tr": clean_cell(row.get("definition_tr")),
                "definition_en": clean_cell(row.get("definition_en")),
                "definition_de": clean_cell(row.get("definition_de")),
                "suggested_en_seed": clean_cell(row.get("suggested_en")),
                "suggested_de_seed": clean_cell(row.get("suggested_de")),
                "heuristic_flags": clean_cell(row.get("heuristic_flags")),
                "reason": clean_cell(row.get("reason")),
                "review_action": "",
                "reviewed_en": "",
                "reviewed_de": "",
                "review_notes": "",
            }
        )

    shard_groups = chunk_rows(round_rows, args.shard_count)
    ensure_parent_dir(output_dir / "placeholder")
    manifest_rows: list[dict[str, str]] = []
    for shard_index, shard_rows in enumerate(shard_groups, start=1):
        shard_id = f"rt01_{shard_index:02d}"
        path = output_dir / "shards" / f"{shard_id}.tsv"
        for row in shard_rows:
            row["shard_id"] = shard_id
        write_tsv(path, SHARD_FIELDS, shard_rows)
        manifest_rows.append(
            {
                "shard_id": shard_id,
                "path": str(path),
                "row_count": str(len(shard_rows)),
                "titles": " | ".join(clean_cell(row.get("title")) for row in shard_rows),
            }
        )

    manifest_path = output_dir / "shard_manifest.tsv"
    write_tsv(manifest_path, MANIFEST_FIELDS, manifest_rows)
    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "selected_row_count": len(round_rows),
        "shard_count": len(shard_groups),
        "missing_ids": missing,
        "manifest_path": str(manifest_path),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
