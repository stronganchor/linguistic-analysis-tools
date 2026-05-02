#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, write_json


SCRIPT_DIR = Path(__file__).resolve().parent
REVIEW_ROOT_DEFAULT = SCRIPT_DIR / "translation_qa_20260416"
RUNS_DIR_DEFAULT = REVIEW_ROOT_DEFAULT / "deepl_auto_review"
AUDIT_PATH_DEFAULT = REVIEW_ROOT_DEFAULT / "difficulty_buckets" / "non_hard.tsv"
VALID_ACTIONS = {"ok", "fix_en", "fix_de", "fix_both", "skip"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepL bulk EN/DE regeneration from Turkish definitions and write review TSV batches.")
    parser.add_argument("--audit-path", default=str(AUDIT_PATH_DEFAULT))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--corrected-label", default="deepl_bulk_corrected_20260417")
    parser.add_argument("--checked-label", default="deepl_bulk_checked_20260417")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{name: clean_cell(row.get(name, "")) for name in fieldnames} for row in rows])


def default_run_dir(root: Path, run_name: str) -> Path:
    if clean_cell(run_name):
        return root / clean_cell(run_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"deepl-run-{timestamp}"


def windows_env_value(name: str) -> str:
    result = subprocess.run(
        ["cmd.exe", "/c", f"echo %{name}%"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    value = clean_cell(result.stdout)
    if not value or value == f"%{name}%":
        return ""
    return value


def resolve_api_key(explicit: str) -> str:
    if clean_cell(explicit):
        return clean_cell(explicit)
    for name in ("DEEPL_API_KEY", "DEEPL_AUTH_KEY"):
        value = clean_cell(os.environ.get(name))
        if value:
            return value
    for name in ("DEEPL_API_KEY", "DEEPL_AUTH_KEY"):
        value = windows_env_value(name)
        if value:
            return value
    raise RuntimeError("No DeepL API key found in process or Windows environment")


def resolve_endpoint(explicit: str, api_key: str) -> str:
    if clean_cell(explicit):
        return clean_cell(explicit)
    if api_key.endswith(":fx"):
        return "https://api-free.deepl.com/v2/translate"
    return "https://api.deepl.com/v2/translate"


def collect_reviewed_ids(run_dir: Path, audit_path: Path) -> set[str]:
    reviewed_ids: set[str] = set()
    if not run_dir.exists():
        return reviewed_ids
    for path in run_dir.rglob("*.tsv"):
        if path == audit_path:
            continue
        try:
            rows = read_tsv(path)
        except Exception:
            continue
        if not rows or "audit_id" not in rows[0]:
            continue
        for row in rows:
            action = clean_cell(row.get("review_action")).lower()
            if action in VALID_ACTIONS:
                audit_id = clean_cell(row.get("audit_id"))
                if audit_id:
                    reviewed_ids.add(audit_id)
    return reviewed_ids


def chunked(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def deepl_translate(
    *,
    api_key: str,
    endpoint: str,
    texts: list[str],
    target_lang: str,
    timeout_seconds: int,
) -> list[str]:
    if not texts:
        return []
    data = parse.urlencode(
        [
            ("source_lang", "TR"),
            ("target_lang", target_lang),
            ("preserve_formatting", "1"),
            ("split_sentences", "nonewlines"),
            *[("text", text) for text in texts],
        ],
        doseq=True,
    ).encode("utf-8")
    req = request.Request(endpoint, data=data, method="POST")
    req.add_header("Authorization", f"DeepL-Auth-Key {api_key}")
    with request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    translations = payload.get("translations") or []
    if len(translations) != len(texts):
        raise RuntimeError(f"DeepL returned {len(translations)} translations for {len(texts)} inputs")
    return [clean_cell(item.get("text")) for item in translations]


def apply_run(args: argparse.Namespace, run_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "apply_mt_translation_qa_shards.py"),
        "--shard-glob",
        str(run_dir / "batches" / "*.tsv"),
        "--summary-path",
        str(run_dir / "apply_summary.json"),
        "--backup-suffix",
        ".pre_deepl_bulk_apply_20260417.bak",
        "--corrected-label",
        args.corrected_label,
        "--checked-label",
        args.checked_label,
    ]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


def rebuild_outputs() -> None:
    subprocess.run([sys.executable, str(SCRIPT_DIR / "build_lltools_dictionary_import_tsv.py")], check=True, cwd=SCRIPT_DIR)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_dictionary_snapshot_json.py"),
            "--input-path",
            str(SCRIPT_DIR / "tsv_output" / "lltools_import" / "lltools_dictionary_combined_public.tsv"),
            "--output-path",
            str(SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-qa-pass-20260416.json"),
        ],
        check=True,
        cwd=SCRIPT_DIR,
    )


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_path)
    run_dir = Path(args.run_dir) if clean_cell(args.run_dir) else default_run_dir(RUNS_DIR_DEFAULT, args.run_name)
    ensure_parent_dir(run_dir / "placeholder")
    batches_dir = run_dir / "batches"
    responses_dir = run_dir / "responses"
    ensure_parent_dir(batches_dir / "placeholder")
    ensure_parent_dir(responses_dir / "placeholder")

    api_key = resolve_api_key(args.api_key)
    endpoint = resolve_endpoint(args.endpoint, api_key)

    reviewed_ids = collect_reviewed_ids(run_dir, audit_path)
    all_rows = [row for row in read_tsv(audit_path) if clean_cell(row.get("audit_id")) not in reviewed_ids]
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]
    batches = chunked(all_rows, args.batch_size)
    if args.max_batches > 0:
        batches = batches[: args.max_batches]

    started_at = datetime.now(timezone.utc).isoformat()
    stats = Counter()
    failures: list[dict[str, str]] = []

    for batch_index, batch_rows in enumerate(batches, start=1):
        batch_id = f"batch_{batch_index:06d}"
        tr_texts = [clean_cell(row.get("definition_tr")) for row in batch_rows]
        try:
            en_texts = deepl_translate(
                api_key=api_key,
                endpoint=endpoint,
                texts=tr_texts,
                target_lang="EN",
                timeout_seconds=args.timeout_seconds,
            )
            de_texts = deepl_translate(
                api_key=api_key,
                endpoint=endpoint,
                texts=tr_texts,
                target_lang="DE",
                timeout_seconds=args.timeout_seconds,
            )
            output_rows: list[dict[str, str]] = []
            for row, reviewed_en, reviewed_de in zip(batch_rows, en_texts, de_texts, strict=True):
                output_rows.append(
                    {
                        **row,
                        "review_action": "fix_both",
                        "reviewed_en": reviewed_en,
                        "reviewed_de": reviewed_de,
                        "review_notes": "DeepL bulk refresh from Turkish source on non-hard row",
                    }
                )
            fieldnames = list(output_rows[0].keys()) if output_rows else []
            write_tsv(batches_dir / f"{batch_id}.tsv", fieldnames, output_rows)
            write_json(
                responses_dir / f"{batch_id}.json",
                {
                    "batch_id": batch_id,
                    "row_count": len(batch_rows),
                    "endpoint": endpoint,
                    "target_langs": ["EN", "DE"],
                    "char_count": sum(len(text) for text in tr_texts),
                },
            )
            stats["processed_rows"] += len(batch_rows)
            stats["successful_batches"] += 1
            print(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "row_count": len(batch_rows),
                        "path": str(batches_dir / f"{batch_id}.tsv"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            failures.append({"batch_id": batch_id, "error": str(exc)})
            stats["failed_batches"] += 1
            print(json.dumps({"batch_id": batch_id, "error": str(exc)}, ensure_ascii=False), flush=True)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    summary = {
        "api_endpoint": endpoint,
        "run_dir": str(run_dir),
        "audit_path": str(audit_path),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frontier_row_count": len(all_rows),
        "batch_size": args.batch_size,
        "planned_batch_count": len(batches),
        "processed_rows": stats["processed_rows"],
        "successful_batch_count": stats["successful_batches"],
        "failed_batch_count": stats["failed_batches"],
        "failures": failures,
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.apply and stats["successful_batches"] > 0:
        apply_run(args, run_dir)
    if args.rebuild and stats["successful_batches"] > 0:
        rebuild_outputs()


if __name__ == "__main__":
    main()
