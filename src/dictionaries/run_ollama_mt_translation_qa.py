#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, normalize_search_text, write_json, write_tsv
from build_mt_translation_qa_shards import AUDIT_FIELDS


SCRIPT_DIR = Path(__file__).resolve().parent
REVIEW_ROOT_DEFAULT = SCRIPT_DIR / "translation_qa_20260416"
AUDIT_PATH_DEFAULT = REVIEW_ROOT_DEFAULT / "mt_translation_audit.tsv"
RUNS_DIR_DEFAULT = REVIEW_ROOT_DEFAULT / "ollama_auto_review"
SYSTEM_PROMPT_PATH_DEFAULT = SCRIPT_DIR / "prompts" / "ollama_mt_translation_qa_system.txt"
OLLAMA_EXE_DEFAULT = Path("ollama.exe")
VALID_ACTIONS = {"ok", "fix_en", "fix_de", "fix_both", "skip"}
CONTENT_FIELDS = ["audit_id", "source_dataset", "entry", "definition_tr", "definition_en", "definition_de", "risk_score", "flags"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Ollama MT QA reviews and write review TSV batches.")
    parser.add_argument("--review-root", default=str(REVIEW_ROOT_DEFAULT))
    parser.add_argument("--audit-path", default=str(AUDIT_PATH_DEFAULT))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--ollama-exe", default=str(OLLAMA_EXE_DEFAULT))
    parser.add_argument("--system-prompt-path", default=str(SYSTEM_PROMPT_PATH_DEFAULT))
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-mode", default="false")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--skip-refresh-audit", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--corrected-label", default="ollama_qa_corrected_20260416")
    parser.add_argument("--checked-label", default="ollama_qa_checked_20260416")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def default_run_dir(root: Path, run_name: str) -> Path:
    if clean_cell(run_name):
        return root / "ollama_auto_review" / clean_cell(run_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / "ollama_auto_review" / f"run-{timestamp}"


def refresh_audit(args: argparse.Namespace, run_dir: Path) -> None:
    refresh_dir = run_dir / "_refresh"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_mt_translation_qa_shards.py"),
        "--output-dir",
        str(refresh_dir),
        "--manifest-path",
        str(refresh_dir / "manifest.tsv"),
        "--summary-path",
        str(refresh_dir / "summary.json"),
        "--audit-path",
        str(Path(args.audit_path)),
    ]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)


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


def build_frontier_rows(audit_path: Path, reviewed_ids: set[str], max_rows: int) -> list[dict[str, str]]:
    rows = [row for row in read_tsv(audit_path) if clean_cell(row.get("audit_id")) not in reviewed_ids]
    if max_rows > 0:
        return rows[:max_rows]
    return rows


def chunked(rows: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_user_message(batch_rows: list[dict[str, str]]) -> str:
    rows_payload = [{field: clean_cell(row.get(field)) for field in CONTENT_FIELDS} for row in batch_rows]
    return (
        "Review the following dictionary rows.\n"
        "Return exactly one result object per input row.\n"
        "Keep the same audit_id values.\n"
        "When review_action is ok, leave reviewed_en and reviewed_de empty strings.\n\n"
        f"{json.dumps({'rows': rows_payload}, ensure_ascii=False)}"
    )


def run_ollama_chat(
    *,
    ollama_exe: Path,
    model: str,
    system_prompt: str,
    user_message: str,
    timeout_seconds: int,
    temperature: float,
    think_mode: str,
    keep_alive: str,
) -> tuple[dict[str, Any], str]:
    think_value: bool | str
    if think_mode.lower() == "false":
        think_value = False
    elif think_mode.lower() == "true":
        think_value = True
    else:
        think_value = think_mode
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "format": "json",
        "think": think_value,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
        },
    }
    payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    powershell_command = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        f"$json = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{payload_b64}')); "
        "$resp = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/chat' -Method Post "
        "-ContentType 'application/json' -Body $json; "
        "$resp | ConvertTo-Json -Depth 12"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", powershell_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama request failed: {clean_cell(result.stderr) or clean_cell(result.stdout)}")
    response_payload = json.loads(result.stdout)
    content = str(response_payload.get("message", {}).get("content") or "").strip()
    return response_payload, content


def extract_json_payload(text: str) -> Any:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty model response")
    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    starts = [("{", "}"), ("[", "]")]
    for left, right in starts:
        left_index = cleaned.find(left)
        right_index = cleaned.rfind(right)
        if left_index >= 0 and right_index > left_index:
            fragment = cleaned[left_index : right_index + 1]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"unable to parse JSON from model response: {cleaned[:300]}")


def validate_result_items(batch_rows: list[dict[str, str]], parsed_payload: Any) -> list[dict[str, str]]:
    if isinstance(parsed_payload, dict):
        items = parsed_payload.get("rows")
    else:
        items = parsed_payload
    if not isinstance(items, list):
        raise ValueError("response JSON must contain a rows list")

    expected_ids = [clean_cell(row.get("audit_id")) for row in batch_rows]
    source_rows_by_id = {clean_cell(row.get("audit_id")): row for row in batch_rows}
    items_by_id: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each row result must be an object")
        audit_id = clean_cell(item.get("audit_id"))
        if not audit_id:
            raise ValueError("missing audit_id in row result")
        if audit_id in items_by_id:
            raise ValueError(f"duplicate audit_id in model response: {audit_id}")
        action = clean_cell(item.get("review_action")).lower()
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid review_action for {audit_id}: {action}")
        reviewed_en = clean_cell(item.get("reviewed_en"))
        reviewed_de = clean_cell(item.get("reviewed_de"))
        review_notes = clean_cell(item.get("review_notes"))
        if action in {"fix_en", "fix_both"} and not reviewed_en:
            raise ValueError(f"missing reviewed_en for {audit_id}")
        if action in {"fix_de", "fix_both"} and not reviewed_de:
            raise ValueError(f"missing reviewed_de for {audit_id}")
        source_row = source_rows_by_id.get(audit_id, {})
        flags = {
            token.strip()
            for token in clean_cell(source_row.get("flags")).split("|")
            if token.strip()
        }
        original_en = clean_cell(source_row.get("definition_en"))
        original_de = clean_cell(source_row.get("definition_de"))
        if action == "ok":
            disqualifying_ok = ("en_duplicate_tokens" in flags or "de_duplicate_tokens" in flags) and (
                "en_enumeration_mismatch" in flags
                or "de_enumeration_mismatch" in flags
                or "en_too_short_for_source" in flags
                or "de_too_short_for_source" in flags
                or "en_de_identical" in flags
            )
            if disqualifying_ok:
                raise ValueError(f"ok not allowed for high-risk duplicated row: {audit_id}")
            reviewed_en = ""
            reviewed_de = ""
        if action in {"fix_en", "fix_both"} and normalize_search_text(reviewed_en) == normalize_search_text(original_en):
            raise ValueError(f"reviewed_en unchanged from original for {audit_id}")
        if action in {"fix_de", "fix_both"} and normalize_search_text(reviewed_de) == normalize_search_text(original_de):
            raise ValueError(f"reviewed_de unchanged from original for {audit_id}")
        tr_tokens = set(normalize_search_text(source_row.get("definition_tr", "")).split())
        for field_name, reviewed_text in (("reviewed_en", reviewed_en), ("reviewed_de", reviewed_de)):
            reviewed_tokens = [token for token in normalize_search_text(reviewed_text).split() if token]
            if len(reviewed_tokens) >= 2 and tr_tokens and set(reviewed_tokens).issubset(tr_tokens):
                raise ValueError(f"{field_name} suspiciously copies Turkish source tokens for {audit_id}")
        items_by_id[audit_id] = {
            "audit_id": audit_id,
            "review_action": action,
            "reviewed_en": reviewed_en,
            "reviewed_de": reviewed_de,
            "review_notes": review_notes,
        }

    missing = [audit_id for audit_id in expected_ids if audit_id not in items_by_id]
    extra = [audit_id for audit_id in items_by_id if audit_id not in set(expected_ids)]
    if missing or extra:
        raise ValueError(f"response ids mismatch; missing={missing} extra={extra}")

    return [items_by_id[audit_id] for audit_id in expected_ids]


def merge_reviews(batch_rows: list[dict[str, str]], validated_items: list[dict[str, str]]) -> list[dict[str, str]]:
    merged_rows: list[dict[str, str]] = []
    for source_row, reviewed in zip(batch_rows, validated_items, strict=True):
        row = dict(source_row)
        row["review_action"] = reviewed["review_action"]
        row["reviewed_en"] = reviewed["reviewed_en"]
        row["reviewed_de"] = reviewed["reviewed_de"]
        row["review_notes"] = reviewed["review_notes"]
        merged_rows.append(row)
    return merged_rows


def review_batch(
    *,
    args: argparse.Namespace,
    system_prompt: str,
    batch_rows: list[dict[str, str]],
    request_path: Path,
    response_path: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    base_user_message = build_user_message(batch_rows)
    retry_feedback = ""
    last_error = ""
    raw_response_payload: dict[str, Any] | None = None
    parsed_content = ""
    for attempt in range(args.retry_count + 1):
        user_message = base_user_message
        if retry_feedback:
            user_message = f"{base_user_message}\n\nPrevious response was invalid:\n{retry_feedback}\nReturn corrected JSON only."
        request_payload = {
            "model": args.model,
            "batch_audit_ids": [clean_cell(row.get("audit_id")) for row in batch_rows],
            "attempt": attempt + 1,
            "system_prompt_path": str(Path(args.system_prompt_path)),
            "user_message": user_message,
        }
        write_json(request_path, request_payload)
        try:
            raw_response_payload, parsed_content = run_ollama_chat(
                ollama_exe=Path(args.ollama_exe),
                model=args.model,
                system_prompt=system_prompt,
                user_message=user_message,
                timeout_seconds=args.timeout_seconds,
                temperature=args.temperature,
                think_mode=args.think_mode,
                keep_alive=args.keep_alive,
            )
            write_json(response_path, raw_response_payload)
            parsed_payload = extract_json_payload(parsed_content)
            validated_items = validate_result_items(batch_rows, parsed_payload)
            return merge_reviews(batch_rows, validated_items), raw_response_payload
        except Exception as exc:
            last_error = clean_cell(str(exc))
            retry_feedback = last_error
            error_payload = {
                "error": last_error,
                "raw_content": parsed_content,
                "raw_response_payload": raw_response_payload,
            }
            write_json(response_path, error_payload)
    raise RuntimeError(last_error or "unknown batch review failure")


def run_apply_and_rebuild(args: argparse.Namespace, run_dir: Path) -> None:
    shard_glob = str(run_dir / "batches" / "batch_*.tsv")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "apply_mt_translation_qa_shards.py"),
            "--shard-glob",
            shard_glob,
            "--summary-path",
            str(run_dir / "apply_summary.json"),
            "--corrected-label",
            args.corrected_label,
            "--checked-label",
            args.checked_label,
        ],
        check=True,
        cwd=SCRIPT_DIR,
    )
    if args.rebuild:
        subprocess.run([sys.executable, str(SCRIPT_DIR / "build_lltools_dictionary_import_tsv.py")], check=True, cwd=SCRIPT_DIR)
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_dictionary_snapshot_json.py"),
                "--output-path",
                "zazacaogren-master-dictionary/ll-tools-dictionary-qa-pass-20260416.json",
            ],
            check=True,
            cwd=SCRIPT_DIR,
        )


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone().isoformat()
    run_started = time.perf_counter()
    review_root = Path(args.review_root)
    audit_path = Path(args.audit_path)
    run_dir = Path(args.run_dir) if clean_cell(args.run_dir) else default_run_dir(review_root, args.run_name)
    batches_dir = run_dir / "batches"
    requests_dir = run_dir / "requests"
    responses_dir = run_dir / "responses"
    ensure_parent_dir(batches_dir / "placeholder")
    ensure_parent_dir(requests_dir / "placeholder")
    ensure_parent_dir(responses_dir / "placeholder")

    if not Path(args.ollama_exe).exists():
        raise SystemExit(f"Ollama executable not found: {args.ollama_exe}")

    if not args.skip_refresh_audit:
        refresh_audit(args, run_dir)

    system_prompt = load_system_prompt(Path(args.system_prompt_path))
    reviewed_ids = collect_reviewed_ids(run_dir, audit_path)
    frontier_rows = build_frontier_rows(audit_path, reviewed_ids, args.max_rows)
    all_batches = chunked(frontier_rows, max(1, args.batch_size))
    if args.max_batches > 0:
        all_batches = all_batches[: args.max_batches]

    summary: dict[str, Any] = {
        "model": args.model,
        "ollama_exe": args.ollama_exe,
        "run_dir": str(run_dir),
        "audit_path": str(audit_path),
        "started_at": started_at,
        "frontier_row_count": len(frontier_rows),
        "batch_size": args.batch_size,
        "planned_batch_count": len(all_batches),
        "processed_batch_count": 0,
        "successful_batch_count": 0,
        "failed_batch_count": 0,
        "review_action_counts": {},
        "failures": [],
    }

    write_json(run_dir / "run_config.json", {k: v for k, v in vars(args).items()})

    for batch_index, batch_rows in enumerate(all_batches, start=1):
        batch_id = f"batch_{batch_index:06d}"
        batch_path = batches_dir / f"{batch_id}.tsv"
        request_path = requests_dir / f"{batch_id}.request.json"
        response_path = responses_dir / f"{batch_id}.response.json"
        summary["processed_batch_count"] += 1
        try:
            reviewed_batch, _ = review_batch(
                args=args,
                system_prompt=system_prompt,
                batch_rows=batch_rows,
                request_path=request_path,
                response_path=response_path,
            )
            write_tsv(batch_path, AUDIT_FIELDS, reviewed_batch)
            action_counts = Counter(clean_cell(row.get("review_action")).lower() for row in reviewed_batch)
            running_counts = Counter(summary["review_action_counts"])
            running_counts.update(action_counts)
            summary["review_action_counts"] = dict(running_counts)
            summary["successful_batch_count"] += 1
            print(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "row_count": len(reviewed_batch),
                        "actions": dict(action_counts),
                        "path": str(batch_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            summary["failed_batch_count"] += 1
            summary["failures"].append({"batch_id": batch_id, "error": clean_cell(str(exc))})
            print(json.dumps({"batch_id": batch_id, "error": clean_cell(str(exc))}, ensure_ascii=False), flush=True)
        write_json(run_dir / "summary.json", summary)
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)
    summary["completed_at"] = datetime.now().astimezone().isoformat()
    summary["duration_seconds"] = round(time.perf_counter() - run_started, 3)
    write_json(run_dir / "summary.json", summary)

    if args.apply and summary["successful_batch_count"] > 0:
        run_apply_and_rebuild(args, run_dir)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
