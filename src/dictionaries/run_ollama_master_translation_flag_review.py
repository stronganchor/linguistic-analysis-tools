#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from _dictionary_pipeline_common import clean_cell, ensure_parent_dir, normalize_search_text, write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "ll-tools-dictionary-reviewed-only-20260417.json"
OUTPUT_ROOT_DEFAULT = SCRIPT_DIR / "zazacaogren-master-dictionary" / "translation_flag_review"
SYSTEM_PROMPT_PATH_DEFAULT = SCRIPT_DIR / "prompts" / "ollama_master_translation_flag_system.txt"
OLLAMA_EXE_DEFAULT = Path("ollama.exe")
PILOT_SAMPLE_DIR_DEFAULT = (
    SCRIPT_DIR / "zazacaogren-master-dictionary" / "audits" / "translation_random_sample_20260417"
)

VALID_STATUSES = {"ok", "needs_review"}
VALID_LANGS = {"en", "de"}
RESULT_FIELDS = [
    "audit_id",
    "entry_index",
    "sense_index",
    "import_key",
    "title",
    "source_id",
    "source_row_idx",
    "page_number",
    "entry_type",
    "definition_tr",
    "definition_en",
    "definition_de",
    "risk_score",
    "flags",
    "review_status",
    "bad_langs",
    "review_notes",
]
CONTENT_FIELDS = [
    "audit_id",
    "title",
    "entry_type",
    "source_id",
    "definition_tr",
    "definition_en",
    "definition_de",
    "risk_score",
    "flags",
]
USER_MESSAGE_PREFIX = (
    "Review the following dictionary rows.\n"
    "Flags are hints, not ground truth.\n"
    "Return exactly one result object per input row.\n"
    "Keep the same audit_id values.\n\n"
)

TURKISH_FUNCTION_WORDS = {
    "ve",
    "veya",
    "ile",
    "icin",
    "için",
    "olan",
    "olmak",
    "etmek",
    "gibi",
    "çok",
    "cok",
    "bir",
    "bu",
    "şey",
    "sey",
    "kendi",
    "üstüne",
    "ustune",
    "altına",
    "altina",
    "önce",
    "sonra",
    "daha",
    "kadar",
    "değil",
    "degil",
    "arası",
    "arasi",
    "yapmak",
    "yapın",
    "yapin",
}
ENGLISH_PRONOUNS = {"i", "you", "we", "they", "he", "she", "me", "him", "her", "them", "my", "your", "our"}
GERMAN_PRONOUNS = {"ich", "du", "er", "sie", "wir", "ihr", "ihnen", "mich", "dich", "uns", "euch", "mein", "dein"}
PLACEHOLDERISH_TOKENS = {"inftira", "sweepen", "node", "php", "konstant"}
TOKEN_RE = __import__("re").compile(r"[0-9A-Za-zÀ-ÿÇĞİÖŞÜçğıöşüêîûâ]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Gemma/Ollama translation flag review over the master dictionary snapshot."
    )
    parser.add_argument("--snapshot-path", default=str(SNAPSHOT_PATH_DEFAULT))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--model", default="gemma3:27b")
    parser.add_argument("--ollama-exe", default=str(OLLAMA_EXE_DEFAULT))
    parser.add_argument("--system-prompt-path", default=str(SYSTEM_PROMPT_PATH_DEFAULT))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-user-message-chars", type=int, default=3200)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--retry-count", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--think-mode", default="false")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--pilot-sample-dir", default=str(PILOT_SAMPLE_DIR_DEFAULT))
    parser.add_argument("--pilot-size", type=int, default=48)
    parser.add_argument("--pilot-bad-count", type=int, default=16)
    parser.add_argument("--pilot-seed", type=int, default=20260417)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--annotated-snapshot-path", default="")
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def default_run_dir(root: Path, run_name: str) -> Path:
    if clean_cell(run_name):
        return root / clean_cell(run_name)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"run-{timestamp}"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_cell(text).lower())


def normalized_tokens(text: str) -> list[str]:
    return [token for token in normalize_search_text(text).split() if token]


def duplicate_token_flag(text: str) -> bool:
    tokens = tokenize(text)
    if len(tokens) < 2:
        return False
    for left, right in zip(tokens, tokens[1:], strict=False):
        if left == right:
            return True
    return len(tokens) != len(set(tokens)) and len(tokens) <= 5


def suspicious_tr_overlap(target_text: str) -> bool:
    tokens = set(normalized_tokens(target_text))
    return bool(tokens & TURKISH_FUNCTION_WORDS)


def enumeration_mismatch(tr: str, target: str) -> bool:
    tr_count = tr.count("2)") + tr.count("3)") + tr.count("4)") + tr.count("5)")
    target_count = target.count("2)") + target.count("3)") + target.count("4)") + target.count("5)")
    return tr_count > target_count


def suspicious_length(tr: str, target: str) -> bool:
    tr_len = len(tokenize(tr))
    target_len = len(tokenize(target))
    if tr_len >= 6 and target_len <= 2:
        return True
    if tr_len >= 10 and target_len <= 4:
        return True
    return False


def phrase_mode_mismatch(tr: str, target: str, pronouns: set[str]) -> bool:
    tr_tokens = normalized_tokens(tr)
    target_tokens = tokenize(target)
    if not target_tokens:
        return False
    if any(token in pronouns for token in target_tokens[:2]):
        return True
    if tr_tokens and tr_tokens[0] in {"ben", "sen", "siz", "o", "biz", "onlar"}:
        return False
    if clean_cell(target).endswith("."):
        return True
    return False


def placeholder_flag(text: str) -> bool:
    tokens = set(tokenize(text))
    return bool(tokens & PLACEHOLDERISH_TOKENS)


def unusual_case_flag(text: str) -> bool:
    cleaned = clean_cell(text)
    return bool(cleaned) and cleaned[0].isupper() and cleaned.count(" ") == 0


def build_flags(tr: str, en: str, de: str) -> tuple[int, list[str]]:
    flags: list[str] = []
    score = 0
    if duplicate_token_flag(en):
        flags.append("en_duplicate_tokens")
        score += 4
    if duplicate_token_flag(de):
        flags.append("de_duplicate_tokens")
        score += 4
    if suspicious_tr_overlap(en):
        flags.append("en_turkish_lexeme_leak")
        score += 3
    if suspicious_tr_overlap(de):
        flags.append("de_turkish_lexeme_leak")
        score += 3
    if enumeration_mismatch(tr, en):
        flags.append("en_enumeration_mismatch")
        score += 2
    if enumeration_mismatch(tr, de):
        flags.append("de_enumeration_mismatch")
        score += 2
    if suspicious_length(tr, en):
        flags.append("en_too_short_for_source")
        score += 3
    if suspicious_length(tr, de):
        flags.append("de_too_short_for_source")
        score += 3
    if phrase_mode_mismatch(tr, en, ENGLISH_PRONOUNS):
        flags.append("en_sentence_mode")
        score += 2
    if phrase_mode_mismatch(tr, de, GERMAN_PRONOUNS):
        flags.append("de_sentence_mode")
        score += 2
    if placeholder_flag(en):
        flags.append("en_placeholder_or_bad_lexeme")
        score += 5
    if placeholder_flag(de):
        flags.append("de_placeholder_or_bad_lexeme")
        score += 5
    if unusual_case_flag(en):
        flags.append("en_single_capitalized_token")
        score += 1
    if unusual_case_flag(de):
        flags.append("de_single_capitalized_token")
        score += 1
    if len(tokenize(tr)) >= 12 and ("..." in en or "..." in de):
        flags.append("ellipsis_truncation")
        score += 4
    if en and de and normalize_search_text(en) == normalize_search_text(de):
        flags.append("en_de_identical")
        score += 2
    return score, flags


def load_snapshot_rows(snapshot_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") or []
    rows: list[dict[str, str]] = []
    for entry_index, entry in enumerate(entries):
        import_key = clean_cell(entry.get("import_key"))
        title = clean_cell(entry.get("title"))
        senses = entry.get("senses") or []
        for sense_index, sense in enumerate(senses):
            tr = clean_cell((sense.get("translations") or {}).get("tr")) or clean_cell(sense.get("definition"))
            en = clean_cell((sense.get("translations") or {}).get("en"))
            de = clean_cell((sense.get("translations") or {}).get("de"))
            risk_score, flags = build_flags(tr, en, de)
            rows.append(
                {
                    "audit_id": f"{import_key}:{sense_index}",
                    "entry_index": str(entry_index),
                    "sense_index": str(sense_index),
                    "import_key": import_key,
                    "title": title,
                    "source_id": clean_cell(sense.get("source_id")),
                    "source_row_idx": clean_cell(sense.get("source_row_idx")),
                    "page_number": clean_cell(sense.get("page_number")),
                    "entry_type": clean_cell(sense.get("entry_type")),
                    "definition_tr": tr,
                    "definition_en": en,
                    "definition_de": de,
                    "risk_score": str(risk_score),
                    "flags": " | ".join(flags),
                }
            )
    return payload, rows


def load_sample_rows(sample_dir: Path) -> list[dict[str, str]]:
    batch_paths = sorted((sample_dir / "batches").glob("batch_*.json"))
    bad_entries = json.loads((sample_dir / "bad_entries.json").read_text(encoding="utf-8"))
    bad_ids = {
        f"{clean_cell(item.get('import_key'))}:{clean_cell(item.get('sense_index'))}"
        for item in bad_entries
    }
    rows: list[dict[str, str]] = []
    for batch_path in batch_paths:
        batch_rows = json.loads(batch_path.read_text(encoding="utf-8"))
        for item in batch_rows:
            import_key = clean_cell(item.get("import_key"))
            sense_index = clean_cell(item.get("sense_index"))
            audit_id = f"{import_key}:{sense_index}"
            tr = clean_cell(item.get("tr"))
            en = clean_cell(item.get("en"))
            de = clean_cell(item.get("de"))
            risk_score, flags = build_flags(tr, en, de)
            rows.append(
                {
                    "audit_id": audit_id,
                    "entry_index": clean_cell(item.get("entry_index")),
                    "sense_index": sense_index,
                    "import_key": import_key,
                    "title": clean_cell(item.get("title")),
                    "source_id": clean_cell(item.get("source_id")),
                    "source_row_idx": clean_cell(item.get("source_row_idx")),
                    "page_number": clean_cell(item.get("page_number")),
                    "entry_type": clean_cell(item.get("entry_type")),
                    "definition_tr": tr,
                    "definition_en": en,
                    "definition_de": de,
                    "risk_score": str(risk_score),
                    "flags": " | ".join(flags),
                    "expected_review_status": "needs_review" if audit_id in bad_ids else "ok",
                }
            )
    return rows


def select_pilot_rows(rows: list[dict[str, str]], size: int, bad_count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    bad_rows = [row for row in rows if clean_cell(row.get("expected_review_status")) == "needs_review"]
    good_rows = [row for row in rows if clean_cell(row.get("expected_review_status")) == "ok"]
    bad_target = min(len(bad_rows), max(0, bad_count))
    good_target = min(len(good_rows), max(0, size - bad_target))
    if bad_target + good_target < min(size, len(rows)):
        extra_needed = min(size, len(rows)) - (bad_target + good_target)
        remaining = [row for row in rows if row not in bad_rows[:bad_target] and row not in good_rows[:good_target]]
        rng.shuffle(remaining)
        extras = remaining[:extra_needed]
    else:
        extras = []
    chosen_bad = rng.sample(bad_rows, bad_target) if bad_target else []
    chosen_good = rng.sample(good_rows, good_target) if good_target else []
    chosen = chosen_bad + chosen_good + extras
    rng.shuffle(chosen)
    return chosen[: min(size, len(chosen))]


def row_payload(row: dict[str, str]) -> dict[str, str]:
    return {field: clean_cell(row.get(field)) for field in CONTENT_FIELDS}


def estimated_user_message_chars(payload_rows: list[dict[str, str]]) -> int:
    return len(USER_MESSAGE_PREFIX) + len(json.dumps({"rows": payload_rows}, ensure_ascii=False))


def chunked(rows: list[dict[str, str]], size: int, max_user_message_chars: int) -> list[list[dict[str, str]]]:
    batches: list[list[dict[str, str]]] = []
    current_rows: list[dict[str, str]] = []
    current_payloads: list[dict[str, str]] = []
    for row in rows:
        payload = row_payload(row)
        candidate_payloads = current_payloads + [payload]
        too_many_rows = len(current_rows) >= size
        too_many_chars = current_payloads and estimated_user_message_chars(candidate_payloads) > max_user_message_chars
        if current_rows and (too_many_rows or too_many_chars):
            batches.append(current_rows)
            current_rows = [row]
            current_payloads = [payload]
        else:
            current_rows.append(row)
            current_payloads.append(payload)
    if current_rows:
        batches.append(current_rows)
    return batches


def load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_user_message(batch_rows: list[dict[str, str]]) -> str:
    rows_payload = [row_payload(row) for row in batch_rows]
    return USER_MESSAGE_PREFIX + json.dumps({"rows": rows_payload}, ensure_ascii=False)


def run_ollama_chat(
    *,
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
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for left, right in (("{", "}"), ("[", "]")):
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
    items_by_id: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each row result must be an object")
        audit_id = clean_cell(item.get("audit_id"))
        if not audit_id:
            raise ValueError("missing audit_id in row result")
        if audit_id in items_by_id:
            raise ValueError(f"duplicate audit_id in model response: {audit_id}")
        review_status = clean_cell(item.get("review_status")).lower()
        if review_status not in VALID_STATUSES:
            raise ValueError(f"invalid review_status for {audit_id}: {review_status}")
        bad_langs_raw = item.get("bad_langs")
        if not isinstance(bad_langs_raw, list):
            raise ValueError(f"bad_langs must be a list for {audit_id}")
        bad_langs = [clean_cell(value).lower() for value in bad_langs_raw if clean_cell(value)]
        if any(lang not in VALID_LANGS for lang in bad_langs):
            raise ValueError(f"invalid bad_langs for {audit_id}: {bad_langs}")
        if len(bad_langs) != len(set(bad_langs)):
            raise ValueError(f"duplicate bad_langs for {audit_id}")
        review_notes = clean_cell(item.get("review_notes"))
        if review_status == "ok" and bad_langs:
            raise ValueError(f"ok row must not include bad_langs for {audit_id}")
        if review_status == "needs_review" and not bad_langs:
            raise ValueError(f"needs_review row must include bad_langs for {audit_id}")
        if not review_notes:
            raise ValueError(f"missing review_notes for {audit_id}")
        items_by_id[audit_id] = {
            "audit_id": audit_id,
            "review_status": review_status,
            "bad_langs": "|".join(bad_langs),
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
        row["review_status"] = reviewed["review_status"]
        row["bad_langs"] = reviewed["bad_langs"]
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


def collect_reviewed_ids(run_dir: Path) -> set[str]:
    reviewed_ids: set[str] = set()
    batches_dir = run_dir / "batches"
    if not batches_dir.exists():
        return reviewed_ids
    for path in sorted(batches_dir.glob("batch_*.tsv")):
        try:
            rows = read_tsv(path)
        except Exception:
            continue
        for row in rows:
            audit_id = clean_cell(row.get("audit_id"))
            review_status = clean_cell(row.get("review_status")).lower()
            if audit_id and review_status in VALID_STATUSES:
                reviewed_ids.add(audit_id)
    return reviewed_ids


def score_pilot_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    counters = Counter()
    mismatches: list[dict[str, str]] = []
    for row in rows:
        expected = clean_cell(row.get("expected_review_status")).lower()
        predicted = clean_cell(row.get("review_status")).lower()
        if not expected:
            continue
        counters["rows"] += 1
        if expected == "needs_review":
            counters["expected_bad"] += 1
        else:
            counters["expected_ok"] += 1
        if predicted == "needs_review":
            counters["predicted_bad"] += 1
        else:
            counters["predicted_ok"] += 1
        if expected == predicted:
            counters["correct"] += 1
            if expected == "needs_review":
                counters["true_positive"] += 1
            else:
                counters["true_negative"] += 1
        else:
            if predicted == "needs_review":
                counters["false_positive"] += 1
            else:
                counters["false_negative"] += 1
            mismatches.append(
                {
                    "audit_id": clean_cell(row.get("audit_id")),
                    "title": clean_cell(row.get("title")),
                    "expected_review_status": expected,
                    "predicted_review_status": predicted,
                    "bad_langs": clean_cell(row.get("bad_langs")),
                    "review_notes": clean_cell(row.get("review_notes")),
                    "definition_tr": clean_cell(row.get("definition_tr")),
                    "definition_en": clean_cell(row.get("definition_en")),
                    "definition_de": clean_cell(row.get("definition_de")),
                }
            )
    precision = counters["true_positive"] / max(1, counters["true_positive"] + counters["false_positive"])
    recall = counters["true_positive"] / max(1, counters["true_positive"] + counters["false_negative"])
    accuracy = counters["correct"] / max(1, counters["rows"])
    return {
        "row_count": counters["rows"],
        "expected_bad": counters["expected_bad"],
        "expected_ok": counters["expected_ok"],
        "predicted_bad": counters["predicted_bad"],
        "predicted_ok": counters["predicted_ok"],
        "true_positive": counters["true_positive"],
        "true_negative": counters["true_negative"],
        "false_positive": counters["false_positive"],
        "false_negative": counters["false_negative"],
        "accuracy": round(accuracy, 4),
        "precision_needs_review": round(precision, 4),
        "recall_needs_review": round(recall, 4),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def maybe_write_annotated_snapshot(
    *,
    args: argparse.Namespace,
    snapshot_payload: dict[str, Any],
    reviewed_rows: list[dict[str, str]],
) -> None:
    output_path = Path(args.annotated_snapshot_path) if clean_cell(args.annotated_snapshot_path) else None
    if not output_path:
        return
    reviews_by_id = {clean_cell(row.get("audit_id")): row for row in reviewed_rows}
    entries = snapshot_payload.get("entries") or []
    for entry in entries:
        import_key = clean_cell(entry.get("import_key"))
        senses = entry.get("senses") or []
        for sense_index, sense in enumerate(senses):
            audit_id = f"{import_key}:{sense_index}"
            review = reviews_by_id.get(audit_id)
            if not review:
                continue
            sense["translation_review_status"] = clean_cell(review.get("review_status"))
            sense["translation_review_bad_langs"] = clean_cell(review.get("bad_langs"))
            sense["translation_review_notes"] = clean_cell(review.get("review_notes"))
    write_json(output_path, snapshot_payload)


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot_path)
    output_root = Path(args.output_root)
    run_dir = Path(args.run_dir) if clean_cell(args.run_dir) else default_run_dir(output_root, args.run_name)
    batches_dir = run_dir / "batches"
    requests_dir = run_dir / "requests"
    responses_dir = run_dir / "responses"
    ensure_parent_dir(batches_dir / "placeholder")
    ensure_parent_dir(requests_dir / "placeholder")
    ensure_parent_dir(responses_dir / "placeholder")

    if not Path(args.ollama_exe).exists():
        raise SystemExit(f"Ollama executable not found: {args.ollama_exe}")

    system_prompt = load_system_prompt(Path(args.system_prompt_path))
    snapshot_payload, snapshot_rows = load_snapshot_rows(snapshot_path)

    if args.pilot_only:
        all_rows = load_sample_rows(Path(args.pilot_sample_dir))
        frontier_rows = select_pilot_rows(all_rows, args.pilot_size, args.pilot_bad_count, args.pilot_seed)
    else:
        reviewed_ids = collect_reviewed_ids(run_dir)
        frontier_rows = [row for row in snapshot_rows if clean_cell(row.get("audit_id")) not in reviewed_ids]
        if args.max_rows > 0:
            frontier_rows = frontier_rows[: args.max_rows]

    all_batches = chunked(frontier_rows, max(1, args.batch_size), max(1200, args.max_user_message_chars))
    if args.max_batches > 0:
        all_batches = all_batches[: args.max_batches]

    summary: dict[str, Any] = {
        "snapshot_path": str(snapshot_path),
        "model": args.model,
        "run_dir": str(run_dir),
        "pilot_only": args.pilot_only,
        "frontier_row_count": len(frontier_rows),
        "batch_size": args.batch_size,
        "planned_batch_count": len(all_batches),
        "processed_batch_count": 0,
        "successful_batch_count": 0,
        "failed_batch_count": 0,
        "review_status_counts": {},
        "failures": [],
        "started_at": datetime.now().astimezone().isoformat(),
    }
    write_json(run_dir / "run_config.json", {k: v for k, v in vars(args).items()})

    run_started = time.perf_counter()
    all_reviewed_rows: list[dict[str, str]] = []
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
            write_tsv(batch_path, RESULT_FIELDS + (["expected_review_status"] if args.pilot_only else []), reviewed_batch)
            all_reviewed_rows.extend(reviewed_batch)
            action_counts = Counter(clean_cell(row.get("review_status")).lower() for row in reviewed_batch)
            running_counts = Counter(summary["review_status_counts"])
            running_counts.update(action_counts)
            summary["review_status_counts"] = dict(running_counts)
            summary["successful_batch_count"] += 1
            print(
                json.dumps(
                    {
                        "batch_id": batch_id,
                        "row_count": len(reviewed_batch),
                        "statuses": dict(action_counts),
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

    if args.pilot_only and all_reviewed_rows:
        pilot_report = score_pilot_rows(all_reviewed_rows)
        write_json(run_dir / "pilot_report.json", pilot_report)
        mismatch_fields = list(pilot_report["mismatches"][0].keys()) if pilot_report["mismatches"] else [
            "audit_id",
            "title",
            "expected_review_status",
            "predicted_review_status",
            "bad_langs",
            "review_notes",
            "definition_tr",
            "definition_en",
            "definition_de",
        ]
        write_tsv(run_dir / "pilot_mismatches.tsv", mismatch_fields, pilot_report["mismatches"])
        summary["pilot_report_path"] = str(run_dir / "pilot_report.json")
        summary["pilot_accuracy"] = pilot_report["accuracy"]
        summary["pilot_precision_needs_review"] = pilot_report["precision_needs_review"]
        summary["pilot_recall_needs_review"] = pilot_report["recall_needs_review"]

    if not args.pilot_only and all_reviewed_rows:
        maybe_write_annotated_snapshot(
            args=args,
            snapshot_payload=snapshot_payload,
            reviewed_rows=all_reviewed_rows,
        )

    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
