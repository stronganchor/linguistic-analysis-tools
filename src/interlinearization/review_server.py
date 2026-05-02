#!/usr/bin/env python3
"""Local interlinearization review server.

Run from the Z workspace:

    python3 interlinearization/review_server.py

Corrections are stored in review_corrections.tsv. Rows marked approved are also
loaded by auto_interlinearize.py on the next run.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
UI_DIR = SCRIPT_DIR / "review_ui"
CORRECTIONS_PATH = SCRIPT_DIR / "review_corrections.tsv"
CELL_NOTES_PATH = SCRIPT_DIR / "review_cell_notes.tsv"

sys.path.insert(0, str(SCRIPT_DIR))
from auto_interlinearize import normalize_key, tokenize_line  # noqa: E402


CORRECTION_FIELDS = [
    "form",
    "corrected_form",
    "lemma",
    "gloss_en",
    "gloss_tr",
    "pos",
    "confidence",
    "source",
    "note",
    "status",
    "lesson_ids",
    "line_ids",
    "scope",
    "updated_at",
    "morphemes",
]

CELL_NOTE_FIELDS = [
    "lesson_id",
    "line_id",
    "token_index",
    "row_label",
    "morph_index",
    "form",
    "current_value",
    "note",
    "status",
    "updated_at",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_morphemes(value: object) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return ""
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return ""
    else:
        parsed = value
    if not isinstance(parsed, list):
        return ""
    cleaned = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        form = str(item.get("form", "")).strip()
        gloss = str(item.get("gloss", "")).strip()
        pos = str(item.get("pos", "")).strip()
        if form or gloss or pos:
            cleaned.append({"form": form, "gloss": gloss, "pos": pos})
    return json.dumps(cleaned, ensure_ascii=False) if cleaned else ""


def find_texts() -> list[dict[str, object]]:
    texts: list[dict[str, object]] = []
    for path in sorted(OUTPUT_DIR.glob("*.interlinear.json")):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        texts.append(
            {
                "lesson_id": data.get("lesson_id", path.stem.replace(".interlinear", "")),
                "title": data.get("title", path.stem),
                "summary": data.get("summary", {}),
                "path": str(path.relative_to(ROOT)),
            }
        )
    return texts


def lesson_path(lesson_id: str) -> Path:
    path = OUTPUT_DIR / f"{lesson_id}.interlinear.json"
    if not path.exists():
        raise FileNotFoundError(lesson_id)
    return path


def load_payload(lesson_id: str) -> dict[str, object]:
    with lesson_path(lesson_id).open(encoding="utf-8") as f:
        return json.load(f)


class CorrectionStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @staticmethod
    def row_key(row: dict[str, str]) -> str:
        key = normalize_key(row.get("form", ""))
        scope = normalize_key(row.get("scope", ""))
        if scope in {"line", "lesson"}:
            return "\0".join([key, scope, row.get("lesson_ids", ""), row.get("line_ids", "")])
        return key

    def load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        corrections: dict[str, dict[str, str]] = {}
        for row in rows:
            key = normalize_key(row.get("form", ""))
            if not key:
                continue
            record = {field: row.get(field, "") for field in CORRECTION_FIELDS}
            corrections[self.row_key(record)] = record
        return corrections

    def save_all(self, corrections: dict[str, dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(
            corrections.values(),
            key=lambda item: (
                normalize_key(item.get("form", "")),
                normalize_key(item.get("scope", "")),
                item.get("lesson_ids", ""),
                item.get("line_ids", ""),
            ),
        )
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=CORRECTION_FIELDS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({field: row.get(field, "") for field in CORRECTION_FIELDS})
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def upsert(self, incoming: dict[str, object]) -> dict[str, str]:
        form = str(incoming.get("form", "")).strip()
        key = normalize_key(form)
        if not key:
            raise ValueError("form is required")

        corrections = self.load()
        existing = corrections.get(key, {})
        lesson_ids = merge_csv(existing.get("lesson_ids", ""), str(incoming.get("lesson_id", "")))
        line_ids = merge_csv(existing.get("line_ids", ""), ",".join(str(v) for v in incoming.get("line_ids", []) if v))
        status = str(incoming.get("status") or existing.get("status") or "draft")
        morphemes = normalize_morphemes(incoming.get("morphemes", existing.get("morphemes", "")))

        row = {
            "form": form,
            "corrected_form": str(incoming.get("corrected_form", existing.get("corrected_form", ""))).strip(),
            "lemma": str(incoming.get("lemma", existing.get("lemma", ""))).strip(),
            "gloss_en": str(incoming.get("gloss_en", existing.get("gloss_en", ""))).strip(),
            "gloss_tr": str(incoming.get("gloss_tr", existing.get("gloss_tr", ""))).strip(),
            "pos": str(incoming.get("pos", existing.get("pos", ""))).strip(),
            "confidence": str(incoming.get("confidence") or existing.get("confidence") or "0.97"),
            "source": "review_ui",
            "note": str(incoming.get("note", existing.get("note", ""))).strip(),
            "status": status,
            "lesson_ids": lesson_ids,
            "line_ids": line_ids,
            "scope": str(incoming.get("scope", existing.get("scope", ""))).strip(),
            "updated_at": utc_now(),
            "morphemes": morphemes,
        }
        corrections[self.row_key(row)] = row
        self.save_all(corrections)
        return row


class CellNoteStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, lesson_id: str = "") -> list[dict[str, str]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            rows = [
                {field: row.get(field, "") for field in CELL_NOTE_FIELDS}
                for row in csv.DictReader(f, delimiter="\t")
            ]
        if lesson_id:
            rows = [row for row in rows if row.get("lesson_id", "") == lesson_id]
        return rows

    def save_all(self, notes: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=self.path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            with tmp_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=CELL_NOTE_FIELDS)
                writer.writeheader()
                for row in notes:
                    writer.writerow({field: row.get(field, "") for field in CELL_NOTE_FIELDS})
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def upsert(self, incoming: dict[str, object]) -> dict[str, str]:
        lesson_id = str(incoming.get("lesson_id", "")).strip()
        line_id = str(incoming.get("line_id", "")).strip()
        token_index = str(incoming.get("token_index", "")).strip()
        row_label = str(incoming.get("row_label", "")).strip().upper()
        morph_index = str(incoming.get("morph_index", "")).strip()
        note = str(incoming.get("note", "")).strip()
        if not lesson_id or not line_id or not token_index or not row_label:
            raise ValueError("lesson_id, line_id, token_index, and row_label are required")
        if not note:
            raise ValueError("note is required")

        row = {
            "lesson_id": lesson_id,
            "line_id": line_id,
            "token_index": token_index,
            "row_label": row_label,
            "morph_index": morph_index,
            "form": str(incoming.get("form", "")).strip(),
            "current_value": str(incoming.get("current_value", "")).strip(),
            "note": note,
            "status": str(incoming.get("status", "open") or "open").strip(),
            "updated_at": utc_now(),
        }
        notes = self.load()
        key = cell_note_key(row)
        replaced = False
        for index, existing in enumerate(notes):
            if cell_note_key(existing) == key:
                notes[index] = row
                replaced = True
                break
        if not replaced:
            notes.append(row)
        self.save_all(notes)
        return row


def cell_note_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("lesson_id", ""),
        row.get("line_id", ""),
        row.get("token_index", ""),
        row.get("row_label", ""),
        row.get("morph_index", ""),
    )


def merge_csv(left: str, right: str) -> str:
    values = []
    seen = set()
    for chunk in [left, right]:
        for item in str(chunk).split(","):
            item = item.strip()
            if item and item not in seen:
                seen.add(item)
                values.append(item)
    return ",".join(values)


def csv_contains(value: str, needle: str) -> bool:
    return needle in {part.strip() for part in str(value).split(",") if part.strip()}


def find_correction(
    corrections: dict[str, dict[str, str]],
    form: str,
    lesson_id: str,
    line_id: str = "",
) -> dict[str, str] | None:
    key = normalize_key(form)
    if not key:
        return None
    fallback = corrections.get(key)
    lesson_match = None
    for row in corrections.values():
        if normalize_key(row.get("form", "")) != key:
            continue
        scope = normalize_key(row.get("scope", ""))
        if scope == "line":
            if csv_contains(row.get("lesson_ids", ""), lesson_id) and csv_contains(row.get("line_ids", ""), line_id):
                return row
        elif scope == "lesson":
            if csv_contains(row.get("lesson_ids", ""), lesson_id):
                lesson_match = row
    return lesson_match or fallback


def apply_correction(token: dict[str, object], correction: dict[str, str] | None) -> dict[str, object]:
    merged = dict(token)
    if correction:
        merged["review_correction"] = correction
        morphemes = normalize_morphemes(correction.get("morphemes", ""))
        if morphemes:
            merged["morphemes"] = json.loads(morphemes)
        if correction.get("corrected_form"):
            merged["corrected_form"] = correction.get("corrected_form", "")
        if correction.get("status") == "approved":
            merged["corrected_lemma"] = correction.get("lemma", "")
            merged["corrected_gloss_en"] = correction.get("gloss_en", "")
            merged["corrected_gloss_tr"] = correction.get("gloss_tr", "")
            merged["corrected_pos"] = correction.get("pos", "")
            merged["corrected_confidence"] = safe_float(correction.get("confidence"), 0.97)
    return merged


def interlinear_lines(payload: dict[str, object]) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for line in payload.get("lines", []):
        lines.append(
            {
                "id": line.get("id", ""),
                "text": line.get("text", ""),
                "start_sec": line.get("start_sec", ""),
                "end_sec": line.get("end_sec", ""),
                "tokens": line.get("tokens", []),
                "phrase_matches": line.get("phrase_matches", []),
            }
        )
    return lines


def state_for_lesson(lesson_id: str, store: CorrectionStore, note_store: CellNoteStore) -> dict[str, object]:
    payload = load_payload(lesson_id)
    corrections = store.load()
    forms: dict[str, dict[str, object]] = {}

    for line in payload.get("lines", []):
        line_id = str(line.get("id", ""))
        line_tokens = line.get("tokens", [])
        for phrase in line.get("phrase_matches", []):
            form = " ".join(str(part) for part in phrase.get("forms", []) if part) or str(phrase.get("form", ""))
            key = normalize_key(form)
            if not key:
                continue
            correction = find_correction(corrections, form, lesson_id, line_id)
            token = apply_correction(
                {
                    "form": form,
                    "lemma": phrase.get("lemma", ""),
                    "gloss_en": phrase.get("gloss_en", ""),
                    "gloss_tr": phrase.get("gloss_tr", ""),
                    "display_gloss": phrase.get("display_gloss", ""),
                    "pos": phrase.get("pos", "EXPR"),
                    "confidence": phrase.get("confidence", 0.97),
                    "match_type": phrase.get("match_type", "phrase_memory"),
                    "source": phrase.get("source", "phrase_memory"),
                    "review_note": phrase.get("review_note", ""),
                },
                correction,
            )
            record = forms.setdefault(
                key,
                {
                    "key": key,
                    "form": form,
                    "count": 0,
                    "correction": correction,
                    "candidate": token,
                    "contexts": [],
                    "status": correction.get("status") if correction else "approved",
                    "lowest_confidence": safe_float(token.get("confidence"), 0.97),
                },
            )
            record["count"] = int(record["count"]) + 1
            record["contexts"].append(
                {
                    "lesson_id": lesson_id,
                    "line_id": line_id,
                    "token_start": phrase.get("start_token", 1),
                    "token_end": phrase.get("end_token", phrase.get("start_token", 1)),
                    "text": line.get("text", ""),
                    "start_sec": line.get("start_sec", ""),
                    "end_sec": line.get("end_sec", ""),
                    "tokens": line_tokens,
                }
            )
        for index, token in enumerate(line_tokens, start=1):
            original_token = dict(token)
            form = str(original_token.get("form", ""))
            key = normalize_key(form)
            correction = find_correction(corrections, form, lesson_id, line_id)
            token = apply_correction(original_token, correction)
            line_tokens[index - 1] = token
            should_review = (
                safe_float(token.get("confidence")) < 0.84
                or token.get("match_type") in {"unknown", "dictionary_folded", "orthography_variant", "prefix_morphology", "suffix_morphology"}
                or correction
            )
            if not should_review:
                continue
            record = forms.setdefault(
                key,
                {
                    "key": key,
                    "form": form,
                    "count": 0,
                    "correction": correction,
                    "candidate": token,
                    "contexts": [],
                    "status": correction.get("status") if correction else "unreviewed",
                    "lowest_confidence": safe_float(token.get("confidence")),
                },
            )
            record["count"] = int(record["count"]) + 1
            record["lowest_confidence"] = min(float(record["lowest_confidence"]), safe_float(token.get("confidence")))
            if safe_float(token.get("confidence")) > safe_float(record["candidate"].get("confidence")):
                record["candidate"] = token
            record["contexts"].append(
                {
                    "lesson_id": lesson_id,
                    "line_id": line_id,
                    "token_index": index,
                    "text": line.get("text", ""),
                    "start_sec": line.get("start_sec", ""),
                    "end_sec": line.get("end_sec", ""),
                    "tokens": line.get("tokens", []),
                }
            )

    def sort_key(item: dict[str, object]) -> tuple[int, float, int, str]:
        status_rank = {"unreviewed": 0, "draft": 1, "approved": 2}.get(str(item.get("status")), 1)
        return (status_rank, float(item.get("lowest_confidence", 1)), -int(item.get("count", 0)), str(item.get("key", "")))

    queue = sorted(forms.values(), key=sort_key)
    return {
        "lesson": {
            "lesson_id": payload.get("lesson_id"),
            "title": payload.get("title"),
            "summary": payload.get("summary", {}),
            "source": payload.get("source", ""),
        },
        "queue": queue,
        "interlinear": interlinear_lines(payload),
        "corrections_path": str(CORRECTIONS_PATH.relative_to(ROOT)),
        "cell_notes": note_store.load(lesson_id),
        "cell_notes_path": str(CELL_NOTES_PATH.relative_to(ROOT)),
    }


def regenerate(lesson_id: str) -> dict[str, object]:
    payload = load_payload(lesson_id)
    source = Path(str(payload.get("source", "")))
    if not source.is_absolute():
        source = ROOT / source
    title = str(payload.get("title") or lesson_id)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "auto_interlinearize.py"),
        "--alignment-tsv",
        str(source),
        "--lesson-id",
        lesson_id,
        "--title",
        title,
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 2),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def text_projected(text: str) -> str:
    return " ".join(tokenize_line(text))


def format_seconds(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def update_line_text(lesson_id: str, line_id: str, text: str) -> dict[str, object]:
    payload = load_payload(lesson_id)
    source = Path(str(payload.get("source", "")))
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(str(source))

    parts = [part.strip() for part in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n") if part.strip()]
    if not parts:
        raise ValueError("line text is required")

    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not fieldnames:
        raise ValueError("source TSV has no header")

    changed = False
    next_rows: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("id", "")) != str(line_id):
            next_rows.append(row)
            continue
        changed = True
        next_rows.extend(split_source_row(row, parts))

    if not changed:
        raise FileNotFoundError(f"line {line_id}")

    fd, tmp_name = tempfile.mkstemp(prefix=source.name, suffix=".tmp", dir=source.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()
            for row in next_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        tmp_path.replace(source)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return {"ok": True, "line_id": line_id, "parts": len(parts), "source": str(source)}


def split_source_row(row: dict[str, str], parts: list[str]) -> list[dict[str, str]]:
    if len(parts) == 1:
        next_row = dict(row)
        apply_text_to_source_row(next_row, parts[0])
        return [next_row]

    token_counts = [max(1, len(tokenize_line(part))) for part in parts]
    total_tokens = sum(token_counts) or len(parts)
    start = safe_float(row.get("start_sec"), 0.0)
    end = safe_float(row.get("end_sec"), start)
    duration = max(0.0, end - start)
    output: list[dict[str, str]] = []
    elapsed_tokens = 0
    base_id = str(row.get("id", ""))
    for index, part in enumerate(parts):
        next_row = dict(row)
        next_row["id"] = f"{base_id}{chr(ord('a') + index)}"
        apply_text_to_source_row(next_row, part)
        if duration > 0:
            part_start = start + duration * (elapsed_tokens / total_tokens)
            elapsed_tokens += token_counts[index]
            part_end = start + duration * (elapsed_tokens / total_tokens)
            next_row["start_sec"] = format_seconds(part_start)
            next_row["end_sec"] = format_seconds(part_end)
            next_row["duration_sec"] = format_seconds(part_end - part_start)
        output.append(next_row)
    return output


def apply_text_to_source_row(row: dict[str, str], text: str) -> None:
    if "text_full" in row:
        row["text_full"] = text
    elif "text" in row:
        row["text"] = text
    if "text_projected" in row:
        row["text_projected"] = text_projected(text)
    if "token_count" in row:
        row["token_count"] = str(len(tokenize_line(text)))


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "ZInterlinearReview/0.1"
    store = CorrectionStore(CORRECTIONS_PATH)
    note_store = CellNoteStore(CELL_NOTES_PATH)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", file=sys.stderr)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/texts":
                self.write_json({"texts": find_texts()})
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                lesson_id = query.get("lesson_id", [""])[0]
                self.write_json(state_for_lesson(lesson_id, self.store, self.note_store))
                return
            self.serve_static(parsed.path)
        except FileNotFoundError:
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            body = self.read_json()
            if parsed.path == "/api/corrections":
                row = self.store.upsert(body)
                self.write_json({"ok": True, "correction": row})
                return
            if parsed.path == "/api/cell-note":
                row = self.note_store.upsert(body)
                self.write_json({"ok": True, "note": row})
                return
            if parsed.path == "/api/regenerate":
                lesson_id = str(body.get("lesson_id", ""))
                self.write_json(regenerate(lesson_id))
                return
            if parsed.path == "/api/line-text":
                lesson_id = str(body.get("lesson_id", ""))
                line_id = str(body.get("line_id", ""))
                text = str(body.get("text", ""))
                self.write_json(update_line_text(lesson_id, line_id, text))
                return
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except ValueError as exc:
            self.write_error(HTTPStatus.BAD_REQUEST, str(exc))
        except FileNotFoundError:
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
        except Exception as exc:
            self.write_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(raw)

    def write_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def write_error(self, status: HTTPStatus, message: str) -> None:
        self.write_json({"ok": False, "error": message}, status)

    def serve_static(self, request_path: str) -> None:
        if request_path in {"", "/"}:
            request_path = "/index.html"
        rel = unquote(request_path).lstrip("/")
        if not re.match(r"^[A-Za-z0-9_./-]+$", rel) or ".." in Path(rel).parts:
            self.write_error(HTTPStatus.BAD_REQUEST, "Invalid path")
            return
        path = UI_DIR / rel
        if not path.exists() or not path.is_file():
            self.write_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        raw = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix in {".html", ".css", ".js"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Run the local Zazaki interlinear review UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    return ServerConfig(host=args.host, port=args.port)


def main() -> int:
    config = parse_args()
    if not UI_DIR.exists():
        raise SystemExit(f"Missing UI directory: {UI_DIR}")
    server = ThreadingHTTPServer((config.host, config.port), ReviewHandler)
    print(f"Review UI: http://{config.host}:{config.port}/")
    print(f"Corrections: {CORRECTIONS_PATH}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
