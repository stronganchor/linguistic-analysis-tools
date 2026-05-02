#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any
from urllib.parse import urlparse

from review_lltools_ipa_comparison import normalize_title_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def root_dir() -> Path:
    return script_dir().parent


def default_audit_path() -> Path:
    return (
        root_dir()
        / "zazacaogren"
        / "full-audio-model-review-april-2026"
        / "zazacaogren-new-mms-live-csv-combined-import.audit.jsonl"
    )


def default_predictions_path() -> Path:
    return (
        root_dir()
        / "zazacaogren"
        / "full-audio-model-review-april-2026"
        / "new-mms-v003-all-audio.jsonl"
    )


def default_output_dir() -> Path:
    return (
        root_dir()
        / "zazacaogren"
        / "full-audio-model-review-april-2026"
        / "manual-unresolved-review-ui"
    )


def default_live_csv_path() -> Path:
    return (
        root_dir()
        / "zazacaogren"
        / "zip files"
        / "ll-tools-wordset-text-genc-palu-20260408-033634.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local browser UI for reviewing unresolved IPA predictions and "
            "exporting accepted decisions as an import-ready JSONL."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--audit-path", default=str(default_audit_path()))
    parser.add_argument("--predictions-path", default=str(default_predictions_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--live-csv-path", default=str(default_live_csv_path()))
    parser.add_argument(
        "--speaker-user-id",
        type=int,
        default=8,
        help="Only include unresolved rows for this speaker_user_id. Use 0 to include all speakers.",
    )
    parser.add_argument(
        "--include-current-imported-conflicts",
        action="store_true",
        help=(
            "Include unresolved rows that already have a current_recording_ipa on the live site. "
            "By default, only not-yet-imported unresolved rows are shown."
        ),
    )
    parser.add_argument(
        "--include-live-csv-covered-rows",
        action="store_true",
        help=(
            "Include unresolved rows whose title or translation already matches a populated "
            "Lexeme/Gloss_tr entry in the live-site CSV."
        ),
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the review UI in the default browser after starting the server.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def path_windows(path: Path) -> str:
    raw = str(path)
    if re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw):
        drive, rest = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw).groups()
        return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
    return raw.replace("/", "\\")


def maybe_convert_windows_to_wsl(raw: str) -> str | None:
    match = re.match(r"^([a-zA-Z]):\\(.*)$", raw)
    if not match:
        return None
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace('\\', '/')}"


def guess_recording_basename(row: dict[str, Any]) -> str:
    ordered_keys = ("audio_windows", "audio_posix", "audio_relpath")
    if os.name != "nt":
        ordered_keys = ("audio_posix", "audio_windows", "audio_relpath")
    for key in ordered_keys:
        raw = str(row.get(key) or "").strip()
        if not raw:
            continue
        if re.match(r"^[a-zA-Z]:\\", raw):
            return PureWindowsPath(raw).name
        return Path(raw).name
    return ""


def resolve_audio_path(row: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for key in ("audio_posix", "audio_windows"):
        raw = str(row.get(key) or "").strip()
        if not raw:
            continue
        if os.name == "nt":
            if raw.startswith("/mnt/"):
                continue
            candidates.append(Path(raw))
            continue
        if re.match(r"^[a-zA-Z]:\\", raw):
            converted = maybe_convert_windows_to_wsl(raw)
            if converted:
                candidates.append(Path(converted))
            continue
        candidates.append(Path(raw))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def audio_path_for_display(row: dict[str, Any]) -> str:
    for key in ("audio_windows", "audio_posix", "audio_relpath"):
        raw = str(row.get(key) or "").strip()
        if raw:
            return raw
    return ""


def bundle_stem(source_bundle: str) -> str:
    text = str(source_bundle or "").strip()
    if text.lower().endswith(".zip"):
        return text[:-4]
    return text


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def human_source(kind: str, index: int | None = None) -> str:
    labels = {
        "current": "Current imported IPA",
        "assistant": "Assistant guess",
        "new_mms": "New MMS",
        "old_mms": "Old MMS",
        "whisper": "Whisper Medium",
        "csv": "CSV candidate",
        "bucket": "Candidate bucket",
    }
    label = labels.get(kind, kind)
    if kind == "csv" and index is not None:
        return f"{label} {index + 1}"
    return label


class ReviewStore:
    def __init__(
        self,
        audit_path: Path,
        predictions_path: Path,
        output_dir: Path,
        live_csv_path: Path,
        speaker_user_id: int,
        include_current_imported_conflicts: bool,
        include_live_csv_covered_rows: bool,
    ):
        self.audit_path = audit_path
        self.predictions_path = predictions_path
        self.output_dir = output_dir
        self.live_csv_path = live_csv_path
        self.speaker_user_id = speaker_user_id
        self.include_current_imported_conflicts = include_current_imported_conflicts
        self.include_live_csv_covered_rows = include_live_csv_covered_rows
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_path = script_dir() / "unresolved_ipa_review.html"
        self.decisions_path = self.output_dir / "unresolved-ipa-review-decisions.jsonl"
        self.import_path = self.output_dir / "unresolved-ipa-review-import.jsonl"
        self.summary_path = self.output_dir / "unresolved-ipa-review-summary.json"
        self.audio_root_dirs = self._discover_audio_roots()

        self.predictions_by_recording = self._load_predictions()
        self.live_csv_keys = self._load_live_csv_keys()
        self.items = self._load_unresolved_items()
        self.index_by_recording = {
            int(item["recording_id"]): idx for idx, item in enumerate(self.items)
        }
        self.decisions_by_recording = self._load_existing_decisions()
        self._apply_decisions_to_items()
        self._write_outputs()

    def _load_predictions(self) -> dict[int, dict[str, Any]]:
        rows = read_jsonl(self.predictions_path)
        mapping: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                mapping[int(row["recording_id"])] = row
            except (KeyError, TypeError, ValueError):
                continue
        return mapping

    def _load_live_csv_keys(self) -> set[str]:
        keys: set[str] = set()
        if not self.live_csv_path.is_file():
            return keys
        with self.live_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not normalize_space(str(row.get("PhoneticForm") or "")):
                    continue
                for field in ("Lexeme", "Gloss_tr"):
                    key = normalize_title_text(str(row.get(field) or ""))
                    if key:
                        keys.add(key)
        return keys

    def _discover_audio_roots(self) -> list[Path]:
        roots: list[Path] = []
        for path in sorted(script_dir().glob("extracted_bundles*")):
            if path.is_dir():
                roots.append(path.resolve())
        return roots

    def _resolve_audio_fallback(self, row: dict[str, Any]) -> Path | None:
        relpath = str(row.get("audio_relpath") or "").strip()
        source = bundle_stem(str(row.get("source_bundle") or ""))
        if not relpath:
            return None
        rel = Path(relpath)
        basename = rel.name
        for root in self.audio_root_dirs:
            if source:
                candidate = root / source / rel
                if candidate.is_file():
                    return candidate
            if basename:
                matches = list(root.glob(f"**/{basename}"))
                for match in matches:
                    if match.is_file():
                        return match.resolve()
        return None

    def _build_candidate_map(self, audit_row: dict[str, Any]) -> list[dict[str, Any]]:
        proposals: dict[str, dict[str, Any]] = {}

        def ensure_entry(ipa: str) -> dict[str, Any] | None:
            ipa = normalize_space(ipa)
            if not ipa:
                return None
            entry = proposals.get(ipa)
            if entry is None:
                entry = {
                    "ipa": ipa,
                    "raw_sources": [],
                    "bucket_support_count": 0,
                    "bucket_models": [],
                    "bucket_rule_counts": [],
                    "csv_support_any": False,
                    "csv_support_typed": False,
                    "sort_rank": 999,
                }
                proposals[ipa] = entry
            return entry

        def add_raw(ipa: str, kind: str, label: str, rank: int) -> None:
            entry = ensure_entry(ipa)
            if entry is None:
                return
            if rank < entry["sort_rank"]:
                entry["sort_rank"] = rank
            if label not in entry["raw_sources"]:
                entry["raw_sources"].append(label)

        add_raw(audit_row.get("current_recording_ipa", ""), "current", human_source("current"), 0)
        add_raw(audit_row.get("assistant_guess_ipa", ""), "assistant", human_source("assistant"), 1)
        add_raw(audit_row.get("new_mms_recording_ipa", ""), "new_mms", human_source("new_mms"), 2)
        add_raw(audit_row.get("old_mms_recording_ipa", ""), "old_mms", human_source("old_mms"), 3)
        add_raw(audit_row.get("whisper_recording_ipa", ""), "whisper", human_source("whisper"), 4)

        for idx, candidate in enumerate(audit_row.get("csv_candidates", []) or []):
            add_raw(candidate, "csv", human_source("csv", idx), 5)

        for bucket in audit_row.get("candidate_buckets", []) or []:
            entry = ensure_entry(bucket.get("chosen_ipa", ""))
            if entry is None:
                continue
            if 6 < entry["sort_rank"]:
                entry["sort_rank"] = 6
            entry["bucket_support_count"] = max(
                int(entry["bucket_support_count"] or 0),
                int(bucket.get("support_count") or 0),
            )
            for model in bucket.get("models", []) or []:
                model = str(model)
                if model and model not in entry["bucket_models"]:
                    entry["bucket_models"].append(model)
            if bucket.get("best_rule_count") is not None:
                entry["bucket_rule_counts"].append(int(bucket.get("best_rule_count") or 0))
            entry["csv_support_any"] = bool(entry["csv_support_any"] or bucket.get("csv_support_any"))
            entry["csv_support_typed"] = bool(
                entry["csv_support_typed"] or bucket.get("csv_support_typed")
            )

        ordered = sorted(
            proposals.values(),
            key=lambda item: (
                item["sort_rank"],
                -int(item["bucket_support_count"] or 0),
                item["ipa"],
            ),
        )
        for entry in ordered:
            entry["bucket_rule_count_min"] = (
                min(entry["bucket_rule_counts"]) if entry["bucket_rule_counts"] else None
            )
            entry.pop("bucket_rule_counts", None)
        return ordered

    def _load_unresolved_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for audit_row in read_jsonl(self.audit_path):
            if audit_row.get("accepted"):
                continue
            recording_id = int(audit_row["recording_id"])
            prediction_row = self.predictions_by_recording.get(recording_id, {})
            prediction_speaker_id = prediction_row.get("speaker_user_id")
            if self.speaker_user_id and prediction_speaker_id != self.speaker_user_id:
                continue
            current_recording_ipa = str(audit_row.get("current_recording_ipa") or "").strip()
            if current_recording_ipa and not self.include_current_imported_conflicts:
                continue
            if not self.include_live_csv_covered_rows:
                lookup_keys = {
                    normalize_title_text(str(audit_row.get("word_title") or "")),
                    normalize_title_text(str(audit_row.get("word_translation") or "")),
                }
                lookup_keys.discard("")
                if any(key in self.live_csv_keys for key in lookup_keys):
                    continue
            item = {
                "word_id": int(audit_row["word_id"]),
                "recording_id": recording_id,
                "word_title": str(audit_row.get("word_title") or ""),
                "word_translation": str(audit_row.get("word_translation") or ""),
                "recording_type": str(audit_row.get("recording_type") or ""),
                "current_recording_ipa": current_recording_ipa,
                "current_rule_count": int(audit_row.get("current_rule_count") or 0),
                "current_canonical": str(audit_row.get("current_canonical") or ""),
                "csv_candidates": [str(v) for v in (audit_row.get("csv_candidates") or [])],
                "assistant_guess_ipa": str(audit_row.get("assistant_guess_ipa") or ""),
                "new_mms_recording_ipa": str(audit_row.get("new_mms_recording_ipa") or ""),
                "old_mms_recording_ipa": str(audit_row.get("old_mms_recording_ipa") or ""),
                "whisper_recording_ipa": str(audit_row.get("whisper_recording_ipa") or ""),
                "candidate_buckets": audit_row.get("candidate_buckets") or [],
                "audio_windows": str(prediction_row.get("audio_windows") or ""),
                "audio_posix": str(prediction_row.get("audio_posix") or ""),
                "audio_relpath": str(prediction_row.get("audio_relpath") or ""),
                "audio_filename": guess_recording_basename(prediction_row),
                "source_bundle": str(prediction_row.get("source_bundle") or ""),
                "category_name": str(prediction_row.get("category_name") or ""),
                "category_slug": str(prediction_row.get("category_slug") or ""),
                "speaker_user_id": prediction_row.get("speaker_user_id"),
                "word_slug": str(prediction_row.get("word_slug") or ""),
                "recording_slug": str(prediction_row.get("recording_slug") or ""),
            }
            resolved_audio = resolve_audio_path(item) or self._resolve_audio_fallback(item)
            if resolved_audio is not None:
                item["resolved_audio_posix"] = str(resolved_audio)
                item["audio_available"] = True
            else:
                item["resolved_audio_posix"] = ""
                item["audio_available"] = False
            item["proposals"] = self._build_candidate_map(audit_row)
            item["saved_decision"] = None
            items.append(item)
        return items

    def _load_existing_decisions(self) -> dict[int, dict[str, Any]]:
        if not self.decisions_path.is_file():
            return {}
        decisions: dict[int, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            try:
                recording_id = int(row["recording_id"])
            except (KeyError, TypeError, ValueError):
                continue
            decisions[recording_id] = row
        return decisions

    def _apply_decisions_to_items(self) -> None:
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            item["saved_decision"] = decision

    def _sorted_decisions(self) -> list[dict[str, Any]]:
        ordered = []
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            if decision:
                ordered.append(decision)
        return ordered

    def _import_rows(self) -> list[dict[str, Any]]:
        rows = []
        for decision in self._sorted_decisions():
            if str(decision.get("decision_type") or "") == "skip":
                continue
            ipa = normalize_space(str(decision.get("recording_ipa") or ""))
            if not ipa:
                continue
            rows.append(
                {
                    "word_id": int(decision["word_id"]),
                    "recording_id": int(decision["recording_id"]),
                    "recording_ipa": ipa,
                    "word_title": str(decision.get("word_title") or ""),
                    "word_translation": str(decision.get("word_translation") or ""),
                }
            )
        return rows

    def _write_outputs(self) -> None:
        decisions = self._sorted_decisions()
        import_rows = self._import_rows()
        write_jsonl(self.decisions_path, decisions)
        write_jsonl(self.import_path, import_rows)
        summary = {
            "generated_at": iso_now(),
            "audit_path": path_windows(self.audit_path),
            "predictions_path": path_windows(self.predictions_path),
            "output_dir": path_windows(self.output_dir),
            "total_unresolved_rows": len(self.items),
            "saved_decisions": len(decisions),
            "import_rows": len(import_rows),
            "decisions_path": path_windows(self.decisions_path),
            "import_path": path_windows(self.import_path),
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def client_state(self) -> dict[str, Any]:
        import_rows = self._import_rows()
        return {
            "generated_at": iso_now(),
            "total_rows": len(self.items),
            "saved_decisions": len(self.decisions_by_recording),
            "import_rows": len(import_rows),
            "speaker_user_id": self.speaker_user_id,
            "include_current_imported_conflicts": self.include_current_imported_conflicts,
            "include_live_csv_covered_rows": self.include_live_csv_covered_rows,
            "paths": {
                "audit_path": path_windows(self.audit_path),
                "predictions_path": path_windows(self.predictions_path),
                "live_csv_path": path_windows(self.live_csv_path),
                "output_dir": path_windows(self.output_dir),
                "decisions_path": path_windows(self.decisions_path),
                "import_path": path_windows(self.import_path),
                "summary_path": path_windows(self.summary_path),
            },
            "items": self.items,
        }

    def set_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        recording_id = int(payload["recording_id"])
        idx = self.index_by_recording[recording_id]
        item = self.items[idx]
        chosen_ipa = normalize_space(str(payload.get("recording_ipa") or ""))
        if not chosen_ipa:
            raise ValueError("No IPA value provided.")

        decision = {
            "word_id": int(item["word_id"]),
            "recording_id": recording_id,
            "recording_ipa": chosen_ipa,
            "word_title": item["word_title"],
            "word_translation": item["word_translation"],
            "decision_type": str(payload.get("decision_type") or "proposal"),
            "source_label": str(payload.get("source_label") or ""),
            "source_kind": str(payload.get("source_kind") or ""),
            "custom_text": normalize_space(str(payload.get("custom_text") or "")),
            "saved_at": iso_now(),
        }
        self.decisions_by_recording[recording_id] = decision
        item["saved_decision"] = decision
        self._write_outputs()
        return {
            "ok": True,
            "decision": decision,
            "saved_decisions": len(self.decisions_by_recording),
            "import_rows": len(self._import_rows()),
        }

    def clear_decision(self, recording_id: int) -> dict[str, Any]:
        self.decisions_by_recording.pop(recording_id, None)
        idx = self.index_by_recording[recording_id]
        self.items[idx]["saved_decision"] = None
        self._write_outputs()
        return {
            "ok": True,
            "saved_decisions": len(self.decisions_by_recording),
            "import_rows": len(self._import_rows()),
        }

    def skip_decision(self, recording_id: int) -> dict[str, Any]:
        idx = self.index_by_recording[recording_id]
        item = self.items[idx]
        decision = {
            "word_id": int(item["word_id"]),
            "recording_id": int(recording_id),
            "recording_ipa": "",
            "word_title": item["word_title"],
            "word_translation": item["word_translation"],
            "decision_type": "skip",
            "source_label": "Skipped by reviewer",
            "source_kind": "skip",
            "custom_text": "",
            "saved_at": iso_now(),
        }
        self.decisions_by_recording[recording_id] = decision
        item["saved_decision"] = decision
        self._write_outputs()
        return {
            "ok": True,
            "decision": decision,
            "saved_decisions": len(self.decisions_by_recording),
            "import_rows": len(self._import_rows()),
        }

    def audio_path_for_recording(self, recording_id: int) -> Path | None:
        idx = self.index_by_recording.get(recording_id)
        if idx is None:
            return None
        item = self.items[idx]
        resolved = str(item.get("resolved_audio_posix") or "").strip()
        if resolved:
            path = Path(resolved)
            if path.is_file():
                return path
        return resolve_audio_path(item) or self._resolve_audio_fallback(item)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def send_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def send_text(handler: BaseHTTPRequestHandler, text: str, status: int = 200) -> None:
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str | None = None) -> None:
    data = path.read_bytes()
    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def send_head_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str | None = None) -> None:
    size = path.stat().st_size
    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )
    handler.send_header("Content-Length", str(size))
    handler.end_headers()


def send_audio(handler: BaseHTTPRequestHandler, path: Path) -> None:
    size = path.stat().st_size
    range_header = handler.headers.get("Range")
    content_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    if not range_header:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                handler.wfile.write(chunk)
        return

    match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match:
        send_text(handler, "Invalid Range header.", status=416)
        return

    start_raw, end_raw = match.groups()
    if start_raw == "":
        suffix_len = int(end_raw or "0")
        if suffix_len <= 0:
            send_text(handler, "Invalid Range header.", status=416)
            return
        start = max(size - suffix_len, 0)
        end = size - 1
    else:
        start = int(start_raw)
        end = int(end_raw) if end_raw else size - 1
    if start < 0 or end < start or start >= size:
        handler.send_response(416)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
        return

    end = min(end, size - 1)
    length = end - start + 1
    handler.send_response(206)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(length))
    handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Accept-Ranges", "bytes")
    handler.end_headers()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(64 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def build_handler(store: ReviewStore):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "UnresolvedIPAReview/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            message = fmt % args
            sys.stdout.write(f"[{self.log_date_time_string()}] {message}\n")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/", "/index.html"):
                send_file(self, store.html_path, "text/html; charset=utf-8")
                return
            if route == "/api/bootstrap":
                send_json(self, store.client_state())
                return
            if route == "/api/download-import":
                data = store.import_path.read_bytes() if store.import_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{store.import_path.name}"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route.startswith("/audio/"):
                recording_id_raw = route.rsplit("/", 1)[-1]
                try:
                    recording_id = int(recording_id_raw)
                except ValueError:
                    send_text(self, "Invalid recording id.", status=400)
                    return
                audio_path = store.audio_path_for_recording(recording_id)
                if audio_path is None or not audio_path.is_file():
                    send_text(self, "Audio not found.", status=404)
                    return
                send_audio(self, audio_path)
                return
            send_text(self, "Not found.", status=404)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/", "/index.html"):
                send_head_file(self, store.html_path, "text/html; charset=utf-8")
                return
            if route == "/api/bootstrap":
                body = json.dumps(store.client_state(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            if route == "/api/download-import":
                data = store.import_path.read_bytes() if store.import_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{store.import_path.name}"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return
            if route.startswith("/audio/"):
                recording_id_raw = route.rsplit("/", 1)[-1]
                try:
                    recording_id = int(recording_id_raw)
                except ValueError:
                    send_text(self, "Invalid recording id.", status=400)
                    return
                audio_path = store.audio_path_for_recording(recording_id)
                if audio_path is None or not audio_path.is_file():
                    send_text(self, "Audio not found.", status=404)
                    return
                size = audio_path.stat().st_size
                content_type = mimetypes.guess_type(audio_path.name)[0] or "audio/wav"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
            send_text(self, "Not found.", status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            try:
                payload = parse_json_body(self)
            except json.JSONDecodeError:
                send_json(self, {"ok": False, "error": "Invalid JSON body."}, status=400)
                return

            if route == "/api/save-decision":
                try:
                    result = store.set_decision(payload)
                except (KeyError, ValueError) as exc:
                    send_json(self, {"ok": False, "error": str(exc)}, status=400)
                    return
                send_json(self, result)
                return

            if route == "/api/clear-decision":
                try:
                    recording_id = int(payload.get("recording_id"))
                except (TypeError, ValueError):
                    send_json(self, {"ok": False, "error": "Invalid recording id."}, status=400)
                    return
                result = store.clear_decision(recording_id)
                send_json(self, result)
                return

            if route == "/api/skip-decision":
                try:
                    recording_id = int(payload.get("recording_id"))
                except (TypeError, ValueError):
                    send_json(self, {"ok": False, "error": "Invalid recording id."}, status=400)
                    return
                result = store.skip_decision(recording_id)
                send_json(self, result)
                return

            send_json(self, {"ok": False, "error": "Not found."}, status=404)

    return ReviewHandler


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit_path).expanduser().resolve()
    predictions_path = Path(args.predictions_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    live_csv_path = Path(args.live_csv_path).expanduser().resolve()

    if not audit_path.is_file():
        raise SystemExit(f"Audit JSONL not found: {audit_path}")
    if not predictions_path.is_file():
        raise SystemExit(f"Predictions JSONL not found: {predictions_path}")

    store = ReviewStore(
        audit_path,
        predictions_path,
        output_dir,
        live_csv_path,
        int(args.speaker_user_id or 0),
        bool(args.include_current_imported_conflicts),
        bool(args.include_live_csv_covered_rows),
    )
    handler = build_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}/"
    print(f"Unresolved IPA review UI: {url}")
    print(f"Audit path: {path_windows(audit_path)}")
    print(f"Predictions path: {path_windows(predictions_path)}")
    print(f"Live CSV path: {path_windows(live_csv_path)}")
    print(f"Output dir: {path_windows(output_dir)}")
    print(f"Import file: {path_windows(store.import_path)}")
    if store.speaker_user_id:
        print(f"Speaker filter: {store.speaker_user_id}")
    else:
        print("Speaker filter: all speakers")
    print(
        "Include current imported conflicts: "
        + ("yes" if store.include_current_imported_conflicts else "no")
    )
    print(
        "Include live CSV covered rows: "
        + ("yes" if store.include_live_csv_covered_rows else "no")
    )
    print(f"Saved decisions: {len(store.decisions_by_recording)} / {len(store.items)}")

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping unresolved IPA review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
