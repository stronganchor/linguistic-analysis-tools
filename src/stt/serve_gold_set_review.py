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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from review_security import ReviewSecurityMixin, configure_review_server, review_html


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_ipa_metadata_path() -> Path:
    return (
        script_dir()
        / "extracted_bundles_whisper_20260410"
        / "ll-tools-stt-training-genc-palu-recording_ipa-20260408-134225"
        / "metadata.jsonl"
    )


def default_text_metadata_path() -> Path:
    return (
        script_dir()
        / "extracted_bundles_text"
        / "ll-tools-stt-training-genc-palu-recording_text-20260409-071931"
        / "metadata.jsonl"
    )


def default_output_dir() -> Path:
    return script_dir() / "gold-set-review-ak-isolation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local browser UI for building an A.K. gold set by reviewing "
            "candidate isolation recordings with IPA and orthography side by side."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--allowed-origin", action="append", default=[], help="Explicit browser origin for a trusted remote/proxy review session; repeatable.")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--ipa-metadata-path", default=str(default_ipa_metadata_path()))
    parser.add_argument("--text-metadata-path", default=str(default_text_metadata_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument(
        "--candidate-jsonl",
        default="",
        help="Optional precomputed candidate JSONL to review instead of rebuilding the full pool.",
    )
    parser.add_argument("--speaker-user-id", type=int, default=8)
    parser.add_argument(
        "--recording-type",
        default="isolation",
        help="Recording type to keep. Default: isolation.",
    )
    parser.add_argument(
        "--include-multiword",
        action="store_true",
        help="Include entries whose IPA or orthography has multiple tokens.",
    )
    parser.add_argument(
        "--include-missing-orthography",
        action="store_true",
        help="Include entries even when no orthography transcription is available.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional cap on the number of candidate items after sorting.",
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


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_for_search(value: str) -> str:
    return normalize_space(value).casefold()


def single_token(value: str) -> bool:
    value = normalize_space(value)
    return bool(value) and " " not in value and "/" not in value and "|" not in value


def count_ipa_vowels(value: str) -> int:
    ipa_vowels = set("aeiouyɨɪʊɛæɑɔəɜʉɐøœ")
    return sum(1 for char in normalize_space(value) if char in ipa_vowels)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def path_windows(path: Path) -> str:
    raw = str(path)
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if match:
        drive, rest = match.groups()
        return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
    return raw.replace("/", "\\")


def maybe_convert_windows_to_wsl(raw: str) -> str | None:
    match = re.match(r"^([a-zA-Z]):\\(.*)$", raw)
    if not match:
        return None
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace('\\', '/')}"


def maybe_convert_wsl_to_windows(raw: str) -> str | None:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", raw)
    if not match:
        return None
    drive, rest = match.groups()
    return f"{drive.upper()}:\\{rest.replace('/', '\\')}"


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
    if path.suffix.lower() == ".html":
        data = review_html(data)
    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream",
    )
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


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


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0") or "0")
    if content_length <= 0:
        return {}
    raw = handler.rfile.read(content_length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class ReviewStore:
    def __init__(
        self,
        ipa_metadata_path: Path,
        text_metadata_path: Path,
        output_dir: Path,
        candidate_jsonl_path: Path | None,
        speaker_user_id: int,
        recording_type: str,
        include_multiword: bool,
        include_missing_orthography: bool,
        max_items: int,
    ):
        self.ipa_metadata_path = ipa_metadata_path
        self.text_metadata_path = text_metadata_path
        self.output_dir = output_dir
        self.candidate_jsonl_path = candidate_jsonl_path
        self.speaker_user_id = speaker_user_id
        self.recording_type = normalize_space(recording_type).lower()
        self.include_multiword = include_multiword
        self.include_missing_orthography = include_missing_orthography
        self.max_items = max(0, int(max_items or 0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_path = script_dir() / "gold_set_review.html"
        self.candidates_path = self.output_dir / "gold-set-candidates.jsonl"
        self.candidates_tsv_path = self.output_dir / "gold-set-candidates.tsv"
        self.confirmed_path = self.output_dir / "gold-set-confirmed.jsonl"
        self.confirmed_tsv_path = self.output_dir / "gold-set-confirmed.tsv"
        self.rejected_path = self.output_dir / "gold-set-rejected.jsonl"
        self.decisions_path = self.output_dir / "gold-set-review-decisions.jsonl"
        self.summary_path = self.output_dir / "gold-set-review-summary.json"

        if self.candidate_jsonl_path:
            self.ipa_rows = []
            self.text_by_recording = {}
            self.skipped_counts = {}
            self.items = self._load_precomputed_items()
        else:
            self.ipa_rows = read_jsonl(self.ipa_metadata_path)
            self.text_by_recording = {
                int(row["recording_id"]): row for row in read_jsonl(self.text_metadata_path)
            }
            self.items = self._load_items()
        self.index_by_recording = {
            int(item["recording_id"]): idx for idx, item in enumerate(self.items)
        }
        self.decisions_by_recording = self._load_existing_decisions()
        self._apply_decisions_to_items()
        self._write_outputs()

    def _choose_orthography(self, ipa_row: dict[str, Any], text_row: dict[str, Any] | None) -> tuple[str, str]:
        candidates = [
            ("recording_text_bundle", normalize_space(str((text_row or {}).get("recording_text") or ""))),
            ("recording_text_inline", normalize_space(str(ipa_row.get("recording_text") or ""))),
            ("word_translation_fallback", normalize_space(str(ipa_row.get("word_translation") or ""))),
        ]
        for source, value in candidates:
            if value:
                return value, source
        return "", ""

    def _resolve_audio_path(self, ipa_row: dict[str, Any]) -> Path | None:
        rel_audio = str(ipa_row.get("audio") or "").strip()
        if not rel_audio:
            return None
        candidate = self.ipa_metadata_path.parent / rel_audio
        if candidate.is_file():
            return candidate.resolve()
        windows_candidate = maybe_convert_windows_to_wsl(rel_audio)
        if windows_candidate:
            path = Path(windows_candidate)
            if path.is_file():
                return path.resolve()
        return None

    def _resolve_audio_reference(self, raw_path: str, rel_audio: str = "") -> Path | None:
        raw_path = str(raw_path or "").strip()
        rel_audio = str(rel_audio or "").strip()

        candidates: list[Path] = []
        if raw_path:
            candidates.append(Path(raw_path))
            wsl_path = maybe_convert_windows_to_wsl(raw_path)
            if wsl_path:
                candidates.append(Path(wsl_path))
            windows_path = maybe_convert_wsl_to_windows(raw_path)
            if windows_path:
                candidates.append(Path(windows_path))

        if rel_audio:
            candidates.append(self.ipa_metadata_path.parent / rel_audio)
            wsl_rel = maybe_convert_windows_to_wsl(rel_audio)
            if wsl_rel:
                candidates.append(Path(wsl_rel))
            windows_rel = maybe_convert_wsl_to_windows(rel_audio)
            if windows_rel:
                candidates.append(Path(windows_rel))

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                try:
                    return candidate.resolve()
                except OSError:
                    return candidate
        return None

    def _candidate_score(self, item: dict[str, Any]) -> int:
        score = 0
        if item["orthography_source"] == "recording_text_bundle":
            score += 12
        elif item["orthography_source"] == "recording_text_inline":
            score += 8
        elif item["orthography_source"] == "word_translation_fallback":
            score += 4
        if item["audio_ext"] == ".wav":
            score += 6
        if 1 <= item["ipa_vowel_count"] <= 3:
            score += 6
        if len(item["recording_ipa"]) <= 10:
            score += 4
        if item["word_translation"]:
            score += 2
        if item["recording_translation"]:
            score += 1
        return score

    def _candidate_tier(self, item: dict[str, Any]) -> str:
        if item["orthography_source"] == "recording_text_bundle" and item["audio_ext"] == ".wav":
            return "preferred"
        if item["orthography_source"] != "word_translation_fallback":
            return "strong"
        return "fallback"

    def _normalize_item_for_client(self, raw_item: dict[str, Any]) -> dict[str, Any]:
        item = dict(raw_item)
        item["recording_id"] = int(item.get("recording_id") or 0)
        item["word_id"] = int(item.get("word_id") or 0)
        item["speaker_user_id"] = int(item.get("speaker_user_id") or 0)
        item["word_title"] = str(item.get("word_title") or "")
        item["word_translation"] = str(item.get("word_translation") or "")
        item["recording_type"] = normalize_space(str(item.get("recording_type") or ""))
        item["speaker_name"] = str(item.get("speaker_name") or "")
        item["recording_ipa"] = normalize_space(str(item.get("recording_ipa") or ""))
        item["orthography_text"] = normalize_space(str(item.get("orthography_text") or ""))
        item["orthography_source"] = str(item.get("orthography_source") or "")
        item["recording_translation"] = str(item.get("recording_translation") or "")
        item["audio_relpath"] = str(item.get("audio_relpath") or "")
        resolved_audio_path = self._resolve_audio_reference(
            raw_path=str(item.get("audio_posix") or ""),
            rel_audio=item["audio_relpath"],
        )
        item["audio_posix"] = str(resolved_audio_path or item.get("audio_posix") or "")
        item["audio_filename"] = str(
            item.get("audio_filename")
            or (resolved_audio_path.name if resolved_audio_path else Path(item["audio_posix"]).name)
        )
        item["audio_ext"] = str(
            item.get("audio_ext") or (resolved_audio_path.suffix.lower() if resolved_audio_path else Path(item["audio_filename"]).suffix.lower())
        )
        item["ipa_vowel_count"] = int(
            item.get("ipa_vowel_count") or count_ipa_vowels(item["recording_ipa"])
        )
        item["candidate_score"] = int(item.get("candidate_score") or 0)
        item["candidate_tier"] = str(item.get("candidate_tier") or self._candidate_tier(item))
        if not item.get("search_blob"):
            item["search_blob"] = normalize_for_search(
                " ".join(
                    [
                        item["word_title"],
                        item["word_translation"],
                        item["orthography_text"],
                        item["recording_ipa"],
                        item["recording_translation"],
                        str(item.get("target_vowels") or ""),
                        str(item.get("selection_bucket") or ""),
                        str(item.get("target_reason") or ""),
                    ]
                )
            )
        item["saved_decision"] = None
        return item

    def _load_precomputed_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not self.candidate_jsonl_path or not self.candidate_jsonl_path.is_file():
            return items
        for row in read_jsonl(self.candidate_jsonl_path):
            try:
                item = self._normalize_item_for_client(row)
            except (TypeError, ValueError, KeyError):
                continue
            if not item["recording_id"] or not item["recording_ipa"] or not item["audio_posix"]:
                continue
            items.append(item)
        if self.max_items:
            items = items[: self.max_items]
        return items

    def _load_items(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        skipped_counts: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1

        for ipa_row in self.ipa_rows:
            try:
                recording_id = int(ipa_row["recording_id"])
            except (KeyError, TypeError, ValueError):
                skip("bad_recording_id")
                continue
            if int(ipa_row.get("speaker_user_id") or 0) != self.speaker_user_id:
                skip("wrong_speaker")
                continue
            recording_types = [normalize_space(str(v)).lower() for v in (ipa_row.get("recording_types") or [])]
            if self.recording_type and self.recording_type not in recording_types:
                skip("wrong_recording_type")
                continue

            recording_ipa = normalize_space(str(ipa_row.get("recording_ipa") or ""))
            if not recording_ipa:
                skip("missing_ipa")
                continue

            text_row = self.text_by_recording.get(recording_id)
            orthography, orth_source = self._choose_orthography(ipa_row, text_row)
            if not orthography and not self.include_missing_orthography:
                skip("missing_orthography")
                continue
            if not self.include_multiword:
                if not single_token(recording_ipa):
                    skip("multiword_ipa")
                    continue
                if orthography and not single_token(orthography):
                    skip("multiword_orthography")
                    continue

            audio_path = self._resolve_audio_path(ipa_row)
            if audio_path is None:
                skip("missing_audio")
                continue

            item = {
                "recording_id": recording_id,
                "word_id": int(ipa_row.get("word_id") or 0),
                "word_title": str(ipa_row.get("word_title") or ""),
                "word_translation": str(ipa_row.get("word_translation") or ""),
                "recording_type": ",".join(recording_types),
                "speaker_user_id": int(ipa_row.get("speaker_user_id") or 0),
                "speaker_name": str(ipa_row.get("speaker_name") or ""),
                "recording_ipa": recording_ipa,
                "orthography_text": orthography,
                "orthography_source": orth_source,
                "recording_translation": str((text_row or {}).get("recording_translation") or ipa_row.get("recording_translation") or ""),
                "audio_relpath": str(ipa_row.get("audio") or ""),
                "audio_posix": str(audio_path),
                "audio_filename": audio_path.name,
                "audio_ext": audio_path.suffix.lower(),
                "ipa_vowel_count": count_ipa_vowels(recording_ipa),
                "search_blob": normalize_for_search(
                    " ".join(
                        [
                            str(ipa_row.get("word_title") or ""),
                            str(ipa_row.get("word_translation") or ""),
                            orthography,
                            recording_ipa,
                            str((text_row or {}).get("recording_translation") or ipa_row.get("recording_translation") or ""),
                        ]
                    )
                ),
                "saved_decision": None,
            }
            item["candidate_score"] = self._candidate_score(item)
            item["candidate_tier"] = self._candidate_tier(item)
            items.append(item)

        items.sort(
            key=lambda item: (
                {"preferred": 0, "strong": 1, "fallback": 2}.get(item["candidate_tier"], 9),
                -int(item["candidate_score"]),
                item["audio_ext"] != ".wav",
                item["word_title"].casefold(),
                int(item["recording_id"]),
            )
        )
        if self.max_items:
            items = items[: self.max_items]
        self.skipped_counts = skipped_counts
        return items

    def _load_existing_decisions(self) -> dict[int, dict[str, Any]]:
        if not self.decisions_path.is_file():
            return {}
        decisions: dict[int, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            try:
                decisions[int(row["recording_id"])] = row
            except (KeyError, TypeError, ValueError):
                continue
        return decisions

    def _apply_decisions_to_items(self) -> None:
        for item in self.items:
            item["saved_decision"] = self.decisions_by_recording.get(int(item["recording_id"]))

    def _sorted_decisions(self) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            if decision:
                ordered.append(decision)
        return ordered

    def _confirmed_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            if not decision or decision.get("status") != "confirmed":
                continue
            rows.append(
                {
                    "word_id": int(item["word_id"]),
                    "recording_id": int(item["recording_id"]),
                    "speaker_user_id": int(item["speaker_user_id"]),
                    "speaker_name": item["speaker_name"],
                    "recording_type": item["recording_type"],
                    "word_title": item["word_title"],
                    "word_translation": item["word_translation"],
                    "orthography_text": item["orthography_text"],
                    "orthography_source": item["orthography_source"],
                    "recording_translation": item["recording_translation"],
                    "recording_ipa": item["recording_ipa"],
                    "audio_relpath": item["audio_relpath"],
                    "audio_posix": item["audio_posix"],
                    "audio_filename": item["audio_filename"],
                    "candidate_score": item["candidate_score"],
                    "candidate_tier": item["candidate_tier"],
                    "review_note": str(decision.get("note") or ""),
                    "review_saved_at": str(decision.get("saved_at") or ""),
                }
            )
        return rows

    def _rejected_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            if not decision or decision.get("status") != "rejected":
                continue
            rows.append(
                {
                    "word_id": int(item["word_id"]),
                    "recording_id": int(item["recording_id"]),
                    "word_title": item["word_title"],
                    "word_translation": item["word_translation"],
                    "orthography_text": item["orthography_text"],
                    "recording_ipa": item["recording_ipa"],
                    "audio_relpath": item["audio_relpath"],
                    "audio_posix": item["audio_posix"],
                    "reason_note": str(decision.get("note") or ""),
                    "review_saved_at": str(decision.get("saved_at") or ""),
                }
            )
        return rows

    def _write_confirmed_tsv(self, rows: list[dict[str, Any]]) -> None:
        with self.confirmed_tsv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "recording_id",
                    "word_id",
                    "speaker_user_id",
                    "speaker_name",
                    "recording_type",
                    "word_title",
                    "word_translation",
                    "orthography_text",
                    "orthography_source",
                    "recording_translation",
                    "recording_ipa",
                    "audio_relpath",
                    "audio_posix",
                    "audio_filename",
                    "candidate_score",
                    "candidate_tier",
                    "review_note",
                    "review_saved_at",
                ]
            )
            for row in rows:
                writer.writerow([row[key] for key in [
                    "recording_id",
                    "word_id",
                    "speaker_user_id",
                    "speaker_name",
                    "recording_type",
                    "word_title",
                    "word_translation",
                    "orthography_text",
                    "orthography_source",
                    "recording_translation",
                    "recording_ipa",
                    "audio_relpath",
                    "audio_posix",
                    "audio_filename",
                    "candidate_score",
                    "candidate_tier",
                    "review_note",
                    "review_saved_at",
                ]])

    def _write_candidates_tsv(self) -> None:
        with self.candidates_tsv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "recording_id",
                    "word_id",
                    "speaker_user_id",
                    "speaker_name",
                    "recording_type",
                    "word_title",
                    "word_translation",
                    "orthography_text",
                    "orthography_source",
                    "recording_translation",
                    "recording_ipa",
                    "audio_relpath",
                    "audio_posix",
                    "audio_filename",
                    "audio_ext",
                    "ipa_vowel_count",
                    "candidate_score",
                    "candidate_tier",
                    "saved_status",
                    "saved_note",
                ]
            )
            for item in self.items:
                decision = item.get("saved_decision") or {}
                writer.writerow(
                    [
                        item["recording_id"],
                        item["word_id"],
                        item["speaker_user_id"],
                        item["speaker_name"],
                        item["recording_type"],
                        item["word_title"],
                        item["word_translation"],
                        item["orthography_text"],
                        item["orthography_source"],
                        item["recording_translation"],
                        item["recording_ipa"],
                        item["audio_relpath"],
                        item["audio_posix"],
                        item["audio_filename"],
                        item["audio_ext"],
                        item["ipa_vowel_count"],
                        item["candidate_score"],
                        item["candidate_tier"],
                        decision.get("status", ""),
                        decision.get("note", ""),
                    ]
                )

    def _write_outputs(self) -> None:
        decisions = self._sorted_decisions()
        confirmed = self._confirmed_rows()
        rejected = self._rejected_rows()
        write_jsonl(self.decisions_path, decisions)
        write_jsonl(self.candidates_path, self.items)
        self._write_candidates_tsv()
        write_jsonl(self.confirmed_path, confirmed)
        write_jsonl(self.rejected_path, rejected)
        self._write_confirmed_tsv(confirmed)
        summary = {
            "generated_at": iso_now(),
            "candidate_jsonl_path": path_windows(self.candidate_jsonl_path) if self.candidate_jsonl_path else "",
            "ipa_metadata_path": path_windows(self.ipa_metadata_path),
            "text_metadata_path": path_windows(self.text_metadata_path),
            "output_dir": path_windows(self.output_dir),
            "speaker_user_id": self.speaker_user_id,
            "recording_type": self.recording_type,
            "include_multiword": self.include_multiword,
            "include_missing_orthography": self.include_missing_orthography,
            "max_items": self.max_items,
            "candidate_items": len(self.items),
            "confirmed_items": len(confirmed),
            "rejected_items": len(rejected),
            "pending_items": len(self.items) - len(confirmed) - len(rejected),
            "candidate_tiers": {
                tier: sum(1 for item in self.items if item["candidate_tier"] == tier)
                for tier in ("preferred", "strong", "fallback")
            },
            "orthography_sources": {
                source: sum(1 for item in self.items if item["orthography_source"] == source)
                for source in ("recording_text_bundle", "recording_text_inline", "word_translation_fallback", "")
            },
            "audio_extensions": {
                ext or "<none>": sum(1 for item in self.items if item["audio_ext"] == ext)
                for ext in sorted({item["audio_ext"] for item in self.items})
            },
            "skipped_counts": self.skipped_counts,
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def client_state(self) -> dict[str, Any]:
        confirmed = self._confirmed_rows()
        rejected = self._rejected_rows()
        return {
            "generated_at": iso_now(),
            "speaker_user_id": self.speaker_user_id,
            "recording_type": self.recording_type,
            "candidate_items": len(self.items),
            "confirmed_items": len(confirmed),
            "rejected_items": len(rejected),
            "pending_items": len(self.items) - len(confirmed) - len(rejected),
            "paths": {
                "candidate_jsonl_path": path_windows(self.candidate_jsonl_path) if self.candidate_jsonl_path else "",
                "ipa_metadata_path": path_windows(self.ipa_metadata_path),
                "text_metadata_path": path_windows(self.text_metadata_path),
                "output_dir": path_windows(self.output_dir),
                "candidates_path": path_windows(self.candidates_path),
                "candidates_tsv_path": path_windows(self.candidates_tsv_path),
                "confirmed_path": path_windows(self.confirmed_path),
                "confirmed_tsv_path": path_windows(self.confirmed_tsv_path),
                "rejected_path": path_windows(self.rejected_path),
                "decisions_path": path_windows(self.decisions_path),
                "summary_path": path_windows(self.summary_path),
            },
            "items": self.items,
        }

    def set_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        recording_id = int(payload["recording_id"])
        idx = self.index_by_recording[recording_id]
        item = self.items[idx]
        status = normalize_space(str(payload.get("status") or "")).lower()
        if status not in {"confirmed", "rejected"}:
            raise ValueError("Status must be `confirmed` or `rejected`.")

        decision = {
            "recording_id": recording_id,
            "word_id": int(item["word_id"]),
            "status": status,
            "note": normalize_space(str(payload.get("note") or "")),
            "saved_at": iso_now(),
        }
        self.decisions_by_recording[recording_id] = decision
        item["saved_decision"] = decision
        self._write_outputs()
        return {
            "ok": True,
            "decision": decision,
            "candidate_items": len(self.items),
            "confirmed_items": len(self._confirmed_rows()),
            "rejected_items": len(self._rejected_rows()),
        }

    def clear_decision(self, recording_id: int) -> dict[str, Any]:
        self.decisions_by_recording.pop(recording_id, None)
        idx = self.index_by_recording[recording_id]
        self.items[idx]["saved_decision"] = None
        self._write_outputs()
        return {
            "ok": True,
            "candidate_items": len(self.items),
            "confirmed_items": len(self._confirmed_rows()),
            "rejected_items": len(self._rejected_rows()),
        }

    def audio_path_for_recording(self, recording_id: int) -> Path | None:
        idx = self.index_by_recording.get(recording_id)
        if idx is None:
            return None
        item = self.items[idx]
        return self._resolve_audio_reference(
            raw_path=str(item.get("audio_posix") or ""),
            rel_audio=str(item.get("audio_relpath") or ""),
        )


def build_handler(store: ReviewStore):
    class ReviewHandler(ReviewSecurityMixin, BaseHTTPRequestHandler):
        server_version = "GoldSetReview/1.0"

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
            if route == "/api/download-confirmed":
                data = store.confirmed_path.read_bytes() if store.confirmed_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{store.confirmed_path.name}"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route == "/api/download-candidates":
                data = store.candidates_path.read_bytes() if store.candidates_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="{store.candidates_path.name}"',
                )
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route.startswith("/audio/"):
                try:
                    recording_id = int(route.rsplit("/", 1)[-1])
                except ValueError:
                    send_text(self, "Invalid recording id.", status=400)
                    return
                audio_path = store.audio_path_for_recording(recording_id)
                if audio_path is None:
                    send_text(self, "Audio not found.", status=404)
                    return
                send_audio(self, audio_path)
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
                send_json(self, store.clear_decision(recording_id))
                return

            send_json(self, {"ok": False, "error": "Not found."}, status=404)

    return ReviewHandler


def main() -> None:
    args = parse_args()
    ipa_metadata_path = Path(args.ipa_metadata_path).expanduser().resolve()
    text_metadata_path = Path(args.text_metadata_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    candidate_jsonl_path = Path(args.candidate_jsonl).expanduser().resolve() if args.candidate_jsonl else None

    if candidate_jsonl_path:
        if not candidate_jsonl_path.is_file():
            raise SystemExit(f"Candidate JSONL not found: {candidate_jsonl_path}")
    else:
        if not ipa_metadata_path.is_file():
            raise SystemExit(f"IPA metadata JSONL not found: {ipa_metadata_path}")
        if not text_metadata_path.is_file():
            raise SystemExit(f"Text metadata JSONL not found: {text_metadata_path}")

    store = ReviewStore(
        ipa_metadata_path=ipa_metadata_path,
        text_metadata_path=text_metadata_path,
        output_dir=output_dir,
        candidate_jsonl_path=candidate_jsonl_path,
        speaker_user_id=int(args.speaker_user_id),
        recording_type=str(args.recording_type),
        include_multiword=bool(args.include_multiword),
        include_missing_orthography=bool(args.include_missing_orthography),
        max_items=int(args.max_items or 0),
    )
    handler = build_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    configure_review_server(server, args.allowed_origin, args.host)

    url = f"http://{args.host}:{args.port}/"
    print(f"Gold-set review UI: {url}")
    if candidate_jsonl_path:
        print(f"Candidate shortlist: {path_windows(candidate_jsonl_path)}")
    else:
        print(f"IPA metadata: {path_windows(ipa_metadata_path)}")
        print(f"Text metadata: {path_windows(text_metadata_path)}")
    print(f"Output dir: {path_windows(output_dir)}")
    print(f"Candidate items: {len(store.items)}")
    print(f"Confirmed: {len(store._confirmed_rows())}")
    print(f"Rejected: {len(store._rejected_rows())}")
    print(f"Pending: {len(store.items) - len(store._confirmed_rows()) - len(store._rejected_rows())}")

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping gold-set review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
