#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from review_lltools_ipa_comparison import normalize_title_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a recording_ipa import file from the full new-MMS pass for rows that are "
            "still missing according to a live-site Lexeme/Gloss_tr CSV export."
        )
    )
    parser.add_argument(
        "--new-path",
        default=str(
            root
            / "zazacaogren"
            / "full-audio-model-review-april-2026"
            / "new-mms-v003-all-audio.jsonl"
        ),
    )
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the live-site wordset text CSV export with Lexeme/PhoneticForm/Gloss_tr.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--audit-path", required=True)
    parser.add_argument(
        "--speaker-user-id",
        type=int,
        default=0,
        help="Optional speaker filter. Use 0 to include all speakers.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def windows_path(path: Path) -> str:
    raw = str(path)
    if raw.startswith("/mnt/") and len(raw) > 6:
        drive = raw[5]
        rest = raw[7:]
        return f"{drive.upper()}:\\{rest.replace('/', '\\')}"
    return raw.replace("/", "\\")


def load_live_csv_index(csv_path: Path) -> tuple[set[str], dict[str, list[dict[str, str]]]]:
    covered_keys: set[str] = set()
    matches_by_key: dict[str, list[dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phonetic = (row.get("PhoneticForm") or "").strip()
            if not phonetic:
                continue
            compact = {
                "Lexeme": (row.get("Lexeme") or "").strip(),
                "Gloss_tr": (row.get("Gloss_tr") or "").strip(),
                "PhoneticForm": phonetic,
            }
            for field in ("Lexeme", "Gloss_tr"):
                key = normalize_title_text(row.get(field) or "")
                if not key:
                    continue
                covered_keys.add(key)
                matches_by_key.setdefault(key, []).append(compact)
    return covered_keys, matches_by_key


def main() -> None:
    args = parse_args()
    new_path = Path(args.new_path).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = Path(args.summary_path).expanduser().resolve()
    audit_path = Path(args.audit_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    covered_keys, matches_by_key = load_live_csv_index(csv_path)

    import_rows: list[dict] = []
    audit_rows: list[dict] = []
    by_speaker = Counter()
    by_type = Counter()
    skipped_existing = 0
    total_new_rows = 0

    with new_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            total_new_rows += 1
            speaker_user_id = int(row.get("speaker_user_id") or 0)
            if args.speaker_user_id and speaker_user_id != args.speaker_user_id:
                continue

            keys = {
                normalize_title_text(row.get("word_title") or ""),
                normalize_title_text(row.get("word_translation") or ""),
            }
            keys.discard("")
            matched = []
            for key in sorted(keys):
                matched.extend(matches_by_key.get(key, []))
            if matched:
                skipped_existing += 1
                continue

            recording_ipa = str(row.get("recording_ipa") or "").strip()
            if not recording_ipa:
                continue

            import_rows.append(
                {
                    "word_id": row.get("word_id"),
                    "recording_id": row.get("recording_id"),
                    "recording_ipa": recording_ipa,
                    "word_title": row.get("word_title") or "",
                    "word_translation": row.get("word_translation") or "",
                }
            )
            audit_rows.append(
                {
                    "word_id": row.get("word_id"),
                    "recording_id": row.get("recording_id"),
                    "recording_ipa": recording_ipa,
                    "word_title": row.get("word_title") or "",
                    "word_translation": row.get("word_translation") or "",
                    "speaker_user_id": speaker_user_id,
                    "recording_type": row.get("recording_type") or "",
                    "audio_windows": row.get("audio_windows") or "",
                    "audio_posix": row.get("audio_posix") or "",
                    "source_bundle": row.get("source_bundle") or "",
                    "category_name": row.get("category_name") or "",
                    "matched_live_csv_entries": matched,
                }
            )
            by_speaker[speaker_user_id] += 1
            by_type[str(row.get("recording_type") or "")] += 1

    write_jsonl(output_path, import_rows)
    write_jsonl(audit_path, audit_rows)
    summary = {
        "new_path": windows_path(new_path),
        "csv_path": windows_path(csv_path),
        "output_path": windows_path(output_path),
        "audit_path": windows_path(audit_path),
        "speaker_user_id_filter": int(args.speaker_user_id or 0),
        "total_new_rows_scanned": total_new_rows,
        "covered_live_csv_keys": len(covered_keys),
        "skipped_existing_live_csv_rows": skipped_existing,
        "import_rows": len(import_rows),
        "by_speaker": dict(by_speaker),
        "by_recording_type": dict(by_type),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
