#!/usr/bin/env python3
import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

from prepare_lltools_ipa_dataset import (
    choose_split,
    extract_bundle,
    list_zip_paths,
    load_training_blacklist,
    normalize_plain_text,
    resolve_bundle_content_root,
    to_windows_path,
)


MANIFEST_HEADERS = [
    "split",
    "audio",
    "audio_windows",
    "audio_posix",
    "target_text",
    "recording_text",
    "recording_translation",
    "recording_type",
    "word_title",
    "word_slug",
    "speaker_user_id",
    "source_bundle",
    "audio_relpath",
]

ALLOWED_ORTHOGRAPHY_RE = re.compile(r"^[A-Za-zÇçĞğİIıÖöŞşÜüÂâÊêÎîÛû'.,?()0-9+\- ]+$")
SENTENCE_PUNCT_RE = re.compile(r"[.?!]+")
Z_LOWER_MAP = str.maketrans(
    {
        "I": "ı",
        "İ": "i",
        "Â": "â",
        "Ê": "ê",
        "Û": "û",
        "Ç": "ç",
        "Ğ": "ğ",
        "Ş": "ş",
        "Ü": "ü",
        "Ö": "ö",
        "Î": "î",
    }
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_blacklist = script_dir / "training_blacklists" / "genc-palu-manual-training-blacklist.json"
    default_import = (
        script_dir.parent
        / "zazacaogren"
        / "orthography-text-fill-april-2026"
        / "genc-palu-recording-text-orthography-import-recording-text-only.jsonl"
    )
    default_review_flags = (
        script_dir.parent
        / "zazacaogren"
        / "orthography-text-fill-april-2026"
        / "genc-palu-recording-text-orthography-import-recording-text-only.review_flags.jsonl"
    )
    default_output_dir = script_dir / "training_data" / "genc-palu-recording_text-20260409-071931-filtered"
    parser = argparse.ArgumentParser(description="Build a filtered LL Tools audio->orthography manifest.")
    parser.add_argument("--zip", action="append", dest="zip_paths", required=True, help="Specific LL Tools training zip(s).")
    parser.add_argument("--extract-dir", default=str(script_dir / "extracted_bundles_text"), help="Where extracted bundle contents should be stored.")
    parser.add_argument("--output-dir", default=str(default_output_dir), help="Where manifest and summaries should be written.")
    parser.add_argument("--speaker-id", type=int, default=8, help="speaker_user_id filter. Use 0 to include all.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--blacklist", default=str(default_blacklist) if default_blacklist.is_file() else "")
    parser.add_argument("--orthography-import", default=str(default_import) if default_import.is_file() else "")
    parser.add_argument("--review-flags", default=str(default_review_flags) if default_review_flags.is_file() else "")
    parser.add_argument("--lowercase-targets", action="store_true", help="Lowercase recording_text targets with Zazaki-aware casing.")
    parser.add_argument("--strip-sentence-punctuation", action="store_true", help="Remove . ? ! from target text.")
    parser.add_argument("--map-a-circumflex-to-e", action="store_true", help="Map â to e after other text normalization.")
    parser.add_argument(
        "--exclude-target-chars",
        default="",
        help="Exclude rows whose normalized target contains any of these characters.",
    )
    return parser.parse_args()


def load_recording_text_overrides(path: Path | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if path is None or not path.is_file():
        return overrides
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            recording_id = normalize_plain_text(str(row.get("recording_id") or ""))
            recording_text = normalize_plain_text(str(row.get("recording_text") or ""))
            if recording_id and recording_text:
                overrides[recording_id] = recording_text
    return overrides


def load_review_flag_ids(path: Path | None) -> set[str]:
    flagged: set[str] = set()
    if path is None or not path.is_file():
        return flagged
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            recording_id = normalize_plain_text(str(row.get("recording_id") or ""))
            if recording_id:
                flagged.add(recording_id)
    return flagged


def blacklist_match(blacklist: dict[str, object], recording_id: str, word_id: str, word_title: str) -> str:
    if recording_id and recording_id in (blacklist.get("recording_ids") or set()):
        return "manual_blacklist_recording"
    if word_id and word_id in (blacklist.get("word_ids") or set()):
        return "manual_blacklist_word"
    if word_title and word_title.casefold() in (blacklist.get("word_titles") or set()):
        return "manual_blacklist_word_title"
    return ""


def detect_suspicious_text(text: str) -> str:
    if not text:
        return "empty_recording_text"
    if not ALLOWED_ORTHOGRAPHY_RE.match(text):
        return "unexpected_orthography_symbol"
    return ""


def zazaki_lower(text: str) -> str:
    return text.translate(Z_LOWER_MAP).lower()


def simplify_target_text(
    text: str,
    *,
    lowercase_targets: bool,
    strip_sentence_punctuation: bool,
    map_a_circumflex_to_e: bool,
) -> str:
    value = normalize_plain_text(text)
    if lowercase_targets:
        value = zazaki_lower(value)
    if strip_sentence_punctuation:
        value = SENTENCE_PUNCT_RE.sub("", value)
    value = normalize_plain_text(value)
    if map_a_circumflex_to_e:
        value = value.replace("â", "e").replace("Â", "e")
    return normalize_plain_text(value)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    extract_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    blacklist_path = Path(args.blacklist).expanduser().resolve() if args.blacklist else None
    blacklist = load_training_blacklist(blacklist_path)
    overrides = load_recording_text_overrides(Path(args.orthography_import).expanduser().resolve() if args.orthography_import else None)
    review_flag_ids = load_review_flag_ids(Path(args.review_flags).expanduser().resolve() if args.review_flags else None)
    excluded_target_chars = set(args.exclude_target_chars or "")

    all_rows: list[dict[str, str]] = []
    exclusion_rows: list[dict[str, str]] = []
    summary = Counter()
    zip_paths = list_zip_paths(output_dir, args.zip_paths)

    for zip_path in zip_paths:
        extracted_bundle_dir = extract_bundle(zip_path, extract_dir, args.force_extract)
        bundle_content_dir = resolve_bundle_content_root(extracted_bundle_dir)
        metadata_csv_path = bundle_content_dir / "metadata.csv"
        if not metadata_csv_path.is_file():
            raise SystemExit(f"Bundle missing metadata.csv: {zip_path}")

        with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for entry in reader:
                summary["rows_seen"] += 1
                recording_id = normalize_plain_text(str(entry.get("recording_id") or ""))
                word_id = normalize_plain_text(str(entry.get("word_id") or ""))
                word_title = normalize_plain_text(str(entry.get("word_title") or ""))
                word_slug = normalize_plain_text(str(entry.get("word_slug") or ""))
                speaker_raw = normalize_plain_text(str(entry.get("speaker_user_id") or ""))
                speaker_id = int(speaker_raw) if speaker_raw.isdigit() else 0
                if args.speaker_id > 0 and speaker_id != args.speaker_id:
                    summary["excluded_speaker_filter"] += 1
                    exclusion_rows.append({
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "speaker_filter",
                    })
                    continue

                blacklist_reason = blacklist_match(blacklist, recording_id, word_id, word_title)
                if blacklist_reason:
                    summary[blacklist_reason] += 1
                    exclusion_rows.append({
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": blacklist_reason,
                    })
                    continue

                if recording_id in review_flag_ids:
                    summary["excluded_review_flags"] += 1
                    exclusion_rows.append({
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "orthography_review_flag",
                    })
                    continue

                recording_text = normalize_plain_text(overrides.get(recording_id, str(entry.get("recording_text") or "")))
                recording_text = simplify_target_text(
                    recording_text,
                    lowercase_targets=args.lowercase_targets,
                    strip_sentence_punctuation=args.strip_sentence_punctuation,
                    map_a_circumflex_to_e=args.map_a_circumflex_to_e,
                )
                suspicious_reason = detect_suspicious_text(recording_text)
                if suspicious_reason:
                    summary[suspicious_reason] += 1
                    exclusion_rows.append({
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": suspicious_reason,
                        "recording_text": recording_text,
                    })
                    continue

                if excluded_target_chars and any(ch in recording_text for ch in excluded_target_chars):
                    summary["excluded_target_chars"] += 1
                    exclusion_rows.append({
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "excluded_target_chars",
                        "recording_text": recording_text,
                    })
                    continue

                audio_relpath = normalize_plain_text(str(entry.get("audio") or ""))
                if not audio_relpath:
                    summary["excluded_missing_audio_path"] += 1
                    continue
                audio_path = bundle_content_dir / audio_relpath
                if not audio_path.is_file():
                    summary["excluded_missing_audio_file"] += 1
                    continue

                recording_type = normalize_plain_text(str(entry.get("recording_types") or "")).split("|")[0] or "unknown"
                split_key = word_slug or word_id or word_title
                split = choose_split(split_key, args.validation_fraction, args.seed)
                all_rows.append({
                    "split": split,
                    "audio": to_windows_path(audio_path),
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "target_text": recording_text,
                    "recording_text": recording_text,
                    "recording_translation": normalize_plain_text(str(entry.get("recording_translation") or "")),
                    "recording_type": recording_type,
                    "word_title": word_title,
                    "word_slug": word_slug,
                    "speaker_user_id": str(speaker_id),
                    "source_bundle": zip_path.name,
                    "audio_relpath": audio_relpath,
                })
                summary["rows_kept"] += 1

    all_rows.sort(key=lambda row: (row["split"], row["word_slug"], row["recording_type"], row["audio_windows"]))
    manifest_path = output_dir / "manifest_text.tsv"
    summary_path = output_dir / "dataset_summary.json"
    exclusions_path = output_dir / "excluded_rows.jsonl"
    write_manifest(manifest_path, all_rows)

    with exclusions_path.open("w", encoding="utf-8") as handle:
        for row in exclusion_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_data = {
        "bundle_files": [str(Path(p).name) for p in args.zip_paths],
        "total_rows_seen": summary["rows_seen"],
        "rows_kept": summary["rows_kept"],
        "excluded_speaker_filter": summary["excluded_speaker_filter"],
        "excluded_review_flags": summary["excluded_review_flags"],
        "excluded_missing_audio_path": summary["excluded_missing_audio_path"],
        "excluded_missing_audio_file": summary["excluded_missing_audio_file"],
        "manual_blacklist_recording": summary["manual_blacklist_recording"],
        "manual_blacklist_word": summary["manual_blacklist_word"],
        "manual_blacklist_word_title": summary["manual_blacklist_word_title"],
        "unexpected_orthography_symbol": summary["unexpected_orthography_symbol"],
        "empty_recording_text": summary["empty_recording_text"],
        "excluded_target_chars": summary["excluded_target_chars"],
        "train_examples": sum(1 for row in all_rows if row["split"] == "train"),
        "eval_examples": sum(1 for row in all_rows if row["split"] == "eval"),
        "review_flags_source": str(Path(args.review_flags).expanduser().resolve()) if args.review_flags else "",
        "orthography_import_source": str(Path(args.orthography_import).expanduser().resolve()) if args.orthography_import else "",
        "blacklist_path": str(blacklist_path) if blacklist_path else "",
        "lowercase_targets": bool(args.lowercase_targets),
        "strip_sentence_punctuation": bool(args.strip_sentence_punctuation),
        "map_a_circumflex_to_e": bool(args.map_a_circumflex_to_e),
        "excluded_target_characters": "".join(sorted(excluded_target_chars)),
    }
    summary_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(f"Wrote excluded: {exclusions_path}")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
