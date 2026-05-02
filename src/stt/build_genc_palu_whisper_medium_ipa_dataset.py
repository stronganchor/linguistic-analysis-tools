import argparse
import csv
import json
import shutil
import wave
import zipfile
from collections import Counter
from pathlib import Path

from prepare_lltools_ipa_dataset import (
    choose_split,
    load_training_blacklist,
    blacklist_match,
    normalize_ipa_text,
    normalize_plain_text,
    to_windows_path,
)


MANIFEST_HEADERS = [
    "split",
    "audio",
    "audio_windows",
    "audio_posix",
    "target_ipa",
    "recording_text",
    "recording_translation",
    "recording_type",
    "word_title",
    "word_slug",
    "categories",
    "category_names",
    "speaker_user_id",
    "source_bundle",
    "audio_relpath",
]


EXTRA_EXCLUDED_RECORDING_IDS = {
    "2650", "3001", "3002",  # İnek watchlist family
    "15475", "15476", "15477",  # Mısır unu watchlist family
    "11926", "11928",  # Balıkçı watchlist family
    "15765",  # Ben de iyiyim watchlist
    "5523", "5524", "5525",  # Un watchlist family
    "2772",  # Agiz question unresolved family mismatch
    "14599",  # Kalem question unresolved onset mismatch
    "7213",  # Kumru question unresolved onset mismatch
    "10018", "10019", "10020",  # Altın family unresolved i/ɨ alternation
    "12062", "12063", "12064",  # 11 family unresolved jundas/jondas alternation
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build a filtered Genç-Palu audio->IPA manifest for Whisper-medium retraining."
    )
    parser.add_argument(
        "--recording-zip",
        required=True,
        help="Path to the LL Tools recording_ipa zip export.",
    )
    parser.add_argument(
        "--wordset-csv",
        required=True,
        help="Path to the latest live wordset metadata export used as the lexical baseline.",
    )
    parser.add_argument(
        "--extract-root",
        default=str(script_dir / "extracted_bundles_whisper"),
        help="Where the recording bundle should be extracted.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where manifest_ipa.tsv and summary files should be written.",
    )
    parser.add_argument(
        "--blacklist",
        default=str(script_dir / "training_blacklists" / "genc-palu-manual-training-blacklist.json"),
        help="Manual training blacklist JSON.",
    )
    parser.add_argument(
        "--audit-import",
        default=str(
            script_dir.parent
            / "zazacaogren"
            / "training-v4-audit-april-2026"
            / "mms-v4-training-audit-high-confidence-import.jsonl"
        ),
        help="JSONL of high-confidence IPA corrections to overlay before training.",
    )
    parser.add_argument(
        "--speaker-id",
        default="8",
        help="speaker_user_id to include. Default is A.K. speaker 8.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of lexical items assigned to eval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic word-level split seed.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract the recording zip even if it is already present.",
    )
    return parser.parse_args()


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"Unsafe zip entry path: {member.filename}")
        archive.extractall(destination)


def ensure_extracted(zip_path: Path, extract_root: Path, force: bool) -> Path:
    destination = extract_root / zip_path.stem
    if destination.exists() and force:
        shutil.rmtree(destination)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)
        safe_extract(zip_path, destination)
    return destination


def resolve_bundle_root(extracted_dir: Path) -> Path:
    direct = extracted_dir / "metadata.csv"
    if direct.is_file():
        return extracted_dir
    matches = sorted(extracted_dir.rglob("metadata.csv"))
    if not matches:
        raise FileNotFoundError(f"No metadata.csv found under {extracted_dir}")
    return matches[0].parent


def load_wordset_csv(path: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lexeme = normalize_plain_text(str(row.get("Lexeme") or ""))
            phonetic = normalize_ipa_text(str(row.get("PhoneticForm") or ""))
            if lexeme and lexeme not in lookup:
                lookup[lexeme] = {
                    "phonetic_form": phonetic,
                    "gloss_tr": normalize_plain_text(str(row.get("Gloss_tr") or "")),
                }
    return lookup


def load_audit_overrides(path: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not path.is_file():
        return overrides
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            recording_id = normalize_plain_text(str(row.get("recording_id") or ""))
            recording_ipa = normalize_ipa_text(str(row.get("recording_ipa") or ""))
            if recording_id and recording_ipa:
                overrides[recording_id] = recording_ipa
    return overrides


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return 0.0 if rate <= 0 else frames / float(rate)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    recording_zip = Path(args.recording_zip).expanduser().resolve()
    wordset_csv = Path(args.wordset_csv).expanduser().resolve()
    extract_root = Path(args.extract_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    blacklist_path = Path(args.blacklist).expanduser().resolve()
    audit_import_path = Path(args.audit_import).expanduser().resolve()

    extract_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    blacklist = load_training_blacklist(blacklist_path)
    wordset_lookup = load_wordset_csv(wordset_csv)
    audit_overrides = load_audit_overrides(audit_import_path)

    extracted = ensure_extracted(recording_zip, extract_root, args.force_extract)
    bundle_root = resolve_bundle_root(extracted)
    metadata_csv = bundle_root / "metadata.csv"

    rows: list[dict[str, str]] = []
    exclusions: list[dict[str, str]] = []
    total_seconds = 0.0
    by_reason = Counter()

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            if normalize_plain_text(str(entry.get("text_field") or "")) != "recording_ipa":
                continue

            speaker_id = normalize_plain_text(str(entry.get("speaker_user_id") or ""))
            if speaker_id != str(args.speaker_id):
                exclusions.append(
                    {
                        "recording_id": normalize_plain_text(str(entry.get("recording_id") or "")),
                        "word_id": normalize_plain_text(str(entry.get("word_id") or "")),
                        "word_title": normalize_plain_text(str(entry.get("word_title") or "")),
                        "reason": "speaker_filter",
                    }
                )
                by_reason["speaker_filter"] += 1
                continue

            recording_id = normalize_plain_text(str(entry.get("recording_id") or ""))
            word_id = normalize_plain_text(str(entry.get("word_id") or ""))
            word_title = normalize_plain_text(str(entry.get("word_title") or ""))

            match_reason = blacklist_match(
                blacklist,
                recording_id=recording_id,
                word_id=word_id,
                word_title=word_title,
            )
            if match_reason:
                exclusions.append(
                    {
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": f"manual_blacklist:{match_reason}",
                    }
                )
                by_reason[f"manual_blacklist:{match_reason}"] += 1
                continue

            if recording_id in EXTRA_EXCLUDED_RECORDING_IDS:
                exclusions.append(
                    {
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "watchlist_or_unresolved_family",
                    }
                )
                by_reason["watchlist_or_unresolved_family"] += 1
                continue

            raw_ipa = audit_overrides.get(
                recording_id,
                normalize_ipa_text(str(entry.get("recording_ipa") or entry.get("text") or "")),
            )
            if not raw_ipa:
                exclusions.append(
                    {
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "missing_target_ipa",
                    }
                )
                by_reason["missing_target_ipa"] += 1
                continue

            audio_relpath = normalize_plain_text(str(entry.get("audio") or ""))
            audio_path = bundle_root / audio_relpath
            if not audio_path.is_file():
                exclusions.append(
                    {
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": "missing_audio",
                    }
                )
                by_reason["missing_audio"] += 1
                continue

            word_key = word_id or word_title
            split = choose_split(word_key, args.validation_fraction, args.seed)
            total_seconds += wav_duration_seconds(audio_path)

            rows.append(
                {
                    "split": split,
                    "audio": to_windows_path(audio_path),
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "target_ipa": raw_ipa,
                    "recording_text": normalize_plain_text(str(entry.get("recording_text") or "")),
                    "recording_translation": normalize_plain_text(str(entry.get("recording_translation") or "")),
                    "recording_type": normalize_plain_text(str(entry.get("recording_types") or "")) or "unknown",
                    "word_title": word_title,
                    "word_slug": "",
                    "categories": "",
                    "category_names": "",
                    "speaker_user_id": speaker_id,
                    "source_bundle": recording_zip.name,
                    "audio_relpath": audio_relpath,
                }
            )

    rows.sort(key=lambda row: (row["split"], row["word_title"], row["recording_type"], row["audio_relpath"]))

    manifest_path = output_dir / "manifest_ipa.tsv"
    summary_path = output_dir / "dataset_summary.json"
    exclusions_path = output_dir / "excluded_rows.jsonl"

    write_manifest(manifest_path, rows)

    with exclusions_path.open("w", encoding="utf-8") as handle:
        for row in exclusions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "recording_zip": str(recording_zip),
        "wordset_csv_base": str(wordset_csv),
        "metadata_csv": str(metadata_csv),
        "audit_import_overlay": str(audit_import_path) if audit_import_path.is_file() else "",
        "manual_blacklist_path": str(blacklist_path),
        "speaker_id": str(args.speaker_id),
        "total_examples": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "by_recording_type": dict(Counter(row["recording_type"] for row in rows)),
        "total_audio_seconds": total_seconds,
        "total_audio_minutes": total_seconds / 60.0,
        "excluded_examples": len(exclusions),
        "excluded_by_reason": dict(by_reason),
        "wordset_rows_loaded": len(wordset_lookup),
        "audit_overrides_applied": len(audit_overrides),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote exclusions: {exclusions_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
