#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from model_paths import stt_model_path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Extract a full-site LL Tools export, build the missing-IPA recording list, and run all four best local models."
    )
    parser.add_argument("--zip-path", required=True, help="Full-site export zip path.")
    parser.add_argument(
        "--extract-root",
        default=str(script_dir / "extracted_bundles_full_site_20260414"),
        help="Where to extract the full-site bundle.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir.parent / "zazacaogren" / "full-site-missing-ipa-four-model-batch-20260414"),
        help="Output directory for the recording list, model outputs, and summaries.",
    )
    parser.add_argument("--speaker-id", default="8", help="speaker_user_id filter.")
    parser.add_argument("--speaker-name", default="A. K.", help="Speaker name label.")
    parser.add_argument(
        "--scope",
        choices=["missing_ipa", "missing_either"],
        default="missing_either",
        help="Which recordings to include for the selected speaker.",
    )
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for the four inference/merge scripts.",
    )
    parser.add_argument(
        "--mms-ipa-model",
        default=str(stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-warmstart-v003-20260407-124628")),
    )
    parser.add_argument(
        "--mms-orth-model",
        default=str(stt_model_path(script_dir, "alternatives", "mms-1b-all-orthography-zza-simple-v001-20260409-071931", "checkpoint-1282")),
    )
    parser.add_argument(
        "--mms-orth-processor-dir",
        default=str(stt_model_path(script_dir, "alternatives", "mms-1b-all-orthography-zza-simple-v001-20260409-071931")),
    )
    parser.add_argument(
        "--whisper-ipa-model",
        default=str(stt_model_path(script_dir, "high-accuracy", "whisper-medium-ipa-nostress-genc-palu-warmstart-v005-20260410-065107", "checkpoint-1254")),
    )
    parser.add_argument(
        "--whisper-ipa-processor-dir",
        default=str(stt_model_path(script_dir, "high-accuracy", "whisper-medium-ipa-nostress-genc-palu-warmstart-v004-20260406")),
    )
    parser.add_argument(
        "--whisper-orth-model",
        default=str(stt_model_path(script_dir, "high-accuracy", "whisper-medium-orth-simple-genc-palu-warmstart-v001-20260410-134348", "checkpoint-2138")),
    )
    parser.add_argument(
        "--whisper-orth-processor-dir",
        default=str(stt_model_path(script_dir, "high-accuracy", "whisper-medium-ipa-nostress-genc-palu-warmstart-v004-20260406")),
    )
    return parser.parse_args()


def to_windows_path(path: Path | str) -> str:
    path_obj = Path(path).expanduser().resolve()
    posix = path_obj.as_posix()
    parts = posix.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper() + ":"
        remainder = "\\".join(parts[3:])
        return drive + ("\\" + remainder if remainder else "")
    return str(path_obj)


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"Unsafe zip entry path: {member.filename}")
        archive.extractall(destination)


def first_meta_value(meta: dict, key: str) -> str:
    value = meta.get(key)
    if isinstance(value, list):
        return str(value[0] or "").strip() if value else ""
    return str(value or "").strip()


def build_category_lookup(bundle_data: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for category in bundle_data.get("categories") or []:
        slug = str(category.get("slug") or "").strip()
        name = str(category.get("name") or "").strip()
        if slug:
            lookup[slug] = name
    return lookup


def build_target_rows(
    *,
    bundle_dir: Path,
    speaker_id: str,
    speaker_name: str,
    scope: str,
    output_tsv: Path,
) -> dict:
    data = json.loads((bundle_dir / "data.json").read_text(encoding="utf-8"))
    site = str(data.get("site") or "").rstrip("/")
    category_lookup = build_category_lookup(data)
    rows: list[dict[str, str]] = []

    for word in data.get("words") or []:
        word_id = str(word.get("origin_id") or "").strip()
        word_title = str(word.get("title") or "").strip()
        word_meta = word.get("meta") or {}
        word_translation = first_meta_value(word_meta, "word_translation")
        categories = [str(item or "").strip() for item in (word.get("categories") or []) if str(item or "").strip()]
        category_slug = "|".join(categories)
        category_name = "|".join(category_lookup.get(item, item) for item in categories)

        for entry in word.get("audio_entries") or []:
            entry_meta = entry.get("meta") or {}
            entry_speaker = first_meta_value(entry_meta, "speaker_user_id")
            if entry_speaker != speaker_id:
                continue

            recording_ipa = first_meta_value(entry_meta, "recording_ipa")
            recording_text = first_meta_value(entry_meta, "recording_text")
            needs_ipa = not recording_ipa
            needs_text = not recording_text
            if scope == "missing_ipa":
                include = needs_ipa
                reason = "missing_recording_ipa" if needs_ipa else ""
                review_bucket = "missing_ipa" if needs_ipa else ""
            else:
                include = needs_ipa or needs_text
                if needs_ipa and needs_text:
                    reason = "missing_recording_ipa_and_text"
                elif needs_ipa:
                    reason = "missing_recording_ipa"
                elif needs_text:
                    reason = "missing_recording_text"
                else:
                    reason = ""
                review_bucket = "missing_transcription" if include else ""
            if not include:
                continue

            audio_file = entry.get("audio_file") or {}
            audio_relpath = str(audio_file.get("file") or "").strip()
            if not audio_relpath:
                continue
            audio_path = bundle_dir / audio_relpath

            recording_type_values = entry.get("recording_types") or []
            recording_type = str(recording_type_values[0] or "").strip() if recording_type_values else ""
            duration_seconds = first_meta_value(entry_meta, "_ll_wordset_games_audio_duration_seconds")

            rows.append(
                {
                    "word_id": word_id,
                    "word_title": word_title,
                    "word_translation": word_translation,
                    "recording_id": str(entry.get("origin_id") or "").strip(),
                    "word_audio_id": "",
                    "recording_type": recording_type,
                    "speaker_user_id": entry_speaker,
                    "speaker_name": speaker_name,
                    "recording_text": recording_text,
                    "recording_translation": "",
                    "recording_ipa": recording_ipa,
                    "reason": reason,
                    "review_bucket": review_bucket,
                    "category_slug": category_slug,
                    "category_name": category_name,
                    "Lexeme": word_translation,
                    "Gloss_tr": word_title,
                    "PhoneticForm": recording_ipa,
                    "all_lexeme_values": word_translation,
                    "all_gloss_values": word_title,
                    "all_phonetic_values": recording_ipa,
                    "lexeme": word_translation,
                    "gloss_tr": word_title,
                    "phonetic_form": recording_ipa,
                    "review-status": review_bucket,
                    "review_status": review_bucket,
                    "audio_relpath": audio_relpath,
                    "audio_url": f"{site}/{audio_relpath}" if site else audio_relpath,
                    "duration_seconds": duration_seconds,
                    "local_audio_found": "1" if audio_path.is_file() else "0",
                }
            )

    rows.sort(key=lambda row: (row["category_slug"], row["word_title"], row["recording_type"], row["recording_id"]))
    fieldnames = [
        "word_id",
        "word_title",
        "word_translation",
        "recording_id",
        "word_audio_id",
        "recording_type",
        "speaker_user_id",
        "speaker_name",
        "recording_text",
        "recording_translation",
        "recording_ipa",
        "reason",
        "review_bucket",
        "category_slug",
        "category_name",
        "Lexeme",
        "Gloss_tr",
        "PhoneticForm",
        "all_lexeme_values",
        "all_gloss_values",
        "all_phonetic_values",
        "lexeme",
        "gloss_tr",
        "phonetic_form",
        "review-status",
        "review_status",
        "audio_relpath",
        "audio_url",
        "duration_seconds",
        "local_audio_found",
    ]
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with output_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "site": site,
        "rows": len(rows),
        "missing_audio_locally": sum(1 for row in rows if row["local_audio_found"] != "1"),
        "categories": len({row["category_slug"] for row in rows if row["category_slug"]}),
        "scope": scope,
    }


def run_step(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    zip_path = Path(args.zip_path).expanduser().resolve()
    extract_root = Path(args.extract_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    python_executable = Path(args.python_executable).expanduser().resolve()
    windows_mode = python_executable.suffix.lower() == ".exe"
    extract_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = extract_root / zip_path.stem
    if not bundle_dir.exists():
        bundle_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path.name} -> {bundle_dir}", flush=True)
        safe_extract(zip_path, bundle_dir)

    scope_label = "missing_ipa" if args.scope == "missing_ipa" else "missing_transcriptions"
    input_tsv = output_dir / f"ak_{scope_label}_rows.tsv"
    build_summary = build_target_rows(
        bundle_dir=bundle_dir,
        speaker_id=str(args.speaker_id),
        speaker_name=args.speaker_name,
        scope=args.scope,
        output_tsv=input_tsv,
    )

    base_stem = f"ak_{scope_label}_20260414"
    mms_ipa_prefix = output_dir / f"{base_stem}_mms_ipa_v003"
    whisper_ipa_prefix = output_dir / f"{base_stem}_whisper_medium_ipa_v005_1254"
    mms_orth_prefix = output_dir / f"{base_stem}_mms_orth_simple_1282"
    whisper_orth_prefix = output_dir / f"{base_stem}_whisper_medium_orth_2138"
    merged_prefix = output_dir / f"{base_stem}_four_model_comparison"

    def fmt(path_like: Path | str) -> str:
        return to_windows_path(path_like) if windows_mode else str(Path(path_like).expanduser().resolve())

    run_step(
        [
            str(python_executable),
            fmt(script_dir / "transcribe_recording_list_with_mms.py"),
            "--input-tsv",
            fmt(input_tsv),
            "--model-source",
            fmt(args.mms_ipa_model),
            "--device",
            args.device,
            "--output-prefix",
            fmt(mms_ipa_prefix),
            "--search-root",
            fmt(bundle_dir),
        ],
    )

    run_step(
        [
            str(python_executable),
            fmt(script_dir / "transcribe_recording_list_with_whisper.py"),
            "--input-tsv",
            fmt(input_tsv),
            "--model-source",
            fmt(args.whisper_ipa_model),
            "--processor-dir",
            fmt(args.whisper_ipa_processor_dir),
            "--device",
            args.device,
            "--target-mode",
            "ipa",
            "--output-prefix",
            fmt(whisper_ipa_prefix),
            "--search-root",
            fmt(bundle_dir),
        ],
    )

    run_step(
        [
            str(python_executable),
            fmt(script_dir / "transcribe_recording_list_with_ctc_text.py"),
            "--input-tsv",
            fmt(input_tsv),
            "--model-dir",
            fmt(args.mms_orth_model),
            "--processor-dir",
            fmt(args.mms_orth_processor_dir),
            "--device",
            args.device,
            "--output-prefix",
            fmt(mms_orth_prefix),
            "--search-root",
            fmt(bundle_dir),
        ],
    )

    run_step(
        [
            str(python_executable),
            fmt(script_dir / "transcribe_recording_list_with_whisper.py"),
            "--input-tsv",
            fmt(input_tsv),
            "--model-source",
            fmt(args.whisper_orth_model),
            "--processor-dir",
            fmt(args.whisper_orth_processor_dir),
            "--device",
            args.device,
            "--target-mode",
            "simple_text",
            "--output-prefix",
            fmt(whisper_orth_prefix),
            "--search-root",
            fmt(bundle_dir),
        ],
    )

    run_step(
        [
            str(python_executable),
            fmt(script_dir / "build_unreviewed_four_model_sheet.py"),
            "--input-tsv",
            fmt(input_tsv),
            "--whisper-ipa-jsonl",
            fmt(whisper_ipa_prefix.with_name(whisper_ipa_prefix.name + "_whisper_predictions.jsonl")),
            "--mms-ipa-tsv",
            fmt(mms_ipa_prefix.with_name(mms_ipa_prefix.name + "_mms_predictions.tsv")),
            "--mms-orth-tsv",
            fmt(mms_orth_prefix.with_name(mms_orth_prefix.name + "_ctc_text_predictions.tsv")),
            "--whisper-orth-tsv",
            fmt(whisper_orth_prefix.with_name(whisper_orth_prefix.name + "_whisper_predictions.tsv")),
            "--output-prefix",
            fmt(merged_prefix),
        ],
    )

    summary = {
        "zip_path": str(zip_path),
        "bundle_dir": str(bundle_dir),
        "speaker_id": str(args.speaker_id),
        "speaker_name": args.speaker_name,
        "scope": args.scope,
        "device": args.device,
        "target_rows": build_summary["rows"],
        "missing_audio_locally": build_summary["missing_audio_locally"],
        "category_count": build_summary["categories"],
        "outputs": {
            "input_tsv": str(input_tsv),
            "mms_ipa_tsv": str(mms_ipa_prefix.with_name(mms_ipa_prefix.name + "_mms_predictions.tsv")),
            "whisper_ipa_jsonl": str(whisper_ipa_prefix.with_name(whisper_ipa_prefix.name + "_whisper_predictions.jsonl")),
            "mms_orth_tsv": str(mms_orth_prefix.with_name(mms_orth_prefix.name + "_ctc_text_predictions.tsv")),
            "whisper_orth_tsv": str(whisper_orth_prefix.with_name(whisper_orth_prefix.name + "_whisper_predictions.tsv")),
            "merged_tsv": str(merged_prefix.with_suffix(".tsv")),
            "merged_jsonl": str(merged_prefix.with_suffix(".jsonl")),
        },
    }
    summary_path = output_dir / f"{base_stem}_four_model_batch.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
