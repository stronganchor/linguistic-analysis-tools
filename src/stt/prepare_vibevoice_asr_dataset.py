#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path

import soundfile as sf


DEFAULT_CONTEXT = [
    "Central Zazaki IPA transcription.",
    "The speaker is A.K. (speaker_user_id 8), Genç Central Zazaki.",
    "Use IPA symbols exactly as heard; preserve affricates, vowels, and dental stop marks.",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert an LL Tools IPA manifest into VibeVoice-ASR LoRA JSON/audio pairs."
    )
    parser.add_argument(
        "--manifest",
        default=str(script_dir / "training_data" / "manifest_ipa.tsv"),
        help="Input manifest_ipa.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "training_data" / "vibevoice_asr_ak_ipa_v001"),
        help="Output dataset folder. Split subfolders are created inside it.",
    )
    parser.add_argument("--speaker-id", default="8", help="speaker_user_id to include.")
    parser.add_argument(
        "--target-column",
        default="target_ipa",
        help="Manifest column to use as the target transcript.",
    )
    parser.add_argument(
        "--context-line",
        action="append",
        default=[],
        help="Customized context line to include in each JSON label. Repeat for multiple lines.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=["train", "eval"],
        help="Optional split(s) to export. Defaults to train and eval.",
    )
    parser.add_argument(
        "--audio-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to place audio files in each split folder.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing output folder.")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def resolve_audio_path(row: dict[str, str]) -> Path:
    raw = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw = raw.strip()
    if not raw:
        raise ValueError("Manifest row is missing an audio path")
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail
    return Path(raw).expanduser()


def audio_duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return round(float(info.frames) / float(info.samplerate), 3)


def place_audio(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        os.symlink(source, destination)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    splits = set(args.split or ["train", "eval"])

    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    context_lines = args.context_line or DEFAULT_CONTEXT
    rows_out: list[dict[str, str]] = []
    counts = {"seen": 0, "exported": 0, "skipped": 0}
    split_counts: dict[str, int] = {split: 0 for split in splits}

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            counts["seen"] += 1
            split = (row.get("split") or "").strip()
            speaker_id = (row.get("speaker_user_id") or "").strip()
            target_text = normalize_text(row.get(args.target_column) or "")
            if split not in splits or speaker_id != str(args.speaker_id) or not target_text:
                counts["skipped"] += 1
                continue

            source_audio = resolve_audio_path(row)
            if not source_audio.is_file():
                raise FileNotFoundError(source_audio)
            duration = audio_duration_seconds(source_audio)

            split_counts[split] += 1
            stem = f"ak_{split}_{split_counts[split]:04d}"
            audio_name = stem + source_audio.suffix.lower()
            json_name = stem + ".json"
            split_dir = output_dir / split
            audio_dest = split_dir / audio_name
            json_dest = split_dir / json_name

            place_audio(source_audio, audio_dest, args.audio_mode)
            label = {
                "audio_duration": duration,
                "audio_path": audio_name,
                "segments": [
                    {
                        "speaker": 0,
                        "text": target_text,
                        "start": 0.0,
                        "end": duration,
                    }
                ],
                "customized_context": context_lines,
            }
            json_dest.write_text(
                json.dumps(label, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            rows_out.append(
                {
                    "split": split,
                    "json": str(json_dest),
                    "audio": str(audio_dest),
                    "source_audio": str(source_audio),
                    "duration_seconds": str(duration),
                    "target_text": target_text,
                    "recording_type": row.get("recording_type") or "",
                    "word_title": row.get("word_title") or "",
                    "source_bundle": row.get("source_bundle") or "",
                }
            )
            counts["exported"] += 1

    index_path = output_dir / "vibevoice_dataset_index.tsv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "json",
            "audio",
            "source_audio",
            "duration_seconds",
            "target_text",
            "recording_type",
            "word_title",
            "source_bundle",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "speaker_user_id": str(args.speaker_id),
        "target_column": args.target_column,
        "audio_mode": args.audio_mode,
        "counts": counts,
        "split_counts": split_counts,
        "context": context_lines,
        "index": str(index_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
