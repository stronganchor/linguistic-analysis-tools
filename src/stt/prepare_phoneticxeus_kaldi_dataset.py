#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import unicodedata
from pathlib import Path

import yaml


STRESS_MARK_PATTERN = re.compile(r"[ˈˌ']")


def normalize_ipa_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = STRESS_MARK_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_ipa_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_ipa_text(value))


def resolve_audio_path(raw_value: str) -> str:
    raw_value = (raw_value or "").strip()
    if not raw_value:
        raise RuntimeError("Missing audio path in manifest row")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return str(Path("/mnt") / drive / tail)

    return str(Path(raw_value).expanduser())


def make_key(row: dict, index: int) -> str:
    base = (
        str(row.get("word_audio_id") or "")
        or str(row.get("word_id") or "")
        or f"sample_{index:06d}"
    )
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_") or f"sample_{index:06d}"
    return f"{base}_pr"


def write_split(split_name: str, rows: list[dict], split_dir: Path, links_root: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    links_root.mkdir(parents=True, exist_ok=True)
    wav_scp_path = split_dir / "wav.scp"
    text_ctc_path = split_dir / "text.ctc"
    lang_path = split_dir / "text"

    with (
        wav_scp_path.open("w", encoding="utf-8") as wav_handle,
        text_ctc_path.open("w", encoding="utf-8") as text_handle,
        lang_path.open("w", encoding="utf-8") as lang_handle,
    ):
        for index, row in enumerate(rows, start=1):
            key = make_key(row, index)
            audio_path = resolve_audio_path(
                str(row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or "")
            )
            target_ipa = compact_ipa_text(str(row.get("target_ipa") or ""))
            if not target_ipa:
                continue

            source_path = Path(audio_path)
            suffix = source_path.suffix or ".wav"
            linked_audio = links_root / f"{key}{suffix}"
            if linked_audio.exists() or linked_audio.is_symlink():
                linked_audio.unlink()
            os.symlink(source_path, linked_audio)

            wav_handle.write(f"{key} {linked_audio}\n")
            text_handle.write(f"{key} {target_ipa}\n")
            lang_handle.write(f"{key} <unk><unk>\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a Kaldi-style PhoneticXeus dataset from manifest_ipa.tsv"
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest_ipa.tsv")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory for generated train/eval Kaldi files",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    total_rows = 0

    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            total_rows += 1
            split = str(row.get("split") or "").strip()
            target_ipa = compact_ipa_text(str(row.get("target_ipa") or ""))
            if not target_ipa:
                continue
            if split == "train":
                train_rows.append(row)
            elif split == "eval":
                eval_rows.append(row)

    if not train_rows or not eval_rows:
        raise RuntimeError(
            f"Expected both train and eval rows in {manifest_path}; got train={len(train_rows)} eval={len(eval_rows)}"
        )

    train_dir = output_root / "train"
    eval_dir = output_root / "eval"
    links_dir = output_root / "audio_links"
    write_split("train", train_rows, train_dir, links_dir / "train")
    write_split("eval", eval_rows, eval_dir, links_dir / "eval")

    dataset_config = {
        "sampling_rate": 16000,
        "datasets": {
            "train": {
                "wav_scp": str(train_dir / "wav.scp"),
                "text_phoneme": str(train_dir / "text.ctc"),
                "language": str(train_dir / "text"),
                "task_set": ["pr"],
            },
            "eval": {
                "wav_scp": str(eval_dir / "wav.scp"),
                "text_phoneme": str(eval_dir / "text.ctc"),
                "language": str(eval_dir / "text"),
                "task_set": ["pr"],
            },
            "predict": {
                "wav_scp": str(eval_dir / "wav.scp"),
                "text_phoneme": str(eval_dir / "text.ctc"),
                "language": str(eval_dir / "text"),
                "task_set": ["pr"],
            },
        },
    }

    (output_root / "dataset_config.yaml").write_text(
        yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_root / "dataset_summary.json").write_text(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "total_rows": total_rows,
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
