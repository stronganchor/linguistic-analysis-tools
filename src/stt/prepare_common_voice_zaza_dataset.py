#!/usr/bin/env python3
import argparse
import csv
import json
import tarfile
from pathlib import Path

import torchaudio

from prepare_lltools_ipa_dataset import normalize_plain_text, to_windows_path


MANIFEST_HEADERS = [
    "split",
    "audio",
    "audio_windows",
    "audio_posix",
    "target_text",
    "cv_split",
    "client_id",
    "variant",
    "sentence_id",
    "source_path",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a Common Voice orthography manifest for CTC warm-start training")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a Common Voice archive (.tar.gz) or an extracted dataset folder.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(script_dir / "external-data" / "common-voice" / "extracted"),
        help="Where tar archives should be extracted.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where manifest_text.tsv and summary.json will be written.",
    )
    parser.add_argument(
        "--lowercase-targets",
        action="store_true",
        help="Lowercase orthography targets in the manifest.",
    )
    return parser.parse_args()


def maybe_extract(input_path: Path, extract_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path

    if not input_path.name.endswith(".tar.gz"):
        raise RuntimeError(f"Unsupported input: {input_path}")

    destination = extract_dir / input_path.name.removesuffix(".tar.gz")
    if destination.exists():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(input_path, "r:gz") as archive:
        archive.extractall(destination)
    return destination


def find_dataset_root(root: Path) -> Path:
    candidates: list[Path] = []
    for train_tsv in root.rglob("train.tsv"):
        parent = train_tsv.parent
        if (parent / "dev.tsv").is_file() and (parent / "test.tsv").is_file():
            candidates.append(parent)
    if not candidates:
        raise RuntimeError(f"Could not find train/dev/test TSV files under {root}")
    candidates.sort(key=lambda p: (0 if "zza" in p.as_posix().lower() else 1, len(p.parts)))
    return candidates[0]


def resolve_clip_dir(dataset_root: Path) -> Path:
    for candidate in (dataset_root / "clips", dataset_root / "audio"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"Could not find clip directory under {dataset_root}")


def duration_seconds(path: Path) -> float:
    info = torchaudio.info(str(path))
    if info.sample_rate <= 0:
        return 0.0
    return float(info.num_frames) / float(info.sample_rate)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = find_dataset_root(maybe_extract(input_path, extract_dir))
    clips_dir = resolve_clip_dir(dataset_root)

    split_map = {
        "train.tsv": "train",
        "dev.tsv": "eval",
        "test.tsv": "eval",
    }

    rows: list[dict[str, str]] = []
    summary = {
        "input": str(input_path),
        "dataset_root": str(dataset_root),
        "clips_dir": str(clips_dir),
        "lowercase_targets": bool(args.lowercase_targets),
        "counts": {},
        "duration_seconds": {},
    }

    for filename, split in split_map.items():
        tsv_path = dataset_root / filename
        split_rows = 0
        split_seconds = 0.0
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                path_value = str(row.get("path") or "").strip()
                text_value = normalize_plain_text(str(row.get("sentence") or row.get("text") or ""))
                if args.lowercase_targets:
                    text_value = text_value.lower()
                if not path_value or not text_value:
                    continue
                audio_path = (clips_dir / path_value).resolve()
                if not audio_path.is_file():
                    continue

                split_rows += 1
                split_seconds += duration_seconds(audio_path)
                rows.append(
                    {
                        "split": split,
                        "audio": str(audio_path),
                        "audio_windows": to_windows_path(audio_path),
                        "audio_posix": audio_path.as_posix(),
                        "target_text": text_value,
                        "cv_split": filename.removesuffix(".tsv"),
                        "client_id": str(row.get("client_id") or ""),
                        "variant": str(row.get("variant") or ""),
                        "sentence_id": str(row.get("sentence_id") or ""),
                        "source_path": path_value,
                    }
                )

        summary["counts"][filename.removesuffix(".tsv")] = split_rows
        summary["duration_seconds"][filename.removesuffix(".tsv")] = split_seconds

    manifest_path = output_dir / "manifest_text.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary["manifest"] = str(manifest_path)
    summary["total_examples"] = len(rows)
    summary["total_duration_seconds"] = sum(summary["duration_seconds"].values())
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
