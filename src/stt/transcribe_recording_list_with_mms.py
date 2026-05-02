#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PureWindowsPath

from serve_local_stt_mms import default_model_source, load_model_bundle, transcribe_file


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Transcribe a TSV recording list with the local MMS IPA model."
    )
    parser.add_argument("--input-tsv", required=True, help="TSV with a recording_id column.")
    parser.add_argument(
        "--model-source",
        default=default_model_source(),
        help="Local MMS model directory. Defaults to the current best local MMS IPA model.",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=["auto", "greedy", "beam"],
        help="CTC decoder mode. Auto respects lltools_ctc_config.json when present.",
    )
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional output prefix. Defaults beside the input TSV.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Audio search root. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        help="Precision setting passed through to load_model_bundle.",
    )
    return parser.parse_args()


def default_search_roots(script_dir: Path) -> list[Path]:
    return [
        script_dir / "extracted_bundles",
        script_dir / "extracted_bundles_20260403",
        script_dir / "extracted_bundles_20260406",
        script_dir / "extracted_bundles_20260407_124628",
        script_dir / "extracted_bundles_20260408_134225",
        script_dir / "extracted_bundles_20260408_134225_blacklist_preview",
        script_dir / "extracted_bundles_full_audio_20260408",
        script_dir / "extracted_bundles_text",
        script_dir / "extracted_bundles_whisper_20260410",
    ]


def make_output_prefix(input_tsv: Path, output_prefix: str) -> Path:
    if output_prefix:
        prefix = Path(output_prefix)
        if prefix.is_absolute():
            return prefix
        return input_tsv.parent / prefix
    return input_tsv.with_suffix("")


def collect_audio_index(roots: list[Path]) -> dict[str, list[Path]]:
    rec_audio: dict[str, list[Path]] = defaultdict(list)
    pattern = re.compile(r"^(\d+)-")
    existing_roots = [root for root in roots if root.exists()]
    if not existing_roots:
        return rec_audio

    if os.name == "nt":
        for root in existing_roots:
            for audio_dir_name in ("audio", "media"):
                for audio_dir in root.rglob(audio_dir_name):
                    if not audio_dir.is_dir():
                        continue
                    for path in audio_dir.iterdir():
                        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
                            continue
                        match = pattern.match(path.name)
                        if match:
                            rec_audio[match.group(1)].append(path.resolve())
        return rec_audio

    cmd = [
        "find",
        *[str(root) for root in existing_roots],
        "-type",
        "f",
        "(",
        "-iname",
        "*.wav",
        "-o",
        "-iname",
        "*.mp3",
        "-o",
        "-iname",
        "*.m4a",
        "-o",
        "-iname",
        "*.aac",
        "-o",
        "-iname",
        "*.ogg",
        ")",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    for raw_path in completed.stdout.splitlines():
        path = Path(raw_path.strip())
        if not path:
            continue
        match = pattern.match(path.name)
        if match:
            rec_audio[match.group(1)].append(path.resolve())
    return rec_audio


def choose_best_audio_path(paths: list[Path]) -> Path:
    def score(path: Path) -> tuple[int, int, str]:
        suffix_rank = {".wav": 0, ".mp3": 1, ".m4a": 2, ".aac": 3, ".ogg": 4}.get(path.suffix.lower(), 9)
        text_bonus = 0 if "extracted_bundles_text" in str(path) else 1
        return (text_bonus, suffix_rank, str(path))

    return sorted(paths, key=score)[0]


def windows_to_posix_path(path_str: str) -> Path:
    windows_path = PureWindowsPath(path_str)
    drive = windows_path.drive.rstrip(":").lower()
    path = Path("/mnt") / drive
    for part in windows_path.parts[1:]:
        if part not in {"\\", "/"}:
            path /= part
    return path


def resolve_existing_path(raw_value: str) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    direct_path = Path(value).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    if re.match(r"^[A-Za-z]:[\\/]", value):
        posix_path = windows_to_posix_path(value)
        if posix_path.exists():
            return posix_path.resolve()
    return None


def resolve_audio_path(row: dict[str, str], rec_audio: dict[str, list[Path]]) -> Path | None:
    for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"):
        resolved = resolve_existing_path(str(row.get(key) or ""))
        if resolved is not None:
            return resolved

    rid = str(row.get("recording_id") or "").strip()
    if not rid:
        return None
    audio_paths = rec_audio.get(rid) or []
    if not audio_paths:
        return None
    return choose_best_audio_path(audio_paths)


def load_rows(input_tsv: Path, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row.get("recording_id"):
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    output_prefix = make_output_prefix(input_tsv, args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    roots = [Path(p).expanduser().resolve() for p in args.search_root] if args.search_root else default_search_roots(script_dir)
    rows = load_rows(input_tsv, args.limit)
    needs_search_index = any(
        not any(str(row.get(key) or "").strip() for key in ("audio_posix", "audio_windows", "audio", "local_audio_path"))
        for row in rows
    )
    rec_audio = collect_audio_index(roots) if needs_search_index else {}

    bundle = load_model_bundle(
        model_source=str(Path(args.model_source).expanduser().resolve()),
        device_name=args.device,
        precision=args.precision,
        decode_mode=args.decoder,
        beam_width=args.beam_width,
    )

    results: list[dict[str, str]] = []
    missing_audio: list[dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        rid = row["recording_id"].strip()
        audio_path = resolve_audio_path(row, rec_audio)
        if audio_path is None:
            missing_audio.append(row)
            continue
        predicted_ipa = transcribe_file(bundle, audio_path)
        result = dict(row)
        result["local_audio_path"] = str(audio_path)
        result["predicted_ipa"] = predicted_ipa
        results.append(result)
        print(f"[{index}/{len(rows)}] {rid} -> {predicted_ipa}", flush=True)

    fieldnames = list(rows[0].keys()) if rows else ["recording_id"]
    for extra in ("local_audio_path", "predicted_ipa"):
        if extra not in fieldnames:
            fieldnames.append(extra)
    tsv_path = output_prefix.with_name(output_prefix.name + "_mms_predictions.tsv")
    jsonl_path = output_prefix.with_name(output_prefix.name + "_mms_predictions.jsonl")
    missing_audio_path = output_prefix.with_name(output_prefix.name + "_missing_audio.tsv")
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.json")

    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    missing_fields = list(rows[0].keys()) if rows else ["recording_id"]
    with missing_audio_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=missing_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(missing_audio)

    summary = {
        "input_tsv": str(input_tsv),
        "model_source": str(Path(args.model_source).expanduser().resolve()),
        "device": args.device,
        "decoder": bundle["decoder_mode"],
        "beam_width": bundle["beam_width"],
        "search_roots": [str(root) for root in roots],
        "input_rows": len(rows),
        "transcribed_rows": len(results),
        "missing_audio_rows": len(missing_audio),
        "outputs": {
            "tsv": str(tsv_path),
            "jsonl": str(jsonl_path),
            "missing_audio_tsv": str(missing_audio_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
