#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create and transcribe sequential phase1a Whisper batches from an existing full split + MMS pass."
    )
    parser.add_argument(
        "--full-run-dir",
        default=str(script_dir / "transcription_outputs" / "phase1a_processed_multi_ipa_cpu_20260410-133537"),
    )
    parser.add_argument(
        "--audio",
        default=str(script_dir.parent / "Z class with C" / "Quizlet recordings" / "processed" / "phase1a_processed.mp3"),
    )
    parser.add_argument("--start-clip", type=int, default=301)
    parser.add_argument("--end-clip", type=int, default=1517)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--whisper-batch-size", type=int, default=8)
    parser.add_argument("--whisper-num-beams", type=int, default=1)
    parser.add_argument("--whisper-max-length", type=int, default=64)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def chunk_label(start_clip: int, end_clip: int) -> str:
    return f"phase1a_{start_clip:04d}_{end_clip:04d}_multi_ipa_cpu_{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    full_run_dir = Path(args.full_run_dir).expanduser().resolve()
    split_src = full_run_dir / "phase_split.tsv"
    mms_src = full_run_dir / "mms_v003_predictions.tsv"
    if not split_src.is_file() or not mms_src.is_file():
        raise SystemExit(f"Missing full-run artifacts in {full_run_dir}")

    all_split_rows = read_tsv(split_src)
    all_mms_rows = read_tsv(mms_src)
    mms_by_id = {int(row["clip_id"]): row for row in all_mms_rows}
    output_root = script_dir / "transcription_outputs"
    driver = script_dir / "transcribe_phase_audio_multi_ipa.py"
    python_exe = script_dir.parent / "audio_corpus" / "venv311" / "Scripts" / "python.exe"
    audio_path = Path(args.audio).expanduser().resolve()

    current = max(1, args.start_clip)
    final_clip = min(args.end_clip, len(all_split_rows))
    while current <= final_clip:
        batch_end = min(current + args.chunk_size - 1, final_clip)
        run_dir = output_root / chunk_label(current, batch_end)
        run_dir.mkdir(parents=True, exist_ok=True)

        subset_split = [row for row in all_split_rows if current <= int(row["clip_id"]) <= batch_end]
        subset_mms = [mms_by_id[int(row["clip_id"])] for row in subset_split if int(row["clip_id"]) in mms_by_id]
        write_tsv(run_dir / "phase_split.tsv", subset_split, list(subset_split[0].keys()))
        write_tsv(run_dir / "mms_v003_predictions.tsv", subset_mms, list(subset_mms[0].keys()))

        cmd = [
            str(python_exe),
            str(driver),
            "--audio",
            str(audio_path),
            "--split-tsv",
            str(run_dir / "phase_split.tsv"),
            "--mms-predictions-tsv",
            str(run_dir / "mms_v003_predictions.tsv"),
            "--whisper-batch-size",
            str(args.whisper_batch_size),
            "--whisper-num-beams",
            str(args.whisper_num_beams),
            "--whisper-max-length",
            str(args.whisper_max_length),
        ]
        print(f"Starting batch {current}-{batch_end}: {run_dir}", flush=True)
        completed = subprocess.run(cmd, cwd=str(script_dir.parent))
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

        current = batch_end + 1


if __name__ == "__main__":
    main()
