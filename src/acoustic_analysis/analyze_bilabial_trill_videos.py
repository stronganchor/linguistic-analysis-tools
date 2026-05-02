#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT
OUT_DIR = ROOT / "video-analysis"
AUDIO_DIR = OUT_DIR / "audio"
FRAMES_DIR = OUT_DIR / "frames"
SHEETS_DIR = OUT_DIR / "sheets"
NOTES_DIR = OUT_DIR / "notes"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE = r"C:\ffmpeg\bin\ffprobe.exe"


def windows_path(path: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_ffmpeg(args: list[str]) -> None:
    command = " ".join(ps_quote(arg) for arg in [FFMPEG, *args])
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"& {command}"], check=True)


def probe(video: Path) -> dict:
    command = " ".join(
        ps_quote(arg)
        for arg in [
            FFPROBE,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            windows_path(video),
        ]
    )
    out = subprocess.check_output(
        ["powershell.exe", "-NoProfile", "-Command", f"& {command}"], text=True
    )
    return json.loads(out)


def extract_audio(video: Path, out_wav: Path) -> None:
    run_ffmpeg(
        [
            "-y",
            "-nostdin",
            "-i",
            windows_path(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            windows_path(out_wav),
        ]
    )


def extract_contact_sheet(video: Path, out_png: Path) -> None:
    run_ffmpeg(
        [
            "-y",
            "-nostdin",
            "-i",
            windows_path(video),
            "-vf",
            "fps=10,scale=480:-1,tile=5x6",
            "-frames:v",
            "1",
            windows_path(out_png),
        ]
    )


def extract_spectrogram(video: Path, out_png: Path) -> None:
    run_ffmpeg(
        [
            "-y",
            "-nostdin",
            "-i",
            windows_path(video),
            "-lavfi",
            "showspectrumpic=s=1200x360:legend=0:color=intensity",
            "-frames:v",
            "1",
            windows_path(out_png),
        ]
    )


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for video in sorted(VIDEO_DIR.glob("*.mp4")):
        info = probe(video)
        duration = float(info["format"]["duration"])
        stem = video.stem
        audio_out = AUDIO_DIR / f"{stem}.wav"
        sheet_out = SHEETS_DIR / f"{stem}.contact-sheet.png"
        spec_out = SHEETS_DIR / f"{stem}.spectrogram.png"
        extract_audio(video, audio_out)
        extract_contact_sheet(video, sheet_out)
        extract_spectrogram(video, spec_out)
        manifest.append(
            {
                "video": video.name,
                "duration_s": duration,
                "audio_wav": str(audio_out),
                "contact_sheet": str(sheet_out),
                "spectrogram": str(spec_out),
            }
        )

    (OUT_DIR / "video-analysis-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    notes = [
        "Generated audio WAVs, contact sheets, and spectrograms for the four A.K. closeup videos.",
        "Use the contact sheets to inspect repeated bilabial closures and the spectrograms to align them with stop/trill bursts.",
    ]
    (NOTES_DIR / "analysis-plan.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
