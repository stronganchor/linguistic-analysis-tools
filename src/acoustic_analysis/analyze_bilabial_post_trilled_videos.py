#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram


ROOT = Path("$Z_WORKSPACE/Phonetic inventory/bilabially post-trilled dental stops")
OUT = ROOT / "audio-analysis"
AUDIO_DIR = OUT / "audio"
FRAMES_DIR = OUT / "frames"
PLOTS_DIR = OUT / "plots"
SUMMARY_PATH = OUT / "summary.json"


FFMPEG = "C:\\ffmpeg\\bin\\ffmpeg.exe"
POWERSHELL = "powershell.exe"


def win_path(path: Path) -> str:
    text = str(path)
    if text.startswith("/mnt/c/"):
        return "C:" + text[6:].replace("/", "\\")
    return text.replace("/", "\\")


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["powershell.exe", "-NoProfile", "-Command", args[0]]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def ensure_dirs() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_audio(video: Path) -> Path:
    wav_path = AUDIO_DIR / f"{video.stem}.wav"
    cmd = (
        f"& '{FFMPEG}' -y -i '{win_path(video)}' -vn -ac 1 -ar 16000 -sample_fmt s16 "
        f"'{win_path(wav_path)}'"
    )
    run_ffmpeg([cmd])
    return wav_path


def extract_frames(video: Path, fps: int = 6) -> Path:
    frame_dir = FRAMES_DIR / video.stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"& '{FFMPEG}' -y -i '{win_path(video)}' -vf fps={fps},scale=320:-1 "
        f"'{win_path(frame_dir / 'frame_%03d.png')}'"
    )
    run_ffmpeg([cmd])
    return frame_dir


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype.kind in {"i", "u"}:
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return rate, data


def make_wave_spec_plot(video: Path, wav_path: Path, duration_s: float) -> dict[str, float]:
    rate, data = read_wav(wav_path)
    t = np.arange(len(data)) / float(rate)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), constrained_layout=True)

    axes[0].plot(t, data, linewidth=0.6, color="#6b4e2e")
    axes[0].set_title(f"{video.name} waveform")
    axes[0].set_xlim(0, max(duration_s, t[-1] if len(t) else 0))
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    f, tt, sxx = spectrogram(data, fs=rate, nperseg=512, noverlap=384, scaling="density", mode="magnitude")
    sxx_db = 20 * np.log10(sxx + 1e-10)
    im = axes[1].pcolormesh(tt, f, sxx_db, shading="auto", cmap="magma")
    axes[1].set_ylim(0, 8000)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Spectrogram")
    fig.colorbar(im, ax=axes[1], pad=0.01, label="dB")

    plot_path = PLOTS_DIR / f"{video.stem}_wave_spec.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return {
        "sample_rate": rate,
        "samples": int(len(data)),
        "duration_seconds": float(len(data) / rate if rate else 0.0),
    }


def make_contact_sheet(frame_dir: Path, video_name: str, cols: int = 6, max_frames: int = 24) -> dict[str, object]:
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        return {"frames": 0, "contact_sheet": ""}
    if len(frame_paths) > max_frames:
        idxs = np.linspace(0, len(frame_paths) - 1, max_frames).round().astype(int)
        frame_paths = [frame_paths[i] for i in idxs]

    rows = int(math.ceil(len(frame_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(frame_paths):
            continue
        img = plt.imread(frame_paths[i])
        ax.imshow(img)
        ax.set_title(frame_paths[i].stem, fontsize=8)

    sheet_path = PLOTS_DIR / f"{Path(video_name).stem}_contact_sheet.png"
    fig.suptitle(video_name, fontsize=13)
    fig.savefig(sheet_path, dpi=160)
    plt.close(fig)
    return {"frames": len(frame_paths), "contact_sheet": str(sheet_path)}


def main() -> None:
    ensure_dirs()
    videos = sorted(ROOT.glob("*.mp4"))
    summary = {"videos": []}
    for video in videos:
        wav = extract_audio(video)
        frame_dir = extract_frames(video)
        audio_stats = make_wave_spec_plot(video, wav, 0.0)
        contact = make_contact_sheet(frame_dir, video.name)
        summary["videos"].append(
            {
                "video": video.name,
                "audio": str(wav),
                "frame_dir": str(frame_dir),
                "wave_spec": str(PLOTS_DIR / f"{video.stem}_wave_spec.png"),
                **audio_stats,
                **contact,
            }
        )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
