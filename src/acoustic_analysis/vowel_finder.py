
# Auto-runs Praat to extract formant tracks from the loaded audio file.
# - Opens audio, converts non-WAV via ffmpeg when needed, then calls Praat.exe --run.
# - Plots waveform, spectrogram, and vowel-space guesses.
# - By default, only shows formant tracks/labels for frames that look vowel-like.
# Requires: python -m pip install sounddevice numpy scipy matplotlib
# Optional for non-WAV input: ffmpeg on PATH.

import os
import sys
import json
import time
import math
import tempfile
import subprocess
import threading
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile
from scipy.signal import stft

CONFIG_FILENAME = "vowel_finder_settings.json"


def get_script_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def default_boundary_timing_priors_path():
    script_dir = get_script_dir()
    path = os.path.join(script_dir, "vowel-boundary-timing-priors.json")
    return path if os.path.isfile(path) else ""


def _unique_existing_paths(paths):
    out = []
    seen = set()
    for path in paths:
        if not path:
            continue
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(path):
            out.append(path)
    return out


def find_praat_executable(configured_path: str = ""):
    configured_path = str(configured_path or "").strip()
    if configured_path and os.path.isfile(configured_path):
        return configured_path

    candidates = []

    which_path = shutil.which("Praat")
    if which_path:
        candidates.append(which_path)

    which_path_exe = shutil.which("Praat.exe")
    if which_path_exe:
        candidates.append(which_path_exe)

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

        candidates.extend([
            os.path.join(program_files, "Praat", "Praat.exe"),
            os.path.join(program_files_x86, "Praat", "Praat.exe"),
            os.path.join(local_app_data, "Programs", "Praat", "Praat.exe"),
        ])
    elif sys.platform == "darwin":
        candidates.extend([
            "/Applications/Praat.app/Contents/MacOS/Praat",
            os.path.expanduser("~/Applications/Praat.app/Contents/MacOS/Praat"),
        ])
    else:
        candidates.extend([
            "/usr/bin/praat",
            "/usr/local/bin/praat",
            "/snap/bin/praat",
        ])

    existing = _unique_existing_paths(candidates)
    return existing[0] if existing else ""


def describe_praat_path(configured_path: str = ""):
    configured_path = str(configured_path or "").strip()
    resolved = find_praat_executable(configured_path)
    if resolved:
        return resolved
    if configured_path:
        return configured_path
    return "(not found)"


DEFAULT_SETTINGS = {
    "praat_exe_path": "",
    "speaker_preset": "adult female",
    "praat_time_step": 0.0,
    "praat_max_formants": 5.0,
    "praat_ceiling_hz": 5500.0,
    "praat_window_length": 0.025,
    "praat_pre_emphasis_hz": 50.0,
    "voicing": 0.30,
    "reject_rhotics": 1,
    "label_density": 18,
    "spec_maxhz": 5000,
    "show_only_vowels": 1,
    "boundary_timing_priors_json": default_boundary_timing_priors_path(),
    "rhotic_f2f3_gap_max": 350.0,
    "min_absolute_db": -42.0,
    "min_relative_db": -24.0,
    "min_f1_hz": 180.0,
    "max_f1_hz": 1100.0,
    "min_f2_hz": 500.0,
    "max_f2_hz": 3200.0,
    "min_f2_minus_f1_hz": 120.0,
    "segment_seed_score": 0.62,
    "segment_extend_score": 0.48,
    "segment_boundary_score": 0.36,
    "segment_bridge_gap_s": 0.02,
    "segment_merge_gap_s": 0.015,
    "core_support_threshold": 0.36,
    "core_support_strong_threshold": 0.40,
    "core_support_edge_threshold": 0.32,
    "min_segment_s": 0.03,
    "max_core_df1_hz_per_s": 900.0,
    "max_core_df2_hz_per_s": 1800.0,
    "min_core_s": 0.035,
}

SPEAKER_PRESETS = {
    "adult male": {
        "praat_ceiling_hz": 5000.0,
        "praat_max_formants": 5.0,
        "praat_window_length": 0.025,
        "praat_pre_emphasis_hz": 50.0,
        "praat_time_step": 0.0,
    },
    "adult female": {
        "praat_ceiling_hz": 5500.0,
        "praat_max_formants": 5.0,
        "praat_window_length": 0.025,
        "praat_pre_emphasis_hz": 50.0,
        "praat_time_step": 0.0,
    },
    "child": {
        "praat_ceiling_hz": 6500.0,
        "praat_max_formants": 5.0,
        "praat_window_length": 0.025,
        "praat_pre_emphasis_hz": 50.0,
        "praat_time_step": 0.0,
    },
}

def get_config_path():
    return os.path.join(get_script_dir(), CONFIG_FILENAME)

def _coerce_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)

def _coerce_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def clamp01(value):
    try:
        value = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def smooth_track(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.pad(np.asarray(values, dtype=np.float64), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_boundary_timing_priors(path: str):
    path = str(path or "").strip()
    if not path:
        return {}, ""

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}, path

    priors = {}
    if not isinstance(raw, dict):
        return priors, path

    for sym, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        count = _coerce_int(payload.get("count"), 0)
        left_s = _coerce_float(payload.get("peak_minus_start_s"), float("nan"))
        right_s = _coerce_float(payload.get("end_minus_peak_s"), float("nan"))
        duration_s = _coerce_float(payload.get("duration_s"), float("nan"))
        if count <= 0:
            continue
        if not all(math.isfinite(v) for v in (left_s, right_s, duration_s)):
            continue
        priors[str(sym)] = {
            "count": count,
            "peak_minus_start_s": max(0.0, left_s),
            "end_minus_peak_s": right_s,
            "duration_s": max(0.03, duration_s),
        }
    return priors, path

# ---------- IPA references ----------
VOWEL_RANGES = {
    "i":  {"F1": (180, 360),  "F2": (2000, 3100)},
    "ɪ":  {"F1": (250, 500),  "F2": (1700, 2600)},
    "e":  {"F1": (300, 650),  "F2": (1700, 2500)},
    "ɛ":  {"F1": (450, 850),  "F2": (1500, 2400)},
    "æ":  {"F1": (650, 1050), "F2": (1400, 2300)},
    "y":  {"F1": (180, 380),  "F2": (1500, 2600)},
    "ʉ":  {"F1": (200, 450),  "F2": (1200, 2000)},
    "ɨ":  {"F1": (250, 450),  "F2": (1300, 2000)},
    "ə":  {"F1": (350, 650),  "F2": (1200, 1800)},
    "ɜ":  {"F1": (450, 800),  "F2": (1100, 1600)},
    "u":  {"F1": (200, 420),  "F2": (600, 1200)},
    "ʊ":  {"F1": (250, 500),  "F2": (700, 1400)},
    "o":  {"F1": (350, 650),  "F2": (600, 1200)},
    "ɔ":  {"F1": (450, 850),  "F2": (600, 1300)},
    "ɑ":  {"F1": (650, 1050), "F2": (800, 1500)},
    "a":  {"F1": (700, 1100), "F2": (1100, 1700)},
    "ɐ":  {"F1": (550, 900),  "F2": (1100, 1700)},
}

VOWEL_REF_POINTS = {
    "i":  (300, 2400),
    "ɪ":  (400, 2000),
    "e":  (450, 2200),
    "ɛ":  (650, 1800),
    "æ":  (850, 1700),
    "y":  (320, 1900),
    "ʉ":  (350, 1500),
    "ɨ":  (350, 1600),
    "ə":  (500, 1500),
    "ɜ":  (650, 1300),
    "u":  (350, 900),
    "ʊ":  (450, 1100),
    "o":  (500, 1000),
    "ɔ":  (650, 900),
    "ɑ":  (800, 1200),
    "a":  (950, 1500),
    "ɐ":  (700, 1500),
}

_ORDER = ["i","ɪ","e","ɛ","æ","y","ʉ","ɨ","ə","ɜ","u","ʊ","o","ɔ","ɑ","a","ɐ"]
LABEL_BBOX = dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.5)

# ---------- Audio helpers ----------
def ffmpeg_to_wav(in_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="praat_like_")
    out_wav = os.path.join(tmp_dir, "audio.wav")
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", out_wav]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return out_wav

def load_audio_any(path: str):
    note = ""
    used_path = path
    if not path.lower().endswith(".wav"):
        used_path = ffmpeg_to_wav(path)
        note = "(Converted to temp mono WAV via ffmpeg for Python/Praat analysis. Native sample rate preserved when possible.)"

    sr, data = wavfile.read(used_path)
    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        x = data.astype(np.float32) / 2147483648.0
    else:
        x = data.astype(np.float32)
        mx = np.max(np.abs(x)) + 1e-9
        if mx > 1.5:
            x = x / mx

    return sr, x, path, used_path, note


def spectrogram_time_edges(tt: np.ndarray, sr: int, nfft: int, hop: int, signal_len: int):
    if len(tt) == 0:
        return np.array([0.0, signal_len / float(sr)], dtype=np.float64)

    half_window = (nfft / float(sr)) * 0.5
    edges = np.empty(len(tt) + 1, dtype=np.float64)
    if len(tt) == 1:
        edges[0] = max(0.0, float(tt[0]) - half_window)
        edges[1] = min(signal_len / float(sr), float(tt[0]) + half_window)
        return edges

    mids = 0.5 * (tt[:-1] + tt[1:])
    edges[1:-1] = mids
    edges[0] = max(0.0, float(tt[0]) - 0.5 * (tt[1] - tt[0]))
    edges[-1] = min(signal_len / float(sr), float(tt[-1]) + 0.5 * (tt[-1] - tt[-2]))
    return edges

# ---------- Praat automation ----------
def praat_script_text():
    return '''
form ExportFormantsToTSV
    sentence inputSound
    sentence outputTsv
    real timeStep 0.005
    real maxFormants 5
    real ceiling 5000
    real windowLength 0.025
    real preEmphasis 50
endform

filedelete 'outputTsv$'
appendFileLine: outputTsv$, "time_s", tab$, "F1", tab$, "F2", tab$, "F3", tab$, "F4"

sound = Read from file: inputSound$
selectObject: sound
formant = To Formant (burg): timeStep, maxFormants, ceiling, windowLength, preEmphasis
selectObject: formant
n = Get number of frames

for i to n
    t = Get time from frame number: i
    f1 = Get value at time: 1, t, "Hertz", "Linear"
    f2 = Get value at time: 2, t, "Hertz", "Linear"
    f3 = Get value at time: 3, t, "Hertz", "Linear"
    f4 = Get value at time: 4, t, "Hertz", "Linear"

    if f1 = undefined
        f1$ = "--undefined--"
    else
        f1$ = fixed$ (f1, 3)
    endif
    if f2 = undefined
        f2$ = "--undefined--"
    else
        f2$ = fixed$ (f2, 3)
    endif
    if f3 = undefined
        f3$ = "--undefined--"
    else
        f3$ = fixed$ (f3, 3)
    endif
    if f4 = undefined
        f4$ = "--undefined--"
    else
        f4$ = fixed$ (f4, 3)
    endif

    appendFileLine: outputTsv$, fixed$ (t, 6), tab$, f1$, tab$, f2$, tab$, f3$, tab$, f4$
endfor

selectObject: sound
plusObject: formant
Remove
'''

def run_praat_formants(audio_path: str,
                       praat_exe: str = "",
                       time_step: float = 0.005,
                       max_formants: float = 5.0,
                       ceiling_hz: float = 5000.0,
                       window_length: float = 0.025,
                       pre_emphasis_hz: float = 50.0):
    resolved_praat = find_praat_executable(praat_exe)

    if not resolved_praat:
        configured_text = str(praat_exe or "").strip() or "(none)"
        raise FileNotFoundError(
            "Praat executable was not found.\n\n"
            f"Configured Praat path: {configured_text}\n\n"
            "Set a valid Praat path in the app settings, or install Praat in a standard location, "
            "or add it to PATH."
        )

    tmp_dir = tempfile.mkdtemp(prefix="praat_formants_")
    script_path = os.path.join(tmp_dir, "export_formants.praat")
    out_tsv = os.path.join(tmp_dir, "formants.tsv")

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(praat_script_text())

    cmd = [
        resolved_praat,
        "--run",
        script_path,
        audio_path,
        out_tsv,
        f"{time_step:.6f}",
        f"{float(max_formants):.3f}",
        f"{float(ceiling_hz):.3f}",
        f"{window_length:.6f}",
        f"{pre_emphasis_hz:.3f}",
    ]

    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "check": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000

    try:
        subprocess.run(cmd, **kwargs)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(
            "Praat formant extraction failed.\n\n"
            f"Resolved Praat path:\n{resolved_praat}\n\n"
            f"Command:\n{' '.join(cmd)}\n\n"
            f"Praat output:\n{msg}"
        ) from e

    if not os.path.exists(out_tsv):
        raise RuntimeError("Praat ran, but no TSV output file was created.")

    with open(out_tsv, "r", encoding="utf-8") as f:
        return f.read()

# ---------- Praat formant listing parsing ----------
def parse_praat_listing(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.lower().startswith("time") or ln.lower().startswith("time_s"):
            continue
        ln = ln.replace("\t", " ")
        parts = [p for p in ln.split(" ") if p]
        if len(parts) < 3:
            continue

        def to_float(p):
            if "--undefined--" in p or "undefined" in p:
                return float("nan")
            try:
                return float(p)
            except Exception:
                return None

        t = to_float(parts[0])
        f1 = to_float(parts[1]) if len(parts) > 1 else None
        f2 = to_float(parts[2]) if len(parts) > 2 else None
        f3 = to_float(parts[3]) if len(parts) > 3 else float("nan")
        f4 = to_float(parts[4]) if len(parts) > 4 else float("nan")
        if t is None or f1 is None or f2 is None:
            continue
        rows.append((t, f1, f2, f3, f4))

    if not rows:
        return None

    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

# ---------- Voicing + vowel-likeness ----------
def rms_db(frame: np.ndarray):
    e = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))) + 1e-12)
    return 20.0 * math.log10(e + 1e-12)

def autocorr_voicing_strength(frame: np.ndarray, sr: int, f0_min=70, f0_max=400):
    frame = frame.astype(np.float64)
    frame = frame - np.mean(frame)
    frame = frame * np.hanning(len(frame))
    denom = np.sum(frame**2) + 1e-12
    if denom <= 1e-12:
        return 0.0

    ac = np.correlate(frame, frame, mode="full")
    ac = ac[len(ac)//2:]
    ac = ac / (ac[0] + 1e-12)

    lag_min = int(sr / f0_max)
    lag_max = int(sr / f0_min)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_max <= lag_min:
        return 0.0

    seg = ac[lag_min:lag_max+1]
    peak = float(np.max(seg))
    return peak


def extract_frame(x: np.ndarray, sr: int, center_s: float, window_s: float) -> np.ndarray:
    frame_size = max(256, int(round(sr * window_s)))
    center = int(round(center_s * sr))
    start = max(0, center - frame_size // 2)
    end = min(len(x), center + frame_size // 2)
    return x[start:end]


def tapered_range_score(value: float, lower: float, upper: float, taper: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return clamp01(1.0 - ((lower - value) / max(taper, 1e-6)))
    return clamp01(1.0 - ((value - upper) / max(taper, 1e-6)))


def build_scored_vowel_frame_rows(t, f1, f2, f3, sr, x, settings):
    times = np.asarray(t, dtype=np.float64)
    f1_values = np.asarray(f1, dtype=np.float64)
    f2_values = np.asarray(f2, dtype=np.float64)
    f3_values = np.asarray(f3, dtype=np.float64)
    if len(times) == 0:
        return []

    window_s = float(settings["praat_window_length"])
    rhotic_reject = bool(int(settings["reject_rhotics"]))
    voicing_floor = float(settings["voicing"])
    min_abs_db = float(settings["min_absolute_db"])
    min_rel_db = float(settings["min_relative_db"])
    min_f1 = float(settings["min_f1_hz"])
    max_f1 = float(settings["max_f1_hz"])
    min_f2 = float(settings["min_f2_hz"])
    max_f2 = float(settings["max_f2_hz"])
    min_sep = float(settings["min_f2_minus_f1_hz"])
    rhotic_gap = float(settings["rhotic_f2f3_gap_max"])

    frame_rows = []
    for index in range(len(times)):
        frame = extract_frame(x, sr, float(times[index]), window_s)
        if len(frame) < 128:
            db_value = -120.0
            voicing_value = 0.0
        else:
            db_value = rms_db(frame)
            voicing_value = autocorr_voicing_strength(frame, sr)

        f1_value = float(f1_values[index])
        f2_value = float(f2_values[index])
        f3_value = float(f3_values[index])

        f1_score = tapered_range_score(f1_value, min_f1, max_f1, 140.0)
        f2_score = tapered_range_score(f2_value, min_f2, max_f2, 350.0)
        separation_score = clamp01((f2_value - f1_value - min_sep) / 240.0) if math.isfinite(f1_value) and math.isfinite(f2_value) else 0.0
        voicing_score = clamp01((voicing_value - max(0.05, voicing_floor - 0.12)) / 0.28)
        abs_db_score = clamp01((db_value - min_abs_db) / 12.0)
        rhotic_penalty = 1.0
        if rhotic_reject and math.isfinite(f2_value) and math.isfinite(f3_value):
            if abs(f3_value - f2_value) <= rhotic_gap:
                rhotic_penalty = 0.0

        base_score = (
            (0.20 * f1_score)
            + (0.20 * f2_score)
            + (0.15 * separation_score)
            + (0.25 * voicing_score)
            + (0.20 * abs_db_score)
        ) * rhotic_penalty

        frame_rows.append(
            {
                "time_s": float(times[index]),
                "F1": f1_value,
                "F2": f2_value,
                "F3": f3_value,
                "db": db_value,
                "voicing": voicing_value,
                "base_score": base_score,
            }
        )

    peak_db = max(frame["db"] for frame in frame_rows) if frame_rows else -120.0
    for index, frame in enumerate(frame_rows):
        relative_db_score = clamp01((frame["db"] - (peak_db + min_rel_db)) / 10.0)
        neighbor_scores = []
        if index > 0 and frame_rows[index - 1]["time_s"] >= frame["time_s"] - 0.03:
            neighbor_scores.append(frame_rows[index - 1]["base_score"])
        if index + 1 < len(frame_rows) and frame_rows[index + 1]["time_s"] <= frame["time_s"] + 0.03:
            neighbor_scores.append(frame_rows[index + 1]["base_score"])
        continuity_score = float(np.mean(neighbor_scores)) if neighbor_scores else frame["base_score"]
        final_score = (
            (0.74 * frame["base_score"])
            + (0.12 * relative_db_score)
            + (0.14 * continuity_score)
        )
        frame["relative_db_score"] = relative_db_score
        frame["continuity_score"] = continuity_score
        frame["score"] = clamp01(final_score)

    return frame_rows


def compute_formant_stability_track(frame_rows):
    if not frame_rows:
        return np.zeros(0, dtype=np.float64)
    times = np.asarray([float(frame["time_s"]) for frame in frame_rows], dtype=np.float64)
    f1_values = np.asarray([float(frame.get("F1") or math.nan) for frame in frame_rows], dtype=np.float64)
    f2_values = np.asarray([float(frame.get("F2") or math.nan) for frame in frame_rows], dtype=np.float64)
    if len(times) < 3:
        return np.ones(len(frame_rows), dtype=np.float64)
    dt = np.gradient(times)
    dt[dt == 0.0] = np.nan
    df1 = np.gradient(f1_values) / dt
    df2 = np.gradient(f2_values) / dt
    f1_stability = np.asarray(
        [clamp01(1.0 - (abs(float(value)) / 650.0)) if math.isfinite(float(value)) else 0.0 for value in df1],
        dtype=np.float64,
    )
    f2_stability = np.asarray(
        [clamp01(1.0 - (abs(float(value)) / 1400.0)) if math.isfinite(float(value)) else 0.0 for value in df2],
        dtype=np.float64,
    )
    return smooth_track((0.44 * f1_stability) + (0.56 * f2_stability), window=5)


def compute_spectral_stripe_track(x: np.ndarray, sr: int, frame_rows, window_s: float = 0.025, max_freq_hz: float = 5000.0):
    if len(x) == 0 or sr <= 0 or not frame_rows:
        return np.zeros(0, dtype=np.float64)

    frame_length = max(256, int(2 ** round(math.log2(max(256, sr * window_s)))))
    half = frame_length // 2
    window = np.hanning(frame_length)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / float(sr))
    keep = freqs <= max_freq_hz
    spectra = []

    for frame in frame_rows:
        center = int(round(float(frame["time_s"]) * sr))
        left = center - half
        right = left + frame_length
        src_left = max(0, left)
        src_right = min(len(x), right)
        padded = np.zeros(frame_length, dtype=np.float64)
        dest_left = src_left - left
        dest_right = dest_left + max(0, src_right - src_left)
        if src_right > src_left:
            padded[dest_left:dest_right] = x[src_left:src_right].astype(np.float64)
        spectrum = np.abs(np.fft.rfft(padded * window))
        spectrum = spectrum[keep] + 1e-8
        spectra.append(spectrum)

    matrix = np.asarray(spectra, dtype=np.float64)
    power = np.square(matrix)
    power_norm = power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-9)
    flatness_raw = np.exp(np.mean(np.log(power), axis=1)) / np.maximum(np.mean(power, axis=1), 1e-9)
    flatness = np.asarray([clamp01((0.95 - float(value)) / 0.75) for value in flatness_raw], dtype=np.float64)
    crest_raw = np.percentile(power, 95.0, axis=1) / np.maximum(np.mean(power, axis=1), 1e-9)
    peakiness = np.asarray([clamp01((float(value) - 2.0) / 10.0) for value in crest_raw], dtype=np.float64)

    similarity = np.zeros(len(frame_rows), dtype=np.float64)
    for index in range(len(frame_rows)):
        sims = []
        current = power_norm[index]
        if index > 0:
            sims.append(float(np.dot(current, power_norm[index - 1]) / (np.linalg.norm(current) * np.linalg.norm(power_norm[index - 1]) + 1e-9)))
        if index + 1 < len(frame_rows):
            sims.append(float(np.dot(current, power_norm[index + 1]) / (np.linalg.norm(current) * np.linalg.norm(power_norm[index + 1]) + 1e-9)))
        similarity[index] = float(np.mean(sims)) if sims else 0.0
    similarity = np.asarray([clamp01((float(value) - 0.55) / 0.40) for value in similarity], dtype=np.float64)

    return smooth_track((0.42 * flatness) + (0.28 * similarity) + (0.30 * peakiness), window=5)

def is_vowel_like_point(t, f1, f2, f3, sr, x,
                        strength_thresh=0.30,
                        f1_bounds=(150, 1200),
                        f2_bounds=(500, 3200),
                        rhotic_reject=True,
                        rhotic_f2f3_gap_max=350,
                        min_db=-42.0):
    if np.isnan(f1) or np.isnan(f2):
        return False, "undefined formants"
    if not (f1_bounds[0] <= f1 <= f1_bounds[1]):
        return False, "F1 bounds"
    if not (f2_bounds[0] <= f2 <= f2_bounds[1]):
        return False, "F2 bounds"

    n = int(sr * 0.025)
    n = max(n, 256)
    center = int(t * sr)
    a = max(0, center - n // 2)
    b = min(len(x), center + n // 2)
    if b - a < 128:
        return False, "short frame"

    frame = x[a:b]
    strength = autocorr_voicing_strength(frame, sr)
    if strength < strength_thresh:
        return False, "weak voicing"

    if rms_db(frame) < min_db:
        return False, "low energy"

    if rhotic_reject and (not np.isnan(f3)):
        if abs(f3 - f2) <= rhotic_f2f3_gap_max:
            return False, "rhotic-like"

    return True, "ok"

def smooth_vowel_mask(mask: np.ndarray, t: np.ndarray, max_gap_s=0.020, min_run_s=0.030):
    mask = np.array(mask, dtype=bool)
    if len(mask) == 0:
        return mask

    out = mask.copy()

    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        left = i - 1
        right = j
        if left >= 0 and right < n and out[left] and out[right]:
            gap_dur = float(t[right] - t[left])
            if gap_dur <= max_gap_s:
                out[i:j] = True
        i = j

    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        dur = float(t[j-1] - t[i]) if j - i > 1 else 0.0
        if dur < min_run_s:
            out[i:j] = False
        i = j

    return out

def mask_to_segments(mask: np.ndarray):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []

    segments = []
    start = int(idx[0])
    prev = int(idx[0])
    for k in idx[1:]:
        k = int(k)
        if k == prev + 1:
            prev = k
            continue
        segments.append((start, prev))
        start = k
        prev = k
    segments.append((start, prev))
    return segments


def build_score_masks(t, f1, f2, frame_rows, settings, core_support_track=None):
    times = np.asarray(t, dtype=np.float64)
    scores = np.asarray([float(frame.get("score") or 0.0) for frame in frame_rows], dtype=np.float64)
    seed_mask = scores >= float(settings["segment_seed_score"])
    extend_mask = scores >= float(settings["segment_extend_score"])
    boundary_mask = scores >= float(settings["segment_boundary_score"])

    bridge_gap_s = float(settings["segment_bridge_gap_s"])
    min_segment_s = float(settings["min_segment_s"])
    if core_support_track is None or len(core_support_track) != len(times):
        core_support_track = np.ones(len(times), dtype=np.float64)
    else:
        core_support_track = np.asarray(core_support_track, dtype=np.float64)

    support_mask = core_support_track >= float(settings.get("core_support_threshold", 0.36))
    strong_support_mask = core_support_track >= float(settings.get("core_support_strong_threshold", 0.40))
    edge_support_mask = core_support_track >= float(settings.get("core_support_edge_threshold", 0.32))

    accepted_mask = seed_mask & support_mask
    accepted_mask = smooth_vowel_mask(accepted_mask, times, max_gap_s=min(bridge_gap_s, 0.012), min_run_s=min_segment_s)
    boundary_mask = smooth_vowel_mask(boundary_mask & (edge_support_mask | accepted_mask), times, max_gap_s=min(bridge_gap_s, 0.010), min_run_s=0.0)

    f1_values = np.asarray(f1, dtype=np.float64)
    f2_values = np.asarray(f2, dtype=np.float64)
    if len(times) >= 3:
        dt = np.gradient(times)
        dt[dt == 0.0] = np.nan
        df1 = np.gradient(f1_values) / dt
        df2 = np.gradient(f2_values) / dt
        stable_mask = (
            accepted_mask
            & np.isfinite(df1)
            & np.isfinite(df2)
            & (np.abs(df1) <= float(settings["max_core_df1_hz_per_s"]))
            & (np.abs(df2) <= float(settings["max_core_df2_hz_per_s"]))
            & strong_support_mask
        )
        stable_mask = smooth_vowel_mask(
            stable_mask,
            times,
            max_gap_s=min(bridge_gap_s, 0.015),
            min_run_s=float(settings["min_core_s"]),
        )
    else:
        stable_mask = seed_mask.copy()

    return scores, seed_mask, accepted_mask, boundary_mask, stable_mask

# ---------- IPA labeling ----------
def ipa_candidates(f1, f2):
    cands = []
    for sym, rr in VOWEL_RANGES.items():
        if rr["F1"][0] <= f1 <= rr["F1"][1] and rr["F2"][0] <= f2 <= rr["F2"][1]:
            cands.append(sym)
    return cands

def format_alt_label(cands):
    if not cands:
        return "[?]"
    cands_sorted = sorted(cands, key=lambda s: _ORDER.index(s) if s in _ORDER else 999)
    if len(cands_sorted) == 1:
        return f"[{cands_sorted[0]}]"
    return "[" + "–".join(cands_sorted) + "]"

def best_guess_symbol(f1, f2, restrict_to=None):
    keys = list(VOWEL_REF_POINTS.keys())
    if restrict_to:
        r = [k for k in restrict_to if k in VOWEL_REF_POINTS]
        if r:
            keys = r

    best = None
    best_d = None
    for sym in keys:
        F1r, F2r = VOWEL_REF_POINTS[sym]
        d = (f1 - F1r)**2 + (f2 - F2r)**2
        if best_d is None or d < best_d:
            best_d = d
            best = sym
    return best or "?"


def estimate_frame_edges(t: np.ndarray):
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return np.array([], dtype=np.float64)
    if len(t) == 1:
        center = float(t[0])
        return np.array([max(0.0, center - 0.01), center + 0.01], dtype=np.float64)

    edges = np.empty(len(t) + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (t[:-1] + t[1:])
    edges[0] = max(0.0, float(t[0]) - 0.5 * float(t[1] - t[0]))
    edges[-1] = float(t[-1]) + 0.5 * float(t[-1] - t[-2])
    return edges


def choose_core_peak_index(t, f1, f2, df1, df2, run_start, run_end, segment_start, segment_end, med_f1, med_f2, frame_scores=None):
    idxs = np.arange(run_start, run_end + 1, dtype=int)
    if len(idxs) == 0:
        return int(run_start)
    if len(idxs) == 1:
        return int(idxs[0])

    local_f1 = np.asarray(f1[idxs], dtype=np.float64)
    local_f2 = np.asarray(f2[idxs], dtype=np.float64)
    local_df1 = np.asarray(df1[idxs], dtype=np.float64)
    local_df2 = np.asarray(df2[idxs], dtype=np.float64)

    stability = 1.0 - np.maximum(np.abs(local_df1) / 900.0, np.abs(local_df2) / 1800.0)
    stability = np.asarray([clamp01(v) for v in stability], dtype=np.float64)

    formant_fit = 1.0 - 0.5 * ((np.abs(local_f1 - med_f1) / 160.0) + (np.abs(local_f2 - med_f2) / 260.0))
    formant_fit = np.asarray([clamp01(v) for v in formant_fit], dtype=np.float64)

    segment_center = 0.5 * (float(t[segment_start]) + float(t[segment_end]))
    segment_half = max(0.03, 0.5 * (float(t[segment_end]) - float(t[segment_start])))
    position = 1.0 - (np.abs(np.asarray(t[idxs], dtype=np.float64) - segment_center) / segment_half)
    position = np.asarray([clamp01(v) for v in position], dtype=np.float64)

    local_scores = np.zeros(len(idxs), dtype=np.float64)
    if frame_scores is not None:
        local_scores = np.asarray([clamp01(frame_scores[index]) for index in idxs], dtype=np.float64)

    score = (0.35 * stability) + (0.30 * formant_fit) + (0.10 * position) + (0.25 * local_scores)
    if not np.any(np.isfinite(score)):
        return int(idxs[len(idxs) // 2])
    return int(idxs[int(np.nanargmax(score))])


def apply_timing_prior_to_core(base_start_s, base_end_s, peak_time_s, segment_start_s, segment_end_s, timing_prior):
    start_s = float(base_start_s)
    end_s = float(base_end_s)
    source = "stable_run"

    if not timing_prior:
        start_s = max(segment_start_s, min(start_s, segment_end_s))
        end_s = max(start_s + 0.01, min(end_s, segment_end_s))
        return start_s, end_s, source

    prior_left_s = max(0.0, float(timing_prior.get("peak_minus_start_s") or 0.0))
    prior_right_s = float(timing_prior.get("end_minus_peak_s") or 0.0)
    prior_duration_s = max(0.03, float(timing_prior.get("duration_s") or 0.03))

    prior_start_s = max(segment_start_s, peak_time_s - prior_left_s)
    prior_end_s = min(segment_end_s, peak_time_s + prior_right_s)
    if prior_end_s <= prior_start_s:
        prior_end_s = min(segment_end_s, prior_start_s + prior_duration_s)

    start_s = max(segment_start_s, (0.40 * start_s) + (0.60 * prior_start_s))
    end_s = min(segment_end_s, (0.08 * end_s) + (0.92 * prior_end_s))

    max_duration_s = min(segment_end_s - segment_start_s, max(prior_duration_s * 1.20, 0.04))
    if (end_s - start_s) > max_duration_s:
        end_s = min(segment_end_s, start_s + max_duration_s)

    min_duration_s = min(max_duration_s, max(0.03, prior_duration_s * 0.55))
    if (end_s - start_s) < min_duration_s:
        center_s = float(peak_time_s)
        half_width = 0.5 * min_duration_s
        start_s = max(segment_start_s, center_s - half_width)
        end_s = min(segment_end_s, center_s + half_width)
        if (end_s - start_s) < min_duration_s:
            start_s = max(segment_start_s, end_s - min_duration_s)

    start_s = max(segment_start_s, min(start_s, segment_end_s))
    end_s = max(start_s + 0.01, min(end_s, segment_end_s))
    source = "stable_run_calibrated"
    return start_s, end_s, source


# ---------- Auto “single-vowel core” inside glides ----------
def extract_simple_cores(
    t,
    f1,
    f2,
    accepted_mask,
    segments,
    boundary_timing_priors=None,
    seed_mask=None,
    boundary_mask=None,
    stable_mask=None,
    frame_scores=None,
    settings=None,
):
    settings = settings or {}
    max_df1 = float(settings.get("max_core_df1_hz_per_s", 900.0))
    max_df2 = float(settings.get("max_core_df2_hz_per_s", 1800.0))
    min_core_ms = float(settings.get("min_core_s", 0.035)) * 1000.0
    boundary_score_floor = float(settings.get("segment_boundary_score", 0.36))

    if len(t) < 3:
        return []

    dt = np.gradient(t)
    dt[dt == 0] = np.nan
    df1 = np.gradient(f1) / dt
    df2 = np.gradient(f2) / dt
    stable = (
        np.asarray(stable_mask, dtype=bool)
        if stable_mask is not None and len(stable_mask) == len(t)
        else (np.abs(df1) <= max_df1) & (np.abs(df2) <= max_df2) & accepted_mask
    )
    frame_edges = estimate_frame_edges(t)

    cores = []
    for s0, s1 in segments:
        idx = np.arange(s0, s1 + 1)
        ok = stable[idx]

        runs = []
        in_run = False
        r0 = None
        for k, v in zip(idx, ok):
            if v and not in_run:
                in_run = True
                r0 = k
            if in_run and (not v):
                runs.append((r0, k - 1))
                in_run = False
        if in_run:
            runs.append((r0, s1))

        good = []
        for a, b in runs:
            if b <= a:
                continue
            dur_ms = (t[b] - t[a]) * 1000.0
            if dur_ms >= min_core_ms:
                if seed_mask is not None and len(seed_mask) == len(t):
                    if not np.any(seed_mask[a : b + 1]):
                        continue
                good.append((a, b))

        if not good:
            continue

        if frame_scores is not None and len(frame_scores) == len(t):
            a, b = max(
                good,
                key=lambda rr: (
                    float(np.nanmean(frame_scores[rr[0] : rr[1] + 1])),
                    (t[rr[1]] - t[rr[0]]),
                ),
            )
        else:
            a, b = max(good, key=lambda rr: (t[rr[1]] - t[rr[0]]))
        idxs = np.arange(a, b + 1)
        med_f1 = float(np.nanmedian(f1[idxs]))
        med_f2 = float(np.nanmedian(f2[idxs]))
        cands = ipa_candidates(med_f1, med_f2)
        alt = format_alt_label(cands)
        sym = best_guess_symbol(med_f1, med_f2, restrict_to=cands if cands else None)
        peak_index = choose_core_peak_index(
            t=t,
            f1=f1,
            f2=f2,
            df1=df1,
            df2=df2,
            run_start=a,
            run_end=b,
            segment_start=s0,
            segment_end=s1,
            med_f1=med_f1,
            med_f2=med_f2,
            frame_scores=frame_scores,
        )

        expanded_start = a
        expanded_end = b
        if boundary_mask is not None and len(boundary_mask) == len(t):
            peak_score = float(frame_scores[peak_index]) if frame_scores is not None and len(frame_scores) == len(t) else boundary_score_floor
            local_floor = max(boundary_score_floor, peak_score * 0.62)
            while expanded_start > s0:
                candidate = expanded_start - 1
                if not boundary_mask[candidate]:
                    break
                if frame_scores is not None and len(frame_scores) == len(t) and frame_scores[candidate] < local_floor:
                    break
                expanded_start = candidate
            while expanded_end < s1:
                candidate = expanded_end + 1
                if not boundary_mask[candidate]:
                    break
                if frame_scores is not None and len(frame_scores) == len(t) and frame_scores[candidate] < local_floor:
                    break
                expanded_end = candidate

        base_start_s = float(frame_edges[expanded_start])
        base_end_s = float(frame_edges[expanded_end + 1])
        segment_start_s = float(frame_edges[s0])
        segment_end_s = float(frame_edges[s1 + 1])
        peak_time_s = float(t[peak_index])
        t0, t1, source = apply_timing_prior_to_core(
            base_start_s=base_start_s,
            base_end_s=base_end_s,
            peak_time_s=peak_time_s,
            segment_start_s=segment_start_s,
            segment_end_s=segment_end_s,
            timing_prior=(boundary_timing_priors or {}).get(sym),
        )

        cores.append({
            "t0": float(t0),
            "t1": float(t1),
            "tc": float(peak_time_s),
            "med_f1": med_f1,
            "med_f2": med_f2,
            "best": sym,
            "alts": alt,
            "source": source,
            "peak_time_s": peak_time_s,
        })

    return cores

# ---------- Label de-clutter ----------
def add_non_overlapping_labels(ax, fig, xs, ys, labels, dx=0.0, dy=0.0, min_px=18, fontsize=8):
    if len(xs) == 0:
        return 0
    fig.canvas.draw()
    kept = []
    drawn = 0
    for x, y, lab in zip(xs, ys, labels):
        px, py = ax.transData.transform((x, y))
        ok = True
        for qpx, qpy in kept:
            if (px - qpx)**2 + (py - qpy)**2 < (min_px**2):
                ok = False
                break
        if not ok:
            continue
        kept.append((px, py))
        ax.text(x + dx, y + dy, lab, fontsize=fontsize, bbox=LABEL_BBOX, color="black")
        drawn += 1
    return drawn

# ---------- Playback ----------
def try_import_sounddevice():
    try:
        import sounddevice as sd
        return sd
    except Exception:
        return None

class SDPlayer:
    def __init__(self, sd_module):
        self.sd = sd_module
        self.stream = None
        self.lock = threading.Lock()
        self.playing = False
        self.start_monotonic = None
        self.pos = 0
        self.x = None
        self.sr = None
        self.playback_start_s = 0.0
        self.playback_end_s = 0.0

    def play(self, x: np.ndarray, sr: int, start_s: float = 0.0, end_s: float | None = None):
        self.stop()

        x = np.asarray(x, dtype=np.float32).reshape(-1)
        sr = int(sr)
        total_samples = len(x)
        total_dur_s = total_samples / float(sr) if sr > 0 else 0.0

        start_s = max(0.0, float(start_s))
        if end_s is None:
            end_s = total_dur_s
        end_s = max(start_s, min(float(end_s), total_dur_s))

        start_idx = int(round(start_s * sr))
        end_idx = int(round(end_s * sr))
        end_idx = max(start_idx + 1, min(end_idx, total_samples))

        with self.lock:
            self.x = x[start_idx:end_idx].copy()
            self.sr = sr
            self.pos = 0
            self.playing = True
            self.start_monotonic = time.monotonic()
            self.playback_start_s = start_idx / float(sr)
            self.playback_end_s = end_idx / float(sr)

        def callback(outdata, frames, time_info, status):
            with self.lock:
                if not self.playing or self.x is None:
                    outdata.fill(0)
                    raise self.sd.CallbackStop()

                end = self.pos + frames
                chunk = self.x[self.pos:end]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                    self.pos = end
                    self.playing = False
                    raise self.sd.CallbackStop()
                else:
                    outdata[:, 0] = chunk
                    self.pos = end

        self.stream = self.sd.OutputStream(
            samplerate=self.sr,
            channels=1,
            dtype="float32",
            callback=callback,
            blocksize=0
        )
        self.stream.start()

    def stop(self):
        with self.lock:
            self.playing = False
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
        self.stream = None

    def is_playing(self):
        with self.lock:
            return bool(self.playing)

    def current_time(self):
        with self.lock:
            if self.start_monotonic is None:
                return None
            elapsed = time.monotonic() - self.start_monotonic
            current = self.playback_start_s + elapsed
            if self.playback_end_s > self.playback_start_s:
                current = min(current, self.playback_end_s)
            return current

# ---------- App ----------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Vowel finder: auto Praat formants + vowel-only display")
        self._maximize_window()

        self.sr = None
        self.x = None
        self.audio_path = None
        self.analysis_audio_path = None
        self.last_praat_text = ""
        self.last_praat_error = ""
        self.last_extraction_settings = None
        self.analysis_note = ""
        self.config_path = get_config_path()
        self.saved_settings = self.load_settings()
        self.boundary_timing_priors = {}
        self.boundary_timing_priors_path = ""
        self.refresh_boundary_timing_priors()

        self.sd = try_import_sounddevice()
        self.player = SDPlayer(self.sd) if self.sd is not None else None
        self.play_after_id = None

        self.formants = None
        self.accepted_mask = None
        self.segments = None
        self.points = None
        self.all_points = None
        self.cores = None

        self.sel_wave_span = None
        self.sel_spec_span = None
        self.sel_vowel_marker = None
        self.sel_vowel_points_artist = None
        self.sel_spec_points_f1 = None
        self.sel_spec_points_f2 = None
        self.sel_text_artist = None
        self.selected_time_range = None
        self.force_full_time_view = True

        self.vowel_points_artist = None
        self.vowel_point_size = 45.0
        self.vowel_selected_size = 55.0
        self.vowel_base_face_rgba = np.array(mcolors.to_rgba("C0", alpha=0.75))
        self.vowel_base_edge_rgba = np.array(mcolors.to_rgba("black", alpha=1.0))
        self.vowel_dim_face_rgba = np.array(mcolors.to_rgba("#B8B8B8", alpha=0.35))
        self.vowel_dim_edge_rgba = np.array(mcolors.to_rgba("#777777", alpha=0.35))

        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        row = tk.Frame(left)
        row.pack(fill=tk.X)
        tk.Button(row, text="Open audio...", command=self.open_audio).pack(side=tk.LEFT)
        tk.Button(row, text="Re-extract (Praat)", command=self.extract_formants_from_current_audio).pack(side=tk.LEFT, padx=6)
        tk.Button(row, text="Clear", command=self.clear_audio).pack(side=tk.LEFT, padx=6)

        row2 = tk.Frame(left)
        row2.pack(fill=tk.X, pady=(6, 0))
        tk.Button(row2, text="Play", command=self.play_audio).pack(side=tk.LEFT)
        tk.Button(row2, text="Play selection", command=self.play_selection_audio).pack(side=tk.LEFT, padx=6)
        tk.Button(row2, text="Stop", command=self.stop_audio).pack(side=tk.LEFT, padx=6)
        tk.Button(row2, text="Analyze + Plot", command=self.analyze_and_plot).pack(side=tk.LEFT, padx=6)

        self.info = tk.Label(left, text="No audio loaded.", justify="left", wraplength=430)
        self.info.pack(anchor="w", pady=(8, 6))

        tk.Label(left, text="Formant TSV").pack(anchor="w")
        self.text = tk.Text(left, width=58, height=10)
        self.text.pack(fill=tk.BOTH, expand=False)
        self.text.bind("<<Modified>>", self.on_text_modified)

        tk.Label(left, text="Praat extraction settings").pack(anchor="w", pady=(10, 2))

        self.var_speaker_preset = tk.StringVar(value=str(self.saved_settings.get("speaker_preset", DEFAULT_SETTINGS["speaker_preset"])))
        self.var_praat_time_step = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("praat_time_step", DEFAULT_SETTINGS["praat_time_step"]), DEFAULT_SETTINGS["praat_time_step"]))
        self.var_praat_max_formants = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("praat_max_formants", DEFAULT_SETTINGS["praat_max_formants"]), DEFAULT_SETTINGS["praat_max_formants"]))
        self.var_praat_ceiling_hz = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("praat_ceiling_hz", DEFAULT_SETTINGS["praat_ceiling_hz"]), DEFAULT_SETTINGS["praat_ceiling_hz"]))
        self.var_praat_window_length = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("praat_window_length", DEFAULT_SETTINGS["praat_window_length"]), DEFAULT_SETTINGS["praat_window_length"]))
        self.var_praat_pre_emphasis_hz = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("praat_pre_emphasis_hz", DEFAULT_SETTINGS["praat_pre_emphasis_hz"]), DEFAULT_SETTINGS["praat_pre_emphasis_hz"]))

        self._build_praat_settings_ui(left)

        tk.Label(left, text="Display/filter settings").pack(anchor="w", pady=(10, 2))

        self.var_voicing = tk.DoubleVar(value=_coerce_float(self.saved_settings.get("voicing", DEFAULT_SETTINGS["voicing"]), DEFAULT_SETTINGS["voicing"]))
        self.var_reject_rhotics = tk.IntVar(value=_coerce_int(self.saved_settings.get("reject_rhotics", DEFAULT_SETTINGS["reject_rhotics"]), DEFAULT_SETTINGS["reject_rhotics"]))
        self.var_label_density = tk.IntVar(value=_coerce_int(self.saved_settings.get("label_density", DEFAULT_SETTINGS["label_density"]), DEFAULT_SETTINGS["label_density"]))
        self.var_spec_maxhz = tk.IntVar(value=_coerce_int(self.saved_settings.get("spec_maxhz", DEFAULT_SETTINGS["spec_maxhz"]), DEFAULT_SETTINGS["spec_maxhz"]))
        self.var_show_only_vowels = tk.IntVar(value=_coerce_int(self.saved_settings.get("show_only_vowels", DEFAULT_SETTINGS["show_only_vowels"]), DEFAULT_SETTINGS["show_only_vowels"]))

        self._bind_settings_traces()

        self._add_slider(
            left,
            "Voicing strictness",
            "Higher = fewer false positives. Lower = more detections.",
            self.var_voicing, 0.15, 0.60, 0.01
        )

        tk.Checkbutton(
            left,
            text="Reject r-like false positives (F2 close to F3)",
            variable=self.var_reject_rhotics,
            command=self.analyze_and_plot
        ).pack(anchor="w", pady=(4, 0))

        tk.Checkbutton(
            left,
            text="Show only likely-vowel formant tracks",
            variable=self.var_show_only_vowels,
            command=self.analyze_and_plot
        ).pack(anchor="w", pady=(4, 0))

        self._add_slider(
            left,
            "Label spacing",
            "Higher = fewer labels. Lower = more labels.",
            self.var_label_density, 10, 40, 1, is_int=True
        )

        self._add_slider(
            left,
            "Spectrogram max Hz",
            "Higher shows more highs. 5000 is usually enough for vowels.",
            self.var_spec_maxhz, 3000, 8000, 100, is_int=True
        )

        self.status = tk.Label(left, text="", justify="left", wraplength=430)
        self.status.pack(anchor="w", pady=(8, 0))

        tk.Label(
            left,
            text=(
                "Tip: When you load audio, the script auto-runs Praat.exe to get formants.\n"
                "Drag on waveform or spectrogram to label a vowel region.\n"
                "Tab plays the current selected span."
            ),
            justify="left",
            wraplength=430
        ).pack(anchor="w", pady=(10, 0))

        right = tk.Frame(root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.fig = plt.Figure(figsize=(11, 7), dpi=100)
        self.fig.subplots_adjust(left=0.10, right=0.985, top=0.96, bottom=0.08, hspace=0.30)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.0, 1.25, 1.0])
        self.ax_wave = self.fig.add_subplot(gs[0, 0])
        self.ax_spec = self.fig.add_subplot(gs[1, 0], sharex=self.ax_wave)
        self.ax_vowel = self.fig.add_subplot(gs[2, 0])

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.time_scrollbar = tk.Scrollbar(right, orient=tk.HORIZONTAL, command=self._on_time_scrollbar)
        self.time_scrollbar.pack(fill=tk.X, pady=(4, 0))

        self._ctrl_held = False
        self.canvas_widget.bind("<Enter>", self._on_canvas_enter, add="+")
        self.root.bind_all("<KeyPress-Control_L>", self._on_ctrl_key_press, add="+")
        self.root.bind_all("<KeyPress-Control_R>", self._on_ctrl_key_press, add="+")
        self.root.bind_all("<KeyRelease-Control_L>", self._on_ctrl_key_release, add="+")
        self.root.bind_all("<KeyRelease-Control_R>", self._on_ctrl_key_release, add="+")
        self.root.bind_all("<FocusOut>", self._on_any_focus_out, add="+")
        self.root.bind_all("<ButtonRelease-1>", self._on_mouse_button_release, add="+")
        self.root.bind_all("<ButtonRelease-2>", self._on_mouse_button_release, add="+")
        self.root.bind_all("<ButtonRelease-3>", self._on_mouse_button_release, add="+")
        self.root.bind_all("<Tab>", self._on_tab_play_selection)

        self.canvas.mpl_connect("scroll_event", self._on_scroll_event)

        self.cursor_wave = None
        self.cursor_spec = None
        self.cursor_time_text = None
        self.span_wave = None
        self.span_spec = None
        self._text_dirty_guard = False
        self._updating_time_scrollbar = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.update_info_label()
        self.analyze_and_plot()

    def _maximize_window(self):
        try:
            self.root.state("zoomed")
            return
        except Exception:
            pass

        try:
            self.root.attributes("-zoomed", True)
            return
        except Exception:
            pass

        try:
            width = int(self.root.winfo_screenwidth())
            height = int(self.root.winfo_screenheight())
            self.root.geometry(f"{width}x{height}+0+0")
        except Exception:
            pass

    def load_settings(self):
        data = dict(DEFAULT_SETTINGS)
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    data.update(raw)
        except Exception:
            pass
        return data

    def _get_saved_praat_path(self):
        return str(self.saved_settings.get("praat_exe_path", DEFAULT_SETTINGS["praat_exe_path"])).strip()

    def _get_saved_boundary_priors_path(self):
        return str(
            self.saved_settings.get(
                "boundary_timing_priors_json",
                DEFAULT_SETTINGS["boundary_timing_priors_json"]
            )
        ).strip()

    def _choose_praat_executable(self):
        initial = self._get_saved_praat_path()
        if not initial:
            detected = find_praat_executable("")
            if detected:
                initial = detected

        if os.name == "nt":
            filetypes = [("Praat executable", "Praat.exe"), ("Executable files", "*.exe"), ("All files", "*.*")]
        else:
            filetypes = [("All files", "*.*"), ("All files", "*")]

        path = filedialog.askopenfilename(
            title="Select Praat executable",
            initialfile=os.path.basename(initial) if initial else "",
            initialdir=os.path.dirname(initial) if initial and os.path.isdir(os.path.dirname(initial)) else get_script_dir(),
            filetypes=filetypes,
        )
        if not path:
            return

        self.saved_settings["praat_exe_path"] = path
        self.save_settings()
        self.update_info_label()
        self.status.config(text=f"Saved Praat path:\n{path}")

    def _clear_praat_executable(self):
        self.saved_settings["praat_exe_path"] = ""
        self.save_settings()
        self.update_info_label()
        self.status.config(text="Cleared saved Praat path. The script will auto-detect Praat again.")

    def _choose_boundary_timing_priors_json(self):
        current = self._get_saved_boundary_priors_path()
        path = filedialog.askopenfilename(
            title="Select boundary timing priors JSON",
            initialfile=os.path.basename(current) if current else "",
            initialdir=os.path.dirname(current) if current and os.path.isdir(os.path.dirname(current)) else get_script_dir(),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        self.saved_settings["boundary_timing_priors_json"] = path
        self.save_settings()
        self.refresh_boundary_timing_priors()
        self.update_info_label()
        self.analyze_and_plot()
        self.status.config(text=f"Saved boundary timing priors JSON:\n{path}")

    def _clear_boundary_timing_priors_json(self):
        self.saved_settings["boundary_timing_priors_json"] = default_boundary_timing_priors_path()
        self.save_settings()
        self.refresh_boundary_timing_priors()
        self.update_info_label()
        self.analyze_and_plot()
        default_text = self._get_saved_boundary_priors_path() or "(none)"
        self.status.config(text=f"Reset boundary timing priors JSON to:\n{default_text}")

    def refresh_boundary_timing_priors(self):
        configured_path = self._get_saved_boundary_priors_path()
        if not configured_path:
            configured_path = default_boundary_timing_priors_path()
        self.boundary_timing_priors, self.boundary_timing_priors_path = load_boundary_timing_priors(configured_path)

    def save_settings(self):
        data = self._get_ui_settings()
        tmp_path = self.config_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.config_path)
            self.saved_settings = dict(data)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _get_ui_settings(self):
        data = dict(self.saved_settings) if isinstance(getattr(self, "saved_settings", None), dict) else {}
        data.update({
            "praat_exe_path": str(data.get("praat_exe_path", DEFAULT_SETTINGS["praat_exe_path"])).strip(),
            "boundary_timing_priors_json": str(
                data.get("boundary_timing_priors_json", DEFAULT_SETTINGS["boundary_timing_priors_json"])
            ).strip(),
            "speaker_preset": str(self.var_speaker_preset.get()).strip() or DEFAULT_SETTINGS["speaker_preset"],
            "praat_time_step": _coerce_float(self.var_praat_time_step.get(), DEFAULT_SETTINGS["praat_time_step"]),
            "praat_max_formants": _coerce_float(self.var_praat_max_formants.get(), DEFAULT_SETTINGS["praat_max_formants"]),
            "praat_ceiling_hz": _coerce_float(self.var_praat_ceiling_hz.get(), DEFAULT_SETTINGS["praat_ceiling_hz"]),
            "praat_window_length": _coerce_float(self.var_praat_window_length.get(), DEFAULT_SETTINGS["praat_window_length"]),
            "praat_pre_emphasis_hz": _coerce_float(self.var_praat_pre_emphasis_hz.get(), DEFAULT_SETTINGS["praat_pre_emphasis_hz"]),
            "voicing": _coerce_float(self.var_voicing.get(), DEFAULT_SETTINGS["voicing"]),
            "reject_rhotics": _coerce_int(self.var_reject_rhotics.get(), DEFAULT_SETTINGS["reject_rhotics"]),
            "label_density": _coerce_int(self.var_label_density.get(), DEFAULT_SETTINGS["label_density"]),
            "spec_maxhz": _coerce_int(self.var_spec_maxhz.get(), DEFAULT_SETTINGS["spec_maxhz"]),
            "show_only_vowels": _coerce_int(self.var_show_only_vowels.get(), DEFAULT_SETTINGS["show_only_vowels"]),
        })
        return data

    def _bind_settings_traces(self):
        traced = [
            self.var_speaker_preset,
            self.var_praat_time_step,
            self.var_praat_max_formants,
            self.var_praat_ceiling_hz,
            self.var_praat_window_length,
            self.var_praat_pre_emphasis_hz,
            self.var_voicing,
            self.var_reject_rhotics,
            self.var_label_density,
            self.var_spec_maxhz,
            self.var_show_only_vowels,
        ]
        for var in traced:
            try:
                var.trace_add("write", self._on_any_setting_changed)
            except Exception:
                pass

    def _on_any_setting_changed(self, *_args):
        self.save_settings()
        self.update_info_label()

    def _current_praat_settings(self):
        return {
            "speaker_preset": str(self.var_speaker_preset.get()).strip() or DEFAULT_SETTINGS["speaker_preset"],
            "time_step": max(0.0, _coerce_float(self.var_praat_time_step.get(), DEFAULT_SETTINGS["praat_time_step"])),
            "max_formants": max(3.0, _coerce_float(self.var_praat_max_formants.get(), DEFAULT_SETTINGS["praat_max_formants"])),
            "ceiling_hz": max(3000.0, _coerce_float(self.var_praat_ceiling_hz.get(), DEFAULT_SETTINGS["praat_ceiling_hz"])),
            "window_length": max(0.005, _coerce_float(self.var_praat_window_length.get(), DEFAULT_SETTINGS["praat_window_length"])),
            "pre_emphasis_hz": max(0.0, _coerce_float(self.var_praat_pre_emphasis_hz.get(), DEFAULT_SETTINGS["praat_pre_emphasis_hz"])),
        }

    def _format_praat_settings_text(self, settings=None):
        s = settings or self._current_praat_settings()
        return (
            f"Preset: {s['speaker_preset']}\n"
            f"Praat extraction: time step {s['time_step']:.3f} s"
            f" ({'automatic-style' if s['time_step'] == 0.0 else 'fixed'}), "
            f"max formants {s['max_formants']:.1f}, ceiling {s['ceiling_hz']:.0f} Hz, "
            f"window {s['window_length']:.3f} s, pre-emphasis {s['pre_emphasis_hz']:.0f} Hz"
        )

    def update_info_label(self):
        priors_path = self._get_saved_boundary_priors_path() or "(default/none)"
        priors_text = (
            f"Priors: {len(self.boundary_timing_priors)} vowels loaded"
            if self.boundary_timing_priors
            else "Priors: not loaded"
        )

        praat_settings = self._current_praat_settings()
        praat_path_text = describe_praat_path(self._get_saved_praat_path())
        praat_text = (
            f"Praat: {praat_settings['max_formants']:.1f} formants, "
            f"{praat_settings['ceiling_hz']:.0f} Hz ceiling, "
            f"{praat_settings['window_length'] * 1000.0:.0f} ms window"
        )

        common_lines = [
            f"Praat path: {praat_path_text}",
            f"Priors file: {priors_path}",
            priors_text,
            praat_text,
        ]

        if self.audio_path:
            audio_name = os.path.basename(self.audio_path)
            lines = [
                f"Audio: {audio_name}",
                f"{self.sr} Hz | {praat_settings['speaker_preset']}",
            ]
            if self.analysis_audio_path and self.analysis_audio_path != self.audio_path:
                lines.append(f"Analysis: {os.path.basename(self.analysis_audio_path)}")
            lines.extend(common_lines)
            if getattr(self, "analysis_note", ""):
                lines.append(str(self.analysis_note))
            self.info.config(text="\n".join(lines))
        else:
            self.info.config(
                text="\n".join(
                    [
                        "No audio loaded.",
                        f"{praat_settings['speaker_preset']}",
                        *common_lines,
                    ]
                )
            )

    def _build_praat_settings_ui(self, parent):
        path_box = tk.Frame(parent)
        path_box.pack(fill=tk.X, pady=(2, 6))

        tk.Label(path_box, text="Praat executable").grid(row=0, column=0, sticky="w")
        tk.Button(path_box, text="Choose Praat...", command=self._choose_praat_executable).grid(row=0, column=1, sticky="w", padx=(8, 0))
        tk.Button(path_box, text="Auto-detect", command=self._clear_praat_executable).grid(row=0, column=2, sticky="w", padx=(6, 0))

        tk.Label(path_box, text="Boundary priors JSON").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Button(path_box, text="Choose JSON...", command=self._choose_boundary_timing_priors_json).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        tk.Button(path_box, text="Reset", command=self._clear_boundary_timing_priors_json).grid(row=1, column=2, sticky="w", padx=(6, 0), pady=(6, 0))

        tk.Label(
            path_box,
            text=(
                "Praat can be auto-detected from PATH or common install locations.\n"
                "You can also save an explicit executable path here.\n"
                "The priors JSON defaults to a file next to this script."
            ),
            fg="#444444",
            justify="left",
            wraplength=430
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 4))

        preset_box = tk.Frame(parent)
        preset_box.pack(fill=tk.X, pady=(2, 4))
        tk.Label(preset_box, text="Speaker preset").pack(anchor="w")
        combo = ttk.Combobox(
            preset_box,
            textvariable=self.var_speaker_preset,
            values=list(SPEAKER_PRESETS.keys()),
            state="readonly",
            width=18,
        )
        combo.pack(anchor="w")
        combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_speaker_preset())

        tk.Label(
            preset_box,
            text=(
                "Typical starting points:\n"
                "Adult male: ceiling 5000 Hz\n"
                "Adult female: ceiling 5500 to 6000 Hz\n"
                "Child: ceiling 6500+ Hz\n"
                "Time step 0 = automatic-style spacing. Use Re-extract after changes."
            ),
            fg="#444444", justify="left", wraplength=430
        ).pack(anchor="w", pady=(2, 4))

        form = tk.Frame(parent)
        form.pack(fill=tk.X)

        self._add_number_field(form, 0, "Time step (s)", self.var_praat_time_step, "0 = automatic")
        self._add_number_field(form, 1, "# formants", self.var_praat_max_formants, "Usually 5")
        self._add_number_field(form, 2, "Ceiling (Hz)", self.var_praat_ceiling_hz, "Male 5000, female 5500-6000")
        self._add_number_field(form, 3, "Window length (s)", self.var_praat_window_length, "Usually 0.025")
        self._add_number_field(form, 4, "Pre-emphasis (Hz)", self.var_praat_pre_emphasis_hz, "Usually 50")

    def _add_number_field(self, parent, row, label, var, hint):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
        entry = tk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky="w", pady=1)
        entry.bind("<FocusOut>", self._on_number_field_commit)
        entry.bind("<Return>", self._on_number_field_commit)
        tk.Label(parent, text=hint, fg="#444444").grid(row=row, column=2, sticky="w", padx=(8, 0), pady=1)

    def _on_number_field_commit(self, _evt=None):
        settings = self._current_praat_settings()
        self.var_praat_time_step.set(settings["time_step"])
        self.var_praat_max_formants.set(settings["max_formants"])
        self.var_praat_ceiling_hz.set(settings["ceiling_hz"])
        self.var_praat_window_length.set(settings["window_length"])
        self.var_praat_pre_emphasis_hz.set(settings["pre_emphasis_hz"])
        self.save_settings()
        self.update_info_label()

    def apply_speaker_preset(self):
        preset_name = str(self.var_speaker_preset.get()).strip().lower()
        preset = SPEAKER_PRESETS.get(preset_name)
        if not preset:
            return
        self.var_praat_time_step.set(preset["praat_time_step"])
        self.var_praat_max_formants.set(preset["praat_max_formants"])
        self.var_praat_ceiling_hz.set(preset["praat_ceiling_hz"])
        self.var_praat_window_length.set(preset["praat_window_length"])
        self.var_praat_pre_emphasis_hz.set(preset["praat_pre_emphasis_hz"])
        self.save_settings()
        self.update_info_label()

    def _add_slider(self, parent, title, desc, var, a, b, step, is_int=False):
        box = tk.Frame(parent)
        box.pack(fill=tk.X, pady=(4, 0))
        tk.Label(box, text=title).pack(anchor="w")
        tk.Label(box, text=desc, fg="#444444", wraplength=430, justify="left").pack(anchor="w")

        scale = tk.Scale(
            box, from_=a, to=b, resolution=step,
            orient=tk.HORIZONTAL, variable=var, length=360,
            command=lambda _x: None
        )
        scale.pack(anchor="w")

        def _released(_evt=None):
            if is_int:
                try:
                    var.set(int(float(var.get())))
                except Exception:
                    pass
            self.analyze_and_plot()

        scale.bind("<ButtonRelease-1>", _released)

    def on_close(self):
        self.stop_audio()
        self.save_settings()
        self.root.destroy()

    def open_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio files", "*.wav *.mp3 *.webm *.m4a *.aac *.ogg *.flac"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            sr, x, orig, used_path, note = load_audio_any(path)
            self.sr = sr
            self.x = x
            self.audio_path = orig
            self.analysis_audio_path = used_path
            praat_path_text = describe_praat_path(self._get_saved_praat_path())
            self.info.config(
                text=(
                    f"Audio: {orig}\n"
                    f"Analysis file: {used_path}\n"
                    f"Sample rate: {sr} Hz\n"
                    f"{note}\n"
                    f"Praat path: {praat_path_text}"
                )
            )
        except Exception as e:
            messagebox.showerror("Audio load failed", str(e))
            return

        self.stop_audio()
        self.force_full_time_view = True
        self.extract_formants_from_current_audio()

    def extract_formants_from_current_audio(self):
        if not self.analysis_audio_path:
            messagebox.showinfo("Praat", "Load an audio file first.")
            return

        settings = self._current_praat_settings()
        praat_path = self._get_saved_praat_path()

        try:
            praat_text = run_praat_formants(
                self.analysis_audio_path,
                praat_exe=praat_path,
                time_step=settings["time_step"],
                max_formants=settings["max_formants"],
                ceiling_hz=settings["ceiling_hz"],
                window_length=settings["window_length"],
                pre_emphasis_hz=settings["pre_emphasis_hz"],
            )
            self.last_extraction_settings = settings.copy()
            self.last_praat_text = praat_text
            self.last_praat_error = ""
            self._replace_text_safely(praat_text)
            self.update_info_label()
            self.status.config(
                text=(
                    "Praat formants extracted automatically using current UI settings.\n"
                    f"{self._format_praat_settings_text(settings)}\n"
                    f"Praat path: {describe_praat_path(praat_path)}"
                )
            )
        except Exception as e:
            self.last_praat_error = str(e)
            self.status.config(text=f"Praat extraction failed.\n{e}")
            messagebox.showerror("Praat extraction failed", str(e))
            self.analyze_and_plot()
            return

        self.analyze_and_plot()

    def _replace_text_safely(self, text):
        self._text_dirty_guard = True
        try:
            self.text.delete("1.0", "end")
            self.text.insert("1.0", text)
            self.text.edit_modified(False)
        finally:
            self._text_dirty_guard = False

    def clear_audio(self):
        self.stop_audio()
        self.sr = None
        self.x = None
        self.audio_path = None
        self.analysis_audio_path = None
        self.last_praat_text = ""
        self.last_praat_error = ""
        self.last_extraction_settings = None
        self.analysis_note = ""
        self.selected_time_range = None
        self.force_full_time_view = True
        self.update_info_label()
        self._replace_text_safely("")
        self.analyze_and_plot()

    def _play_time_range(self, start_s=0.0, end_s=None, empty_message="Load an audio file first."):
        if self.sr is None or self.x is None:
            messagebox.showinfo("Play", empty_message)
            return False
        if self.sd is None or self.player is None:
            messagebox.showerror("Play failed", "Install sounddevice:\npy -m pip install sounddevice")
            return False

        dur = len(self.x) / float(self.sr)
        start_s = max(0.0, float(start_s))
        if end_s is None:
            end_s = dur
        end_s = max(start_s, min(float(end_s), dur))

        if end_s - start_s < max(1.0 / float(self.sr), 0.001):
            messagebox.showinfo("Play selection", "Select a longer span first.")
            return False

        self.stop_audio()
        try:
            self.player.play(self.x, self.sr, start_s=start_s, end_s=end_s)
        except Exception as e:
            messagebox.showerror("Play failed", str(e))
            return False

        self._ensure_playback_artists()
        self.cursor_wave.set_xdata([start_s, start_s])
        self.cursor_spec.set_xdata([start_s, start_s])
        self.cursor_time_text.set_text(f"t = {start_s:.3f} s")
        self.canvas.draw_idle()
        self._schedule_cursor_update()
        return True

    def play_audio(self):
        self._play_time_range(0.0, None)

    def play_selection_audio(self):
        if self.selected_time_range is None:
            messagebox.showinfo("Play selection", "Select a span on the waveform or spectrogram first.")
            return
        start_s, end_s = self.selected_time_range
        if self._play_time_range(start_s, end_s, empty_message="Load an audio file first."):
            self.status.config(text=f"Playing selected span {start_s:.3f}-{end_s:.3f}s.")

    def _on_tab_play_selection(self, _event=None):
        self.play_selection_audio()
        return "break"

    def stop_audio(self):
        if self.play_after_id is not None:
            try:
                self.root.after_cancel(self.play_after_id)
            except Exception:
                pass
            self.play_after_id = None
        if self.player is not None:
            self.player.stop()

    def _schedule_cursor_update(self):
        if self.play_after_id is not None:
            try:
                self.root.after_cancel(self.play_after_id)
            except Exception:
                pass
        self.play_after_id = self.root.after(30, self._update_playback_cursor_safe)

    def _update_playback_cursor_safe(self):
        try:
            self._update_playback_cursor()
        except Exception as e:
            self.stop_audio()
            self.status.config(text=f"Playback cursor stopped: {e}")

    def _update_playback_cursor(self):
        if self.player is None or not self.player.is_playing():
            self.play_after_id = None
            return
        pt = self.player.current_time()
        if pt is None:
            self.stop_audio()
            return

        dur = len(self.x) / self.sr
        if pt > dur:
            pt = dur

        self._ensure_playback_artists()
        self.cursor_wave.set_xdata([pt, pt])
        self.cursor_spec.set_xdata([pt, pt])
        self.cursor_time_text.set_text(f"t = {pt:.3f} s")
        self.canvas.draw_idle()
        self._schedule_cursor_update()

    def _ensure_playback_artists(self):
        if self.cursor_wave is None:
            self.cursor_wave = self.ax_wave.axvline(0, linewidth=1.5, alpha=0.9)
        if self.cursor_spec is None:
            self.cursor_spec = self.ax_spec.axvline(0, linewidth=1.5, alpha=0.9)
        if self.cursor_time_text is None:
            self.cursor_time_text = self.ax_wave.text(
                0.01, 0.90, "", transform=self.ax_wave.transAxes,
                fontsize=9, bbox=LABEL_BBOX, color="black"
            )

    def _get_visible_time_window(self):
        if self.sr is None or self.x is None or len(self.x) == 0:
            return None

        dur = len(self.x) / float(self.sr)
        xmin, xmax = self.ax_wave.get_xlim()
        xmin = max(0.0, float(xmin))
        xmax = min(float(dur), float(xmax))

        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            return None
        if xmax <= xmin:
            return None
        return xmin, xmax


    def on_text_modified(self, _evt):
        if self._text_dirty_guard:
            return
        self._text_dirty_guard = True
        try:
            self.text.edit_modified(False)
            self.stop_audio()
            self.analyze_and_plot()
        finally:
            self._text_dirty_guard = False


    def _sync_time_axes(self, xmin=None, xmax=None):
        if self.sr is None or self.x is None:
            self._update_time_scrollbar()
            return

        dur = len(self.x) / float(self.sr)
        if xmin is None or xmax is None:
            xmin, xmax = 0.0, dur

        xmin = max(0.0, float(xmin))
        xmax = min(dur, float(xmax))
        if xmax <= xmin:
            xmax = min(dur, xmin + 0.001)

        self.ax_wave.set_xlim(xmin, xmax)
        self.ax_spec.set_xlim(xmin, xmax)
        self.ax_wave.margins(x=0)
        self.ax_spec.margins(x=0)
        self._update_time_scrollbar()

    def _x_zoom_limits(self, current_xlim, center, scale_factor, xmin_bound, xmax_bound, min_span=0.010):
        left, right = sorted([float(current_xlim[0]), float(current_xlim[1])])
        current_span = max(right - left, min_span)
        new_span = max(min_span, current_span * float(scale_factor))
        max_span = max(xmax_bound - xmin_bound, min_span)
        new_span = min(new_span, max_span)

        if center is None or not np.isfinite(center):
            center = 0.5 * (left + right)
        center = min(max(float(center), xmin_bound), xmax_bound)

        new_left = center - new_span * 0.5
        new_right = center + new_span * 0.5

        if new_left < xmin_bound:
            shift = xmin_bound - new_left
            new_left += shift
            new_right += shift
        if new_right > xmax_bound:
            shift = new_right - xmax_bound
            new_left -= shift
            new_right -= shift

        new_left = max(xmin_bound, new_left)
        new_right = min(xmax_bound, new_right)

        if new_right - new_left < min_span:
            new_right = min(xmax_bound, new_left + min_span)

        return new_left, new_right

    def _on_canvas_enter(self, _event=None):
        try:
            self.canvas_widget.focus_set()
        except Exception:
            pass

    def _on_ctrl_key_press(self, _event=None):
        self._ctrl_held = True

    def _on_ctrl_key_release(self, _event=None):
        self._ctrl_held = False

    def _on_any_focus_out(self, _event=None):
        self._ctrl_held = False

    def _on_mouse_button_release(self, _event=None):
        if not self.root.focus_displayof():
            self._ctrl_held = False

    def _ctrl_pressed_for_event(self, event):
        if getattr(self, "_ctrl_held", False):
            return True

        key = str(getattr(event, "key", "") or "").lower()
        if "control" in key or key == "ctrl":
            return True

        gui_event = getattr(event, "guiEvent", None)
        state = getattr(gui_event, "state", None)
        if state is not None:
            try:
                if int(state) & 0x0004:
                    return True
            except Exception:
                pass
        return False

    def _update_time_scrollbar(self):
        if not hasattr(self, "time_scrollbar"):
            return

        dur = None
        if self.sr is not None and self.x is not None and len(self.x) > 0:
            dur = len(self.x) / float(self.sr)

        self._updating_time_scrollbar = True
        try:
            if not dur or dur <= 0:
                self.time_scrollbar.set(0.0, 1.0)
                return

            xmin, xmax = self.ax_wave.get_xlim()
            xmin = max(0.0, float(xmin))
            xmax = min(float(dur), float(xmax))
            if xmax <= xmin:
                xmax = min(float(dur), xmin + 0.001)

            first = xmin / float(dur)
            last = xmax / float(dur)
            first = min(max(first, 0.0), 1.0)
            last = min(max(last, first), 1.0)
            self.time_scrollbar.set(first, last)
        finally:
            self._updating_time_scrollbar = False

    def _on_time_scrollbar(self, *args):
        if self._updating_time_scrollbar:
            return
        if self.sr is None or self.x is None or len(self.x) == 0:
            return
        if not args:
            return

        dur = len(self.x) / float(self.sr)
        cur_left, cur_right = self.ax_wave.get_xlim()
        window = max(0.010, float(cur_right) - float(cur_left))

        cmd = args[0]
        if cmd == "moveto" and len(args) >= 2:
            frac = float(args[1])
            new_left = frac * dur
        elif cmd == "scroll" and len(args) >= 3:
            step = int(args[1])
            what = str(args[2]).lower()
            if what == "pages":
                delta = window * 0.9 * step
            else:
                delta = window * 0.1 * step
            new_left = float(cur_left) + delta
        else:
            return

        new_left = max(0.0, min(float(dur) - window, new_left)) if dur > window else 0.0
        new_right = min(float(dur), new_left + window)
        if new_right - new_left < 0.010 and dur >= 0.010:
            new_left = max(0.0, new_right - 0.010)
        self._sync_time_axes(new_left, new_right)
        self.canvas.draw_idle()

    def _pan_time_window(self, delta_s):
        if self.sr is None or self.x is None or len(self.x) == 0:
            return
        dur = len(self.x) / float(self.sr)
        cur_left, cur_right = self.ax_wave.get_xlim()
        window = max(0.010, float(cur_right) - float(cur_left))
        if dur <= window:
            self._sync_time_axes(0.0, dur)
            return

        new_left = max(0.0, min(dur - window, float(cur_left) + float(delta_s)))
        new_right = min(dur, new_left + window)
        self._sync_time_axes(new_left, new_right)

    def _on_scroll_event(self, event):
        if self.sr is None or self.x is None:
            return
        if event.inaxes not in (self.ax_wave, self.ax_spec):
            return

        dur = len(self.x) / float(self.sr)
        xlim = self.ax_wave.get_xlim()

        if self._ctrl_pressed_for_event(event):
            button = str(getattr(event, "button", "")).lower()
            if button == "up":
                scale = 0.80
            elif button == "down":
                scale = 1.25
            else:
                step = getattr(event, "step", 0)
                scale = 0.80 if step > 0 else 1.25

            xmin, xmax = self._x_zoom_limits(
                xlim,
                getattr(event, "xdata", None),
                scale,
                0.0,
                dur,
                min_span=0.010,
            )
            self._sync_time_axes(xmin, xmax)
            self.canvas.draw_idle()
            return

        cur_left, cur_right = xlim
        window = max(0.010, float(cur_right) - float(cur_left))
        if dur <= window:
            return

        button = str(getattr(event, "button", "")).lower()
        step = getattr(event, "step", 0)
        if button == "up" or step > 0:
            direction = -1.0
        elif button == "down" or step < 0:
            direction = 1.0
        else:
            return

        self._pan_time_window(direction * window * 0.10)
        self.canvas.draw_idle()

    def analyze_and_plot(self):
        self.stop_audio()
        self.refresh_boundary_timing_priors()
        self.update_info_label()

        preserved_time_window = None if self.force_full_time_view else self._get_visible_time_window()

        self.ax_wave.clear()
        self.ax_spec.clear()
        self.ax_vowel.clear()

        self.cursor_wave = None
        self.cursor_spec = None
        self.cursor_time_text = None

        self.selected_time_range = None

        self.span_wave = None
        self.span_spec = None

        self.formants = None
        self.accepted_mask = None
        self.segments = []
        self.points = None
        self.all_points = None
        self.cores = []
        self.vowel_points_artist = None

        if self.sr is not None and self.x is not None:
            ta = np.arange(len(self.x)) / self.sr
            self.ax_wave.plot(ta, self.x, linewidth=0.8)
            self.ax_wave.set_title("Waveform (drag to select a vowel span)")
        else:
            self.ax_wave.set_title("Waveform (load audio)")
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amp")

        txt = self.text.get("1.0", "end").strip()
        parsed = parse_praat_listing(txt) if txt else None
        if parsed is None:
            self.draw_vowel_space_reference()
            self.ax_spec.set_title("Spectrogram (load audio; Praat TSV will fill automatically)")
            self.ax_wave.tick_params(axis="x", labelbottom=False)
            self._sync_time_axes()
            self.canvas.draw()
            if self.audio_path:
                self.status.config(text="No valid formant TSV loaded. Try Re-extract (Praat).")
            else:
                self.status.config(text="Load audio. Praat formants will be extracted automatically.")
            return

        t, f1, f2, f3, f4 = parsed
        self.formants = (t, f1, f2, f3, f4)

        detection_settings = dict(DEFAULT_SETTINGS)
        if isinstance(self.saved_settings, dict):
            detection_settings.update(self.saved_settings)
        detection_settings.update({
            "speaker_preset": str(self.var_speaker_preset.get()).strip() or DEFAULT_SETTINGS["speaker_preset"],
            "praat_time_step": _coerce_float(self.var_praat_time_step.get(), DEFAULT_SETTINGS["praat_time_step"]),
            "praat_max_formants": _coerce_float(self.var_praat_max_formants.get(), DEFAULT_SETTINGS["praat_max_formants"]),
            "praat_ceiling_hz": _coerce_float(self.var_praat_ceiling_hz.get(), DEFAULT_SETTINGS["praat_ceiling_hz"]),
            "praat_window_length": _coerce_float(self.var_praat_window_length.get(), DEFAULT_SETTINGS["praat_window_length"]),
            "praat_pre_emphasis_hz": _coerce_float(self.var_praat_pre_emphasis_hz.get(), DEFAULT_SETTINGS["praat_pre_emphasis_hz"]),
            "voicing": _coerce_float(self.var_voicing.get(), DEFAULT_SETTINGS["voicing"]),
            "reject_rhotics": _coerce_int(self.var_reject_rhotics.get(), DEFAULT_SETTINGS["reject_rhotics"]),
        })

        if self.sr is not None and self.x is not None:
            max_hz = int(self.var_spec_maxhz.get())
            dyn_db = 55
            nfft = 1024
            hop = 256

            ff, tt, Z = stft(self.x, fs=self.sr, nperseg=nfft, noverlap=nfft-hop, nfft=nfft, padded=False, boundary=None)
            S = np.abs(Z) + 1e-12
            S_db = 20.0 * np.log10(S)
            top = np.percentile(S_db, 99.0)
            vmin = top - dyn_db
            vmax = top

            keep = ff <= max_hz
            f2d = ff[keep]
            S2d = S_db[keep, :]

            time_edges = spectrogram_time_edges(tt, self.sr, nfft, hop, len(self.x))
            freq_edges = np.linspace(float(f2d[0]), float(f2d[-1]), len(f2d) + 1)
            self.ax_spec.pcolormesh(
                time_edges,
                freq_edges,
                S2d,
                shading="auto",
                cmap="gray_r",
                vmin=vmin,
                vmax=vmax,
            )
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Hz")

        if self.sr is None or self.x is None:
            accepted_mask = np.ones(len(t), dtype=bool)
            seed_mask = np.ones(len(t), dtype=bool)
            boundary_mask = np.ones(len(t), dtype=bool)
            stable_mask = np.ones(len(t), dtype=bool)
            frame_scores = np.ones(len(t), dtype=np.float64)
        else:
            frame_rows = build_scored_vowel_frame_rows(t, f1, f2, f3, self.sr, self.x, detection_settings)
            stability_track = compute_formant_stability_track(frame_rows)
            stripe_track = compute_spectral_stripe_track(
                self.x,
                self.sr,
                frame_rows,
                window_s=float(detection_settings["praat_window_length"]),
                max_freq_hz=float(self.var_spec_maxhz.get()),
            )
            core_support_track = smooth_track((0.55 * stability_track) + (0.45 * stripe_track), window=5)
            frame_scores, seed_mask, accepted_mask, boundary_mask, stable_mask = build_score_masks(
                t,
                f1,
                f2,
                frame_rows,
                detection_settings,
                core_support_track=core_support_track,
            )

        segments = mask_to_segments(accepted_mask)
        self.accepted_mask = accepted_mask
        self.segments = segments

        show_only = bool(self.var_show_only_vowels.get())
        plot_f1 = np.where(accepted_mask, f1, np.nan) if show_only else f1
        plot_f2 = np.where(accepted_mask, f2, np.nan) if show_only else f2

        self.ax_spec.plot(t, plot_f1, linewidth=1.2, label="F1")
        self.ax_spec.plot(t, plot_f2, linewidth=1.2, label="F2")
        self.ax_spec.legend(loc="upper right")
        if show_only:
            self.ax_spec.set_title("Spectrogram + likely-vowel F1/F2")
        else:
            self.ax_spec.set_title("Spectrogram + full F1/F2")

        pts_t = t[accepted_mask]
        pts_f1 = f1[accepted_mask]
        pts_f2 = f2[accepted_mask]
        pt_cands = [ipa_candidates(a, b) for a, b in zip(pts_f1, pts_f2)]
        pt_labels = [format_alt_label(c) for c in pt_cands]

        self.points = {"t": pts_t, "f1": pts_f1, "f2": pts_f2, "labels": pt_labels}

        all_mask = np.isfinite(f1) & np.isfinite(f2)
        all_pts_t = t[all_mask]
        all_pts_f1 = f1[all_mask]
        all_pts_f2 = f2[all_mask]
        all_pt_cands = [ipa_candidates(a, b) for a, b in zip(all_pts_f1, all_pts_f2)]
        all_pt_labels = [format_alt_label(c) for c in all_pt_cands]
        self.all_points = {"t": all_pts_t, "f1": all_pts_f1, "f2": all_pts_f2, "labels": all_pt_labels}

        if self.sr is not None and self.x is not None and len(pts_t) > 0:
            ya = np.interp(pts_t, np.arange(len(self.x))/self.sr, self.x)
            self.ax_wave.scatter(pts_t, ya, s=14, alpha=0.85)
            self.ax_spec.scatter(pts_t, pts_f1, s=10, alpha=0.8)
            self.ax_spec.scatter(pts_t, pts_f2, s=10, alpha=0.8)

            min_px = int(self.var_label_density.get())
            add_non_overlapping_labels(self.ax_wave, self.fig, pts_t, ya, pt_labels, dy=0.07, min_px=min_px, fontsize=8)
            add_non_overlapping_labels(self.ax_spec, self.fig, pts_t, pts_f2, pt_labels, dy=80, min_px=min_px, fontsize=8)

        cores = extract_simple_cores(
            t,
            f1,
            f2,
            accepted_mask,
            segments,
            boundary_timing_priors=self.boundary_timing_priors,
            seed_mask=seed_mask,
            boundary_mask=boundary_mask,
            stable_mask=stable_mask,
            frame_scores=frame_scores,
            settings=detection_settings,
        )
        self.cores = cores
        for c in cores:
            t0, t1 = c["t0"], c["t1"]
            self.ax_wave.axvspan(t0, t1, alpha=0.10)
            self.ax_spec.axvspan(t0, t1, alpha=0.10)
            self.ax_spec.text(
                c["tc"], c["med_f2"] + 80, f"[{c['best']}] {c['alts']}",
                fontsize=9, bbox=LABEL_BBOX, color="black"
            )

        self.draw_vowel_space_reference()
        if len(pts_t) > 0:
            facecolors = np.tile(self.vowel_base_face_rgba, (len(pts_t), 1))
            edgecolors = np.tile(self.vowel_base_edge_rgba, (len(pts_t), 1))
            self.vowel_points_artist = self.ax_vowel.scatter(
                pts_f2, pts_f1,
                s=self.vowel_point_size,
                marker="o",
                facecolors=facecolors,
                edgecolors=edgecolors
            )
            self._apply_vowel_chart_filter(None)

        self._install_span_selectors()
        self._ensure_playback_artists()
        self.ax_wave.tick_params(axis="x", labelbottom=False)
        if self.sr is not None and self.x is not None:
            if preserved_time_window is not None:
                self._sync_time_axes(preserved_time_window[0], preserved_time_window[1])
            else:
                self._sync_time_axes()
            self.force_full_time_view = False

        calibrated_cores = sum(1 for c in cores if c.get("source") == "stable_run_calibrated")
        self.status.config(
            text=(
                f"Praat frames: {len(t)}. Likely-vowel frames: {len(pts_t)}. Auto cores: {len(cores)}.\n"
                f"Auto-core timing: {calibrated_cores} calibrated with gold-set priors, {len(cores) - calibrated_cores} fallback stable runs.\n"
                f"Display mode: {'only likely vowels' if show_only else 'all formant frames'}.\n"
                f"Drag-select on waveform or spectrogram to label your chosen vowel region."
            )
        )

        self.canvas.draw()

    def _clear_selection_artists(self):
        if self.sel_wave_span is not None:
            try:
                self.sel_wave_span.remove()
            except Exception:
                pass
            self.sel_wave_span = None

        if self.sel_spec_span is not None:
            try:
                self.sel_spec_span.remove()
            except Exception:
                pass
            self.sel_spec_span = None

        if self.sel_vowel_marker is not None:
            try:
                self.sel_vowel_marker.remove()
            except Exception:
                pass
            self.sel_vowel_marker = None

        if self.sel_vowel_points_artist is not None:
            try:
                self.sel_vowel_points_artist.remove()
            except Exception:
                pass
            self.sel_vowel_points_artist = None

        if self.sel_spec_points_f1 is not None:
            try:
                self.sel_spec_points_f1.remove()
            except Exception:
                pass
            self.sel_spec_points_f1 = None

        if self.sel_spec_points_f2 is not None:
            try:
                self.sel_spec_points_f2.remove()
            except Exception:
                pass
            self.sel_spec_points_f2 = None

        if self.sel_text_artist is not None:
            try:
                self.sel_text_artist.remove()
            except Exception:
                pass
            self.sel_text_artist = None

    def _apply_vowel_chart_filter(self, mask=None):
        if self.vowel_points_artist is None or self.points is None:
            return

        n = len(self.points["t"])
        if n == 0:
            return

        facecolors = np.tile(self.vowel_base_face_rgba, (n, 1))
        edgecolors = np.tile(self.vowel_base_edge_rgba, (n, 1))
        sizes = np.full(n, self.vowel_point_size, dtype=float)

        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if len(mask) == n:
                facecolors[~mask] = self.vowel_dim_face_rgba
                edgecolors[~mask] = self.vowel_dim_edge_rgba
                sizes[mask] = self.vowel_selected_size

        self.vowel_points_artist.set_facecolors(facecolors)
        self.vowel_points_artist.set_edgecolors(edgecolors)
        self.vowel_points_artist.set_sizes(sizes)

    def draw_vowel_space_reference(self):
        ax = self.ax_vowel
        ax.set_title("Vowel space (F1/F2)")
        ax.set_xlabel("F2 (Hz)  ← front ... back →")
        ax.set_ylabel("F1 (Hz)  ↑ close ... open ↓")
        ax.set_xlim(3000, 500)
        ax.set_ylim(1200, 200)
        ax.grid(True, alpha=0.35)

        for sym, (F1r, F2r) in VOWEL_REF_POINTS.items():
            ax.scatter([F2r], [F1r], s=22, alpha=0.55)
            ax.text(F2r, F1r-15, sym, fontsize=9, alpha=0.75, bbox=LABEL_BBOX, color="black")

    def _install_span_selectors(self):
        def on_select_wave(xmin, xmax):
            self._handle_selection("waveform", xmin, xmax)

        def on_select_spec(xmin, xmax):
            self._handle_selection("spectrogram", xmin, xmax)

        self.span_wave = SpanSelector(
            self.ax_wave, on_select_wave, "horizontal", useblit=True,
            props=dict(alpha=0.18), interactive=True
        )
        self.span_spec = SpanSelector(
            self.ax_spec, on_select_spec, "horizontal", useblit=True,
            props=dict(alpha=0.18), interactive=True
        )

    def _handle_selection(self, src, xmin, xmax):
        if self.formants is None or self.points is None or self.all_points is None:
            return
        if xmin == xmax:
            return
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        self.selected_time_range = (float(xmin), float(xmax))

        likely_t = self.points["t"]
        likely_mask = (likely_t >= xmin) & (likely_t <= xmax)
        likely_count = int(np.sum(likely_mask))

        use_points = self.points
        use_mask = likely_mask
        used_source = "likely-vowel"
        used_note = ""

        if likely_count < 2:
            all_t = self.all_points["t"]
            all_mask = (all_t >= xmin) & (all_t <= xmax)
            all_count = int(np.sum(all_mask))
            if all_count > 0:
                use_points = self.all_points
                use_mask = all_mask
                used_source = "all-formant fallback"
                used_note = f"{likely_count} likely-vowel point(s); using {all_count} full formant frame(s) instead."
            else:
                use_points = self.all_points
                use_mask = all_mask
                used_source = "all-formant fallback"
                used_note = "No Praat formant frames inside selection."

        self._clear_selection_artists()
        self.sel_wave_span = self.ax_wave.axvspan(xmin, xmax, alpha=0.20)
        self.sel_spec_span = self.ax_spec.axvspan(xmin, xmax, alpha=0.20)
        self._apply_vowel_chart_filter(likely_mask)

        selected_count = int(np.sum(use_mask))
        if selected_count == 0:
            self.status.config(
                text=(
                    f"Selection {xmin:.3f}-{xmax:.3f}s ({src}):\n"
                    f"{used_note or 'No usable formant frames inside selection.'}\n"
                    "Press Tab or click Play selection to hear it."
                )
            )
            self.canvas.draw_idle()
            return

        t_pts = use_points["t"]
        f1_pts = use_points["f1"]
        f2_pts = use_points["f2"]
        sel_t = t_pts[use_mask]
        sel_f1 = f1_pts[use_mask]
        sel_f2 = f2_pts[use_mask]

        self.sel_spec_points_f1 = self.ax_spec.scatter(sel_t, sel_f1, s=28, marker="o", facecolors="none", edgecolors="#00a6ff", linewidths=1.0, alpha=0.9)
        self.sel_spec_points_f2 = self.ax_spec.scatter(sel_t, sel_f2, s=28, marker="o", facecolors="none", edgecolors="#ff8c00", linewidths=1.0, alpha=0.9)
        self.sel_vowel_points_artist = self.ax_vowel.scatter(sel_f2, sel_f1, s=70, marker="o", facecolors="none", edgecolors="#cc6600", linewidths=1.2, alpha=0.9)

        med_f1 = float(np.nanmedian(sel_f1))
        med_f2 = float(np.nanmedian(sel_f2))
        cands = ipa_candidates(med_f1, med_f2)
        alts = format_alt_label(cands)
        best = best_guess_symbol(med_f1, med_f2, restrict_to=cands if cands else None)

        self.sel_vowel_marker = self.ax_vowel.scatter(
            [med_f2], [med_f1], s=260, marker="X", edgecolors="k"
        )
        self.sel_text_artist = self.ax_vowel.text(
            med_f2, med_f1 - 25, f"[{best}] {alts}",
            fontsize=11, bbox=LABEL_BBOX, color="black"
        )

        self.status.config(
            text=(
                f"Selection {xmin:.3f}-{xmax:.3f}s ({src}):\n"
                f"Source: {used_source} ({selected_count} frame{'s' if selected_count != 1 else ''})"
                + (f" | {used_note}" if used_note else "")
                + "\n"
                f"Median F1={med_f1:.1f} Hz, F2={med_f2:.1f} Hz\n"
                f"Best guess: [{best}]   alternatives: {alts}\n"
                "Press Tab or click Play selection to hear it."
            )
        )

        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
