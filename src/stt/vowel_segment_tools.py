#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _default_praat_exe() -> str:
    if os.name == "nt":
        return r"C:\Program Files (x86)\Praat\Praat.exe"
    return "/mnt/c/Program Files (x86)/Praat/Praat.exe"


DEFAULT_PRAAT_EXE = _default_praat_exe()
DEFAULT_SETTINGS = {
    "speaker_preset": "adult female",
    "praat_time_step": 0.0,
    "praat_max_formants": 5.0,
    "praat_ceiling_hz": 6500.0,
    "praat_window_length": 0.025,
    "praat_pre_emphasis_hz": 50.0,
    "voicing": 0.31,
    "reject_rhotics": 1,
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
    "min_segment_s": 0.03,
    "max_core_df1_hz_per_s": 900.0,
    "max_core_df2_hz_per_s": 1800.0,
    "min_core_s": 0.035,
}

PRAAT_SCRIPT_TEXT = """form ExportFormantsToTSV
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
"""

VOWEL_RANGES = {
    "i": {"F1": (180, 360), "F2": (2000, 3100)},
    "ɪ": {"F1": (250, 500), "F2": (1700, 2600)},
    "e": {"F1": (300, 650), "F2": (1700, 2500)},
    "ɛ": {"F1": (450, 850), "F2": (1500, 2400)},
    "æ": {"F1": (650, 1050), "F2": (1400, 2300)},
    "y": {"F1": (180, 380), "F2": (1500, 2600)},
    "ʉ": {"F1": (200, 450), "F2": (1200, 2000)},
    "ɨ": {"F1": (250, 450), "F2": (1300, 2000)},
    "ə": {"F1": (350, 650), "F2": (1200, 1800)},
    "ɜ": {"F1": (450, 800), "F2": (1100, 1600)},
    "u": {"F1": (200, 420), "F2": (600, 1200)},
    "ʊ": {"F1": (250, 500), "F2": (700, 1400)},
    "o": {"F1": (350, 650), "F2": (600, 1200)},
    "ɔ": {"F1": (450, 850), "F2": (600, 1300)},
    "ɑ": {"F1": (650, 1050), "F2": (800, 1500)},
    "a": {"F1": (700, 1100), "F2": (1100, 1700)},
    "ɐ": {"F1": (550, 900), "F2": (1100, 1700)},
}

VOWEL_REF_POINTS = {
    "i": (300, 2400),
    "ɪ": (400, 2000),
    "e": (450, 2200),
    "ɛ": (650, 1800),
    "æ": (850, 1700),
    "y": (320, 1900),
    "ʉ": (350, 1500),
    "ɨ": (350, 1600),
    "ə": (500, 1500),
    "ɜ": (650, 1300),
    "u": (350, 900),
    "ʊ": (450, 1100),
    "o": (500, 1000),
    "ɔ": (650, 900),
    "ɑ": (800, 1200),
    "a": (950, 1500),
    "ɐ": (700, 1500),
}

BROAD_VOWEL_REFS = {
    "i": (350.0, 2200.0),
    "e": (575.0, 1900.0),
    "central": (475.0, 1500.0),
    "o": (500.0, 1000.0),
    "a": (900.0, 1550.0),
}

IPA_TO_BROAD_VOWEL = {
    "i": "i",
    "ɪ": "i",
    "e": "e",
    "ɛ": "e",
    "æ": "a",
    "y": "i",
    "ʉ": "central",
    "ɨ": "central",
    "ə": "central",
    "ɜ": "central",
    "u": "o",
    "ʊ": "o",
    "o": "o",
    "ɔ": "o",
    "ɑ": "a",
    "a": "a",
    "ɐ": "central",
}

IPA_VOWEL_CHARS = set("aeiouyɨɪʊɛæɑɔəɜʉɐøœ")


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_settings(path: Path | None = None) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if path is None or not path.exists():
        return settings
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return settings
    for key, default_value in DEFAULT_SETTINGS.items():
        if key not in raw:
            continue
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            settings[key] = _coerce_int(raw[key], default_value)
        elif isinstance(default_value, float):
            settings[key] = _coerce_float(raw[key], default_value)
        else:
            settings[key] = raw[key]
    return settings


def read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    peak = float(np.max(np.abs(audio))) + 1e-9
    if peak > 1.5:
        audio = audio / peak
    return audio, int(sample_rate)


def ensure_praat_script(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PRAAT_SCRIPT_TEXT, encoding="utf-8")
    return path


def maybe_convert_wsl_to_windows(raw: str) -> str | None:
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(raw or ""))
    if not match:
        return None
    drive, rest = match.groups()
    return f"{drive.upper()}:\\{rest.replace('/', '\\')}"


def path_for_praat(path: str | Path) -> str:
    raw = str(path)
    if os.name == "nt":
        converted = maybe_convert_wsl_to_windows(raw)
        return converted or raw
    try:
        return subprocess.check_output(["wslpath", "-w", raw], text=True).strip()
    except Exception:
        return raw


def executable_for_host(path: str | Path) -> str:
    raw = str(path)
    if os.name == "nt":
        converted = maybe_convert_wsl_to_windows(raw)
        return converted or raw
    return raw


def stage_audio_to_wav(audio_path: Path, wav_path: Path) -> None:
    if wav_path.exists():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = read_audio_mono(audio_path)
    sf.write(str(wav_path), audio, sample_rate)


def run_praat_formants(
    praat_exe: str,
    script_path: Path,
    wav_path: Path,
    tsv_path: Path,
    settings: dict,
) -> None:
    if tsv_path.exists():
        return
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable_for_host(praat_exe),
        "--run",
        path_for_praat(script_path),
        path_for_praat(wav_path),
        path_for_praat(tsv_path),
        str(settings["praat_time_step"]),
        str(settings["praat_max_formants"]),
        str(settings["praat_ceiling_hz"]),
        str(settings["praat_window_length"]),
        str(settings["praat_pre_emphasis_hz"]),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def parse_formant_tsv(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            parsed = {"time_s": _coerce_float(row.get("time_s"), math.nan)}
            for key in ("F1", "F2", "F3", "F4"):
                value = row.get(key, "")
                if "undefined" in value.lower():
                    parsed[key] = math.nan
                else:
                    parsed[key] = _coerce_float(value, math.nan)
            if math.isfinite(parsed["time_s"]):
                rows.append(parsed)
    return rows


def rms_db(frame: np.ndarray) -> float:
    energy = float(np.sqrt(np.mean(np.square(frame.astype(np.float64)))) + 1e-12)
    return 20.0 * math.log10(energy + 1e-12)


def autocorr_voicing_strength(frame: np.ndarray, sr: int, f0_min: int = 70, f0_max: int = 400) -> float:
    frame = frame.astype(np.float64)
    frame = frame - np.mean(frame)
    frame = frame * np.hanning(len(frame))
    denom = np.sum(frame**2) + 1e-12
    if denom <= 1e-12:
        return 0.0
    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    autocorr = autocorr / (autocorr[0] + 1e-12)
    lag_min = int(sr / f0_max)
    lag_max = min(int(sr / f0_min), len(autocorr) - 1)
    if lag_max <= lag_min:
        return 0.0
    return float(np.max(autocorr[lag_min : lag_max + 1]))


def extract_frame(audio: np.ndarray, sr: int, center_s: float, window_s: float) -> np.ndarray:
    frame_size = max(256, int(round(sr * window_s)))
    center = int(round(center_s * sr))
    start = max(0, center - frame_size // 2)
    end = min(len(audio), center + frame_size // 2)
    return audio[start:end]


def tapered_range_score(value: float, lower: float, upper: float, taper: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if lower <= value <= upper:
        return 1.0
    if value < lower:
        return clamp01(1.0 - ((lower - value) / max(taper, 1e-6)))
    return clamp01(1.0 - ((value - upper) / max(taper, 1e-6)))


def smooth_binary_mask(mask: np.ndarray, times: np.ndarray, max_gap_s: float, min_run_s: float) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if len(out) == 0:
        return out

    index = 0
    total = len(out)
    while index < total:
        if out[index]:
            index += 1
            continue
        end = index
        while end < total and not out[end]:
            end += 1
        left = index - 1
        right = end
        if left >= 0 and right < total and out[left] and out[right]:
            if float(times[right] - times[left]) <= max_gap_s:
                out[index:end] = True
        index = end

    index = 0
    while index < total:
        if not out[index]:
            index += 1
            continue
        end = index
        while end < total and out[end]:
            end += 1
        duration = float(times[end - 1] - times[index]) if end - index > 1 else 0.0
        if duration < min_run_s:
            out[index:end] = False
        index = end

    return out


def mask_to_index_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    runs = []
    start = int(indices[0])
    previous = int(indices[0])
    for raw_index in indices[1:]:
        current = int(raw_index)
        if current == previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = current
        previous = current
    runs.append((start, previous))
    return runs


def frame_edges_from_centers(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.array([0.0], dtype=np.float64)
    if len(times) == 1:
        center = float(times[0])
        return np.array([max(0.0, center - 0.01), center + 0.01], dtype=np.float64)
    edges = np.empty(len(times) + 1, dtype=np.float64)
    midpoints = 0.5 * (times[:-1] + times[1:])
    edges[1:-1] = midpoints
    first_step = float(times[1] - times[0])
    last_step = float(times[-1] - times[-2])
    edges[0] = max(0.0, float(times[0]) - (0.5 * first_step))
    edges[-1] = float(times[-1]) + (0.5 * last_step)
    return edges


def nearest_broad_label(f1: float, f2: float) -> str:
    best_label = "central"
    best_distance = None
    for label, (ref_f1, ref_f2) in BROAD_VOWEL_REFS.items():
        distance = math.hypot(f1 - ref_f1, f2 - ref_f2)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_label = label
    return best_label


def ipa_candidates(f1: float, f2: float) -> list[str]:
    candidates = []
    for symbol, ranges in VOWEL_RANGES.items():
        if ranges["F1"][0] <= f1 <= ranges["F1"][1] and ranges["F2"][0] <= f2 <= ranges["F2"][1]:
            candidates.append(symbol)
    return candidates


def best_guess_symbol(f1: float, f2: float, restrict_to: list[str] | None = None) -> str:
    options = restrict_to if restrict_to else list(VOWEL_REF_POINTS)
    best_symbol = "?"
    best_distance = None
    for symbol in options:
        if symbol not in VOWEL_REF_POINTS:
            continue
        ref_f1, ref_f2 = VOWEL_REF_POINTS[symbol]
        distance = math.hypot(f1 - ref_f1, f2 - ref_f2)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_symbol = symbol
    return best_symbol


def broad_vowel_sequence(value: str) -> list[str]:
    text = unicodedata.normalize("NFC", value or "")
    output = []
    for char in text:
        mapped = IPA_TO_BROAD_VOWEL.get(char)
        if mapped:
            output.append(mapped)
    return output


def count_ipa_vowels(value: str) -> int:
    text = unicodedata.normalize("NFC", value or "")
    return sum(1 for char in text if char in IPA_VOWEL_CHARS)


def build_vowel_frame_rows(formant_rows: list[dict], audio: np.ndarray, sr: int, settings: dict) -> list[dict]:
    if not formant_rows:
        return []

    frame_rows = []
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

    for row in formant_rows:
        frame = extract_frame(audio, sr, row["time_s"], window_s)
        if len(frame) < 128:
            db_value = -120.0
            voicing_value = 0.0
        else:
            db_value = rms_db(frame)
            voicing_value = autocorr_voicing_strength(frame, sr)

        f1_value = float(row["F1"])
        f2_value = float(row["F2"])
        f3_value = float(row["F3"])

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
                "time_s": float(row["time_s"]),
                "F1": f1_value,
                "F2": f2_value,
                "F3": f3_value,
                "F4": float(row["F4"]),
                "db": db_value,
                "voicing": voicing_value,
                "f1_score": f1_score,
                "f2_score": f2_score,
                "separation_score": separation_score,
                "abs_db_score": abs_db_score,
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


def _merge_segments_if_needed(segments: list[dict], merge_gap_s: float) -> list[dict]:
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if (
            segment["broad_label"] == previous["broad_label"]
            and (segment["start_s"] - previous["end_s"]) <= merge_gap_s
        ):
            left_frames = previous["_frame_rows"]
            right_frames = segment["_frame_rows"]
            combined = left_frames + right_frames
            merged[-1] = build_segment_dict(
                previous["_frame_start"],
                segment["_frame_end"],
                previous["_core_start"],
                segment["_core_end"],
                previous["_frame_edges"],
                combined,
        )
            continue
        merged.append(segment)
    return merged


def build_segment_dict(
    frame_start: int,
    frame_end: int,
    core_start: int,
    core_end: int,
    frame_edges: np.ndarray,
    frame_rows: list[dict],
) -> dict:
    times = np.asarray([frame["time_s"] for frame in frame_rows], dtype=np.float64)
    f1_values = np.asarray([frame["F1"] for frame in frame_rows], dtype=np.float64)
    f2_values = np.asarray([frame["F2"] for frame in frame_rows], dtype=np.float64)
    f3_values = np.asarray([frame["F3"] for frame in frame_rows], dtype=np.float64)
    scores = np.asarray([frame["score"] for frame in frame_rows], dtype=np.float64)
    db_values = np.asarray([frame["db"] for frame in frame_rows], dtype=np.float64)
    voicing_values = np.asarray([frame["voicing"] for frame in frame_rows], dtype=np.float64)
    median_f1 = float(np.nanmedian(f1_values))
    median_f2 = float(np.nanmedian(f2_values))
    candidates = ipa_candidates(median_f1, median_f2)
    best = best_guess_symbol(median_f1, median_f2, restrict_to=candidates if candidates else None)
    broad_label = nearest_broad_label(median_f1, median_f2)
    start_s = float(frame_edges[frame_start])
    end_s = float(frame_edges[frame_end + 1])
    core_start_s = float(frame_edges[core_start])
    core_end_s = float(frame_edges[core_end + 1])
    return {
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": max(0.0, end_s - start_s),
        "core_start_s": core_start_s,
        "core_end_s": core_end_s,
        "center_s": float(np.nanmedian(times)),
        "frames": int(len(frame_rows)),
        "median_f1": median_f1,
        "median_f2": median_f2,
        "median_f3": float(np.nanmedian(f3_values)),
        "mean_score": float(np.nanmean(scores)),
        "peak_score": float(np.nanmax(scores)),
        "mean_db": float(np.nanmean(db_values)),
        "mean_voicing": float(np.nanmean(voicing_values)),
        "label": broad_label,
        "broad_label": broad_label,
        "ipa_guess": best,
        "ipa_candidates": candidates,
        "_frame_start": frame_start,
        "_frame_end": frame_end,
        "_core_start": core_start,
        "_core_end": core_end,
        "_frame_edges": frame_edges,
        "_frame_rows": frame_rows,
    }


def strip_internal_segment_fields(segments: list[dict]) -> list[dict]:
    cleaned = []
    for index, segment in enumerate(segments, start=1):
        row = {key: value for key, value in segment.items() if not key.startswith("_")}
        row["segment_index"] = index
        cleaned.append(row)
    return cleaned


def detect_vowel_segments_from_formant_rows(
    formant_rows: list[dict],
    audio: np.ndarray,
    sr: int,
    settings: dict,
) -> tuple[list[dict], list[dict]]:
    frame_rows = build_vowel_frame_rows(formant_rows, audio, sr, settings)
    if not frame_rows:
        return [], []

    times = np.asarray([frame["time_s"] for frame in frame_rows], dtype=np.float64)
    scores = np.asarray([frame["score"] for frame in frame_rows], dtype=np.float64)
    seed_mask = scores >= float(settings["segment_seed_score"])
    extend_mask = scores >= float(settings["segment_extend_score"])
    boundary_mask = scores >= float(settings["segment_boundary_score"])

    bridge_gap_s = float(settings["segment_bridge_gap_s"])
    extend_mask = smooth_binary_mask(extend_mask, times, max_gap_s=bridge_gap_s, min_run_s=0.0)
    boundary_mask = smooth_binary_mask(boundary_mask, times, max_gap_s=bridge_gap_s, min_run_s=0.0)
    frame_edges = frame_edges_from_centers(times)
    f1_values = np.asarray([frame["F1"] for frame in frame_rows], dtype=np.float64)
    f2_values = np.asarray([frame["F2"] for frame in frame_rows], dtype=np.float64)

    if len(times) >= 3:
        dt = np.gradient(times)
        dt[dt == 0.0] = np.nan
        df1 = np.gradient(f1_values) / dt
        df2 = np.gradient(f2_values) / dt
        stable_mask = (
            extend_mask
            & np.isfinite(df1)
            & np.isfinite(df2)
            & (np.abs(df1) <= float(settings["max_core_df1_hz_per_s"]))
            & (np.abs(df2) <= float(settings["max_core_df2_hz_per_s"]))
        )
        stable_mask = smooth_binary_mask(
            stable_mask,
            times,
            max_gap_s=min(bridge_gap_s, 0.015),
            min_run_s=float(settings["min_core_s"]),
        )
    else:
        stable_mask = seed_mask.copy()

    segments = []
    for start_index, end_index in mask_to_index_runs(extend_mask):
        if not np.any(seed_mask[start_index : end_index + 1]):
            continue
        core_runs = [
            (core_start, core_end)
            for core_start, core_end in mask_to_index_runs(stable_mask[start_index : end_index + 1])
            if np.any(seed_mask[start_index + core_start : start_index + core_end + 1])
        ]
        if not core_runs:
            core_runs = [(0, end_index - start_index)]

        for core_position, (core_start_rel, core_end_rel) in enumerate(core_runs):
            core_start = start_index + core_start_rel
            core_end = start_index + core_end_rel
            left_limit = start_index
            right_limit = end_index
            if core_position > 0:
                left_limit = start_index + core_runs[core_position - 1][1] + 1
            if core_position + 1 < len(core_runs):
                right_limit = start_index + core_runs[core_position + 1][0] - 1

            expanded_start = core_start
            expanded_end = core_end
            while expanded_start > left_limit and boundary_mask[expanded_start - 1]:
                expanded_start -= 1
            while expanded_end < right_limit and boundary_mask[expanded_end + 1]:
                expanded_end += 1

            segment = build_segment_dict(
                expanded_start,
                expanded_end,
                core_start,
                core_end,
                frame_edges,
                frame_rows[expanded_start : expanded_end + 1],
            )
            if segment["duration_s"] < float(settings["min_segment_s"]):
                continue
            segments.append(segment)

    segments = _merge_segments_if_needed(segments, float(settings["segment_merge_gap_s"]))
    return frame_rows, strip_internal_segment_fields(segments)


def detect_vowel_segments_for_file(
    audio_path: Path,
    settings: dict,
    praat_exe: str,
    script_path: Path,
    wav_path: Path,
    tsv_path: Path,
) -> dict:
    stage_audio_to_wav(audio_path, wav_path)
    run_praat_formants(praat_exe, script_path, wav_path, tsv_path, settings)
    audio, sr = read_audio_mono(wav_path)
    formant_rows = parse_formant_tsv(tsv_path)
    frame_rows, segments = detect_vowel_segments_from_formant_rows(formant_rows, audio, sr, settings)
    return {
        "sample_rate": sr,
        "formant_rows": len(formant_rows),
        "vowel_frames": len([row for row in frame_rows if row["score"] >= float(settings["segment_extend_score"])]),
        "segments": segments,
    }
