#!/usr/bin/env python3
"""
Heuristic onset classifier for isolated word recordings.

Labels:
- glottal_stop
- epiglottal
- pharyngeal
- none

This is not a phonetic oracle. It is a rule-based acoustic guess.
It works best on isolated tokens with 50-150 ms of leading silence.
Epiglottal labels are especially uncertain and are confidence-capped.

Dependencies:
    pip install numpy scipy soundfile librosa
Optional:
    pip install praat-parselmouth

Usage:
    python laryngeal_onset_classifier.py path/to/file.wav
    python laryngeal_onset_classifier.py --json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import scipy.signal as sps
import librosa

try:
    import parselmouth  # type: ignore
except Exception:
    parselmouth = None

PRAAT_EXE_CANDIDATES = [
    r"C:\Program Files (x86)\Praat\Praat.exe",
    r"C:\Program Files\Praat\Praat.exe",
]


# -----------------------------
# Small helpers
# -----------------------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def clean_input_path(text: str) -> str:
    text = (text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
        text = text[1:-1].strip()
    return text


def find_praat_exe() -> Optional[str]:
    for candidate in PRAAT_EXE_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


# -----------------------------
# Data containers
# -----------------------------

@dataclass
class FrameVoicing:
    time_s: float
    voiced: bool
    f0_hz: Optional[float]
    periodicity: float
    rms: float


@dataclass
class AnalysisFeatures:
    onset_s: float
    onset20_s: float
    rise_time_ms: float
    voiced_ratio_20ms: float
    voiced_ratio_40ms: float
    mean_periodicity_20ms: float
    mean_periodicity_40ms: float
    f0_cv_20ms: float
    f0_cv_40ms: float
    micro_gap_ms: float
    low_ratio_early: float
    centroid_early_hz: float
    zcr_early: float
    flatness_early: float
    low_ratio_stable: float
    centroid_stable_hz: float
    zcr_stable: float
    formant_source: str
    onset_f1_hz: Optional[float]
    onset_f2_hz: Optional[float]
    stable_f1_hz: Optional[float]
    stable_f2_hz: Optional[float]


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    scores: Dict[str, float]
    features: AnalysisFeatures
    notes: List[str]


# -----------------------------
# Audio analysis
# -----------------------------

def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    try:
        y, sr = sf.read(str(path), always_2d=False)
    except RuntimeError as exc:
        raise SystemExit(f"Could not read audio file: {path}\n{exc}") from exc

    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        raise SystemExit("Audio file is empty.")

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak

    if sr >= 200:
        b, a = sps.butter(2, 50.0 / (sr / 2.0), btype="highpass")
        y = sps.filtfilt(b, a, y)

    return y, int(sr)


def detect_onset(y: np.ndarray, sr: int) -> Tuple[float, Dict[str, Any]]:
    frame = max(64, int(0.005 * sr))
    hop = max(1, int(0.001 * sr))
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=False)[0]
    times = np.arange(len(rms), dtype=np.float64) * hop / sr

    noise_frames = max(1, int(0.03 * sr / hop))
    noise_floor = float(np.median(rms[:noise_frames]))
    threshold = max(noise_floor * 4.0, float(np.percentile(rms, 95)) * 0.05, 1e-4)

    onset_idx: Optional[int] = None
    for i in range(max(0, len(rms) - 4)):
        if rms[i] > threshold and int(np.sum(rms[i:i + 5] > threshold)) >= 3:
            onset_idx = i
            break
    if onset_idx is None:
        onset_idx = int(np.argmax(rms))

    coarse_onset = float(times[onset_idx])

    env = np.abs(sps.hilbert(y))
    if sr >= 120:
        b, a = sps.butter(2, 30.0 / (sr / 2.0), btype="low")
        env_s = sps.filtfilt(b, a, env)
    else:
        env_s = env

    start_i = min(len(env_s) - 1, max(0, int(coarse_onset * sr)))
    stop_i = min(len(env_s), start_i + int(0.12 * sr))
    local_max = float(np.max(env_s[start_i:stop_i])) if stop_i > start_i else float(np.max(env_s))
    back_threshold = max(noise_floor * 1.5, local_max * 0.03)

    refined_i = start_i
    while refined_i > 0 and env_s[refined_i - 1] > back_threshold:
        refined_i -= 1

    return refined_i / sr, {
        "rms": rms,
        "times": times,
        "noise_floor": noise_floor,
        "threshold": threshold,
        "env_s": env_s,
    }


def autocorr_voicing_track(
    y: np.ndarray,
    sr: int,
    start_s: float,
    duration_s: float = 0.08,
    frame_ms: float = 15.0,
    hop_ms: float = 2.0,
    f0_min: float = 70.0,
    f0_max: float = 400.0,
) -> List[FrameVoicing]:
    start_i = max(0, int(start_s * sr))
    end_i = min(len(y), start_i + int(duration_s * sr))
    seg = y[start_i:end_i]

    frame = max(64, int(frame_ms * sr / 1000.0))
    hop = max(1, int(hop_ms * sr / 1000.0))
    lag_min = max(1, int(sr / f0_max))
    lag_max = min(frame - 2, int(sr / f0_min))
    if len(seg) < frame:
        return []

    seg_rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    out: List[FrameVoicing] = []

    for i in range(0, len(seg) - frame + 1, hop):
        fr = seg[i:i + frame]
        fr = fr - np.mean(fr)
        rms = float(np.sqrt(np.mean(fr ** 2)))
        win = fr * np.hamming(len(fr))
        ac = sps.correlate(win, win, mode="full")[len(win) - 1:]
        if ac[0] <= 1e-12 or rms < 0.1 * seg_rms:
            out.append(FrameVoicing(
                time_s=(start_i + i + frame / 2.0) / sr,
                voiced=False,
                f0_hz=None,
                periodicity=0.0,
                rms=rms,
            ))
            continue

        nac = ac / ac[0]
        region = nac[lag_min:lag_max]
        if region.size == 0:
            out.append(FrameVoicing(
                time_s=(start_i + i + frame / 2.0) / sr,
                voiced=False,
                f0_hz=None,
                periodicity=0.0,
                rms=rms,
            ))
            continue

        peak_lag = int(np.argmax(region)) + lag_min
        periodicity = float(nac[peak_lag])
        voiced = periodicity > 0.35
        f0_hz = float(sr / peak_lag) if voiced else None
        out.append(FrameVoicing(
            time_s=(start_i + i + frame / 2.0) / sr,
            voiced=voiced,
            f0_hz=f0_hz,
            periodicity=periodicity,
            rms=rms,
        ))

    return out


def _window_voicing_stats(track: List[FrameVoicing], onset_s: float, window_ms: float) -> Tuple[float, float, float]:
    rows = [r for r in track if r.time_s <= onset_s + window_ms / 1000.0]
    if not rows:
        return 0.0, 0.0, 1.0

    voiced_ratio = float(np.mean([r.voiced for r in rows]))
    mean_periodicity = float(np.mean([r.periodicity for r in rows]))
    f0s = [r.f0_hz for r in rows if r.voiced and r.f0_hz is not None]
    if len(f0s) >= 2:
        f0_cv = float(np.std(f0s) / max(np.mean(f0s), 1e-9))
    else:
        f0_cv = 1.0
    return voiced_ratio, mean_periodicity, f0_cv


def spectral_stats(y: np.ndarray, sr: int, start_s: float, duration_s: float) -> Dict[str, float]:
    start_i = max(0, int(start_s * sr))
    end_i = min(len(y), start_i + int(duration_s * sr))
    seg = y[start_i:end_i]
    if len(seg) < 128:
        return {
            "centroid_hz": 0.0,
            "low_ratio": 0.0,
            "zcr": 0.0,
            "flatness": 0.0,
        }

    win = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * win)) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
    total = float(np.sum(spectrum) + 1e-12)

    centroid_hz = float(np.sum(freqs * spectrum) / total)
    low_ratio = float(np.sum(spectrum[(freqs >= 0) & (freqs < 1500)]) / total)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(seg, frame_length=len(seg), hop_length=len(seg), center=False)))

    n_fft = min(2048, 2 ** int(np.floor(np.log2(len(seg)))))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=seg, n_fft=n_fft, hop_length=len(seg), center=False)))

    return {
        "centroid_hz": centroid_hz,
        "low_ratio": low_ratio,
        "zcr": zcr,
        "flatness": flatness,
    }


# -----------------------------
# Praat extraction
# -----------------------------

def praat_script_text() -> str:
    return r'''
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


def run_praat_formants(audio_path: Path, praat_exe: str,
                       time_step: float = 0.005,
                       max_formants: float = 5.0,
                       ceiling_hz: float = 5500.0,
                       window_length: float = 0.025,
                       pre_emphasis_hz: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tmp_dir = tempfile.mkdtemp(prefix="praat_formants_")
    script_path = os.path.join(tmp_dir, "export_formants.praat")
    out_tsv = os.path.join(tmp_dir, "formants.tsv")

    with open(script_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(praat_script_text())

    cmd = [
        praat_exe,
        "--run",
        script_path,
        str(audio_path),
        out_tsv,
        f"{time_step:.6f}",
        f"{float(max_formants):.3f}",
        f"{float(ceiling_hz):.3f}",
        f"{window_length:.6f}",
        f"{pre_emphasis_hz:.3f}",
    ]

    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "check": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000

    subprocess.run(cmd, **kwargs)

    if not os.path.exists(out_tsv):
        raise RuntimeError("Praat ran, but no TSV output file was created.")

    times: List[float] = []
    f1s: List[float] = []
    f2s: List[float] = []
    f3s: List[float] = []
    f4s: List[float] = []
    with open(out_tsv, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue

            def parse_val(s: str) -> float:
                s = (s or "").strip()
                if "undefined" in s.lower() or s == "":
                    return float("nan")
                return float(s)

            times.append(parse_val(row[0]))
            f1s.append(parse_val(row[1]))
            f2s.append(parse_val(row[2]))
            f3s.append(parse_val(row[3]))
            f4s.append(parse_val(row[4]))

    if not times:
        raise RuntimeError("Praat returned an empty formant track.")

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(f1s, dtype=np.float64),
        np.asarray(f2s, dtype=np.float64),
        np.asarray(f3s, dtype=np.float64),
        np.asarray(f4s, dtype=np.float64),
    )


def _clean_formant_value(x: float) -> Optional[float]:
    if math.isnan(x) or x <= 0:
        return None
    return float(x)


def _nearest_defined_value(times: np.ndarray, values: np.ndarray, target_s: float, max_dist_s: float = 0.02) -> Optional[float]:
    if times.size == 0 or values.size == 0:
        return None
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        return None
    idxs = np.where(valid)[0]
    nearest = int(idxs[np.argmin(np.abs(times[idxs] - target_s))])
    if abs(float(times[nearest]) - float(target_s)) > max_dist_s:
        return None
    return _clean_formant_value(float(values[nearest]))


def maybe_extract_formants(path: Path, onset20_s: float, stable_s: float) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    if parselmouth is not None:
        try:
            snd = parselmouth.Sound(str(path))
            formant = snd.to_formant_burg(
                time_step=0.005,
                max_number_of_formants=5.0,
                maximum_formant=5500.0,
                window_length=0.025,
                pre_emphasis_from=50.0,
            )
            onset_f1 = _clean_formant_value(float(formant.get_value_at_time(1, onset20_s)))
            onset_f2 = _clean_formant_value(float(formant.get_value_at_time(2, onset20_s)))
            stable_f1 = _clean_formant_value(float(formant.get_value_at_time(1, stable_s)))
            stable_f2 = _clean_formant_value(float(formant.get_value_at_time(2, stable_s)))
            if any(v is not None for v in (onset_f1, onset_f2, stable_f1, stable_f2)):
                return "parselmouth", onset_f1, onset_f2, stable_f1, stable_f2
        except Exception:
            pass

    praat_exe = find_praat_exe()
    if not praat_exe:
        return "none", None, None, None, None

    try:
        times, f1s, f2s, _f3s, _f4s = run_praat_formants(path, praat_exe=praat_exe)
        onset_f1 = _nearest_defined_value(times, f1s, onset20_s)
        onset_f2 = _nearest_defined_value(times, f2s, onset20_s)
        stable_f1 = _nearest_defined_value(times, f1s, stable_s)
        stable_f2 = _nearest_defined_value(times, f2s, stable_s)
        if any(v is not None for v in (onset_f1, onset_f2, stable_f1, stable_f2)):
            return f"praat_cli:{praat_exe}", onset_f1, onset_f2, stable_f1, stable_f2
    except Exception:
        return "none", None, None, None, None

    return "none", None, None, None, None


# -----------------------------
# Feature extraction
# -----------------------------

def extract_features(path: Path, y: np.ndarray, sr: int) -> AnalysisFeatures:
    onset_s, aux = detect_onset(y, sr)
    env_s = aux["env_s"]

    start_i = max(0, int(onset_s * sr))
    end_i = min(len(y), start_i + int(0.12 * sr))
    onset_win = env_s[start_i:end_i]
    onset_max = float(np.max(onset_win) + 1e-12)

    above10 = np.where(onset_win >= 0.10 * onset_max)[0]
    above60 = np.where(onset_win >= 0.60 * onset_max)[0]
    t10 = float(above10[0] / sr) if len(above10) else 0.0
    t60 = float(above60[0] / sr) if len(above60) else 0.12
    rise_time_ms = max(0.0, (t60 - t10) * 1000.0)

    onset20_scan = env_s[start_i:min(len(y), start_i + int(0.04 * sr))]
    onset20_max = float(np.max(onset20_scan) + 1e-12)
    above20 = np.where(onset20_scan >= 0.20 * onset20_max)[0]
    onset20_s = onset_s + (float(above20[0] / sr) if len(above20) else 0.0)

    track = autocorr_voicing_track(y, sr, onset_s)
    voiced_ratio_20ms, mean_periodicity_20ms, f0_cv_20ms = _window_voicing_stats(track, onset_s, 20.0)
    voiced_ratio_40ms, mean_periodicity_40ms, f0_cv_40ms = _window_voicing_stats(track, onset_s, 40.0)

    early_env = env_s[start_i:min(len(y), start_i + int(0.04 * sr))]
    early_max = float(np.max(early_env) + 1e-12)
    micro_gap_ms = 0.0
    idx20 = np.where(early_env >= 0.20 * early_max)[0]
    if len(idx20):
        dipped = early_env[idx20[0]:] < 0.25 * early_max
        runs: List[Tuple[int, int]] = []
        run_start: Optional[int] = None
        for i, flag in enumerate(dipped):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                runs.append((run_start, i))
                run_start = None
        if run_start is not None:
            runs.append((run_start, len(dipped)))
        durs_ms = [((e - s) / sr) * 1000.0 for s, e in runs if 2.0 <= ((e - s) / sr) * 1000.0 <= 15.0]
        micro_gap_ms = float(max(durs_ms)) if durs_ms else 0.0

    early_spec = spectral_stats(y, sr, onset20_s, 0.03)
    stable_s = onset_s + 0.08
    if stable_s + 0.04 > len(y) / sr:
        stable_s = max(onset_s + 0.03, len(y) / sr - 0.05)
    stable_spec = spectral_stats(y, sr, stable_s, min(0.04, len(y) / sr - stable_s))

    formant_source, onset_f1_hz, onset_f2_hz, stable_f1_hz, stable_f2_hz = maybe_extract_formants(path, onset20_s, stable_s)

    return AnalysisFeatures(
        onset_s=float(onset_s),
        onset20_s=float(onset20_s),
        rise_time_ms=float(rise_time_ms),
        voiced_ratio_20ms=float(voiced_ratio_20ms),
        voiced_ratio_40ms=float(voiced_ratio_40ms),
        mean_periodicity_20ms=float(mean_periodicity_20ms),
        mean_periodicity_40ms=float(mean_periodicity_40ms),
        f0_cv_20ms=float(f0_cv_20ms),
        f0_cv_40ms=float(f0_cv_40ms),
        micro_gap_ms=float(micro_gap_ms),
        low_ratio_early=float(early_spec["low_ratio"]),
        centroid_early_hz=float(early_spec["centroid_hz"]),
        zcr_early=float(early_spec["zcr"]),
        flatness_early=float(early_spec["flatness"]),
        low_ratio_stable=float(stable_spec["low_ratio"]),
        centroid_stable_hz=float(stable_spec["centroid_hz"]),
        zcr_stable=float(stable_spec["zcr"]),
        formant_source=formant_source,
        onset_f1_hz=onset_f1_hz,
        onset_f2_hz=onset_f2_hz,
        stable_f1_hz=stable_f1_hz,
        stable_f2_hz=stable_f2_hz,
    )


# -----------------------------
# Scoring
# -----------------------------

def formant_pharyngeal_evidence(feat: AnalysisFeatures) -> float:
    if (
        feat.onset_f1_hz is None or
        feat.onset_f2_hz is None or
        feat.stable_f1_hz is None or
        feat.stable_f2_hz is None
    ):
        return 0.0

    f1_raise = feat.onset_f1_hz - feat.stable_f1_hz
    f2_lower = feat.stable_f2_hz - feat.onset_f2_hz
    evidence_f1 = clamp01((f1_raise - 100.0) / 250.0)
    evidence_f2 = clamp01((f2_lower - 120.0) / 300.0)
    return 0.5 * evidence_f1 + 0.5 * evidence_f2


def classify(feat: AnalysisFeatures) -> ClassificationResult:
    notes: List[str] = []

    abrupt = clamp01((25.0 - feat.rise_time_ms) / 20.0)
    gradual = clamp01((feat.rise_time_ms - 20.0) / 45.0)
    continuous = clamp01((feat.voiced_ratio_40ms - 0.45) / 0.35)
    early_voicing = clamp01((feat.voiced_ratio_20ms - 0.30) / 0.35)
    irregular = (
        0.6 * clamp01((feat.f0_cv_40ms - 0.05) / 0.15) +
        0.4 * clamp01((0.45 - feat.mean_periodicity_20ms) / 0.20)
    )
    brief_gap = (
        clamp01((feat.micro_gap_ms - 2.0) / 6.0) *
        clamp01((15.0 - feat.micro_gap_ms) / 8.0)
    )
    low_freq = clamp01((feat.low_ratio_early - 0.60) / 0.25)
    bursty = clamp01((feat.zcr_early - 0.20) / 0.15)
    formant_phary = formant_pharyngeal_evidence(feat)

    glottal_score = (
        0.55 * abrupt +
        0.20 * (1.0 - early_voicing) +
        0.15 * (1.0 - irregular) +
        0.10 * (1.0 - brief_gap)
    )

    pharyngeal_score = (
        0.34 * gradual +
        0.22 * continuous +
        0.12 * early_voicing +
        0.10 * low_freq +
        0.08 * (1.0 - bursty) +
        0.14 * formant_phary
    )

    epiglottal_score = (
        0.35 * brief_gap +
        0.35 * irregular +
        0.15 * continuous +
        0.15 * (1.0 - abrupt)
    )

    none_score = (
        0.30 * bursty +
        0.20 * (1.0 - continuous) +
        0.20 * (1.0 - gradual) +
        0.15 * (1.0 - abrupt) +
        0.15 * (1.0 - low_freq)
    )

    raw = {
        "glottal_stop": clamp01(glottal_score),
        "epiglottal": clamp01(epiglottal_score),
        "pharyngeal": clamp01(pharyngeal_score),
        "none": clamp01(none_score),
    }

    values = np.array(list(raw.values()), dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0:
        scores = {k: 0.25 for k in raw}
    else:
        scores = {k: float(v / total) for k, v in raw.items()}

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    label, top = ranked[0]
    second = ranked[1][1]
    margin = top - second

    signal_quality = 0.5 * feat.mean_periodicity_40ms + 0.5 * clamp01((feat.low_ratio_stable - 0.70) / 0.25)
    confidence = clamp01(0.22 + 0.45 * top + 0.40 * margin + 0.10 * signal_quality)
    confidence = min(confidence, 0.85)

    if label == "epiglottal":
        confidence = min(confidence, 0.55)
        notes.append("Epiglottal labels are confidence-capped. Acoustic separation from pharyngeal is weak without direct articulatory data.")

    if feat.formant_source == "none":
        confidence *= 0.95
        notes.append("Praat formants were not available. Pharyngeal confidence is reduced.")
    else:
        notes.append(f"Formant cue source: {feat.formant_source}.")

    if feat.rise_time_ms < 18.0:
        notes.append("Abrupt onset favors glottal stop.")
    elif feat.rise_time_ms > 30.0:
        notes.append("Gradual voiced onset favors pharyngeal or pharyngealized onset.")

    if feat.micro_gap_ms >= 3.0:
        notes.append("A brief post-onset dip was found. That can support an epiglottal analysis if the voicing is also irregular.")

    if feat.onset_s < 0.02:
        notes.append("The token starts almost immediately. Word-initial glottal closure is harder to infer if the file has been tightly trimmed.")

    if feat.voiced_ratio_40ms < 0.25:
        notes.append("Very weak early voicing. The onset may not be vowel-like. 'none' may be more trustworthy here.")

    return ClassificationResult(
        label=label,
        confidence=float(confidence),
        scores=scores,
        features=feat,
        notes=notes,
    )


# -----------------------------
# Presentation
# -----------------------------

def pretty_label(label: str) -> str:
    mapping = {
        "glottal_stop": "glottal stop",
        "epiglottal": "epiglottal",
        "pharyngeal": "pharyngeal",
        "none": "none of these",
    }
    return mapping.get(label, label.replace("_", " "))


def confidence_words(score: float) -> str:
    score = float(score)
    if score >= 0.85:
        return "very likely"
    if score >= 0.65:
        return "likely"
    if score >= 0.45:
        return "somewhat likely"
    if score >= 0.25:
        return "unlikely"
    return "very unlikely"


def result_to_jsonable(res: ClassificationResult) -> Dict[str, Any]:
    return {
        "label": res.label,
        "label_pretty": pretty_label(res.label),
        "confidence": round(res.confidence, 3),
        "confidence_text": confidence_words(res.confidence),
        "scores": {k: round(v, 3) for k, v in res.scores.items()},
        "score_text": {pretty_label(k): confidence_words(v) for k, v in res.scores.items()},
        "features": asdict(res.features),
        "notes": res.notes,
    }


def print_human(res: ClassificationResult, path: Path) -> None:
    print(f"File: {path}")
    print(f"Likeliest onset: {pretty_label(res.label)}")
    print(f"Overall confidence: {res.confidence:.2f} ({confidence_words(res.confidence)})")
    print("Interpretation:")
    if res.label == "none":
        print("  The script thinks none of these is the best fit.")
    else:
        print(f"  The script thinks a {pretty_label(res.label)} onset is the best fit.")
    print("Category likelihoods:")
    for key, value in sorted(res.scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {pretty_label(key):14s} {value:.3f}  {confidence_words(value)}")

    feat = res.features
    print("Key acoustic cues:")
    print(f"  onset_s              {feat.onset_s:.4f}")
    print(f"  rise_time_ms         {feat.rise_time_ms:.1f}")
    print(f"  voiced_ratio_20ms    {feat.voiced_ratio_20ms:.3f}")
    print(f"  voiced_ratio_40ms    {feat.voiced_ratio_40ms:.3f}")
    print(f"  mean_periodicity_40  {feat.mean_periodicity_40ms:.3f}")
    print(f"  micro_gap_ms         {feat.micro_gap_ms:.2f}")
    print(f"  low_ratio_early      {feat.low_ratio_early:.3f}")
    print(f"  zcr_early            {feat.zcr_early:.3f}")

    if feat.formant_source != "none":
        print("Formants:")
        print(f"  onset_f1_hz          {feat.onset_f1_hz}")
        print(f"  onset_f2_hz          {feat.onset_f2_hz}")
        print(f"  stable_f1_hz         {feat.stable_f1_hz}")
        print(f"  stable_f2_hz         {feat.stable_f2_hz}")

    if res.notes:
        print("Notes:")
        for note in res.notes:
            print(f"  - {note}")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heuristic word-initial glottal/epiglottal/pharyngeal onset classifier")
    parser.add_argument("audio_path", nargs="?", help="Path to an audio file")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    audio_path = clean_input_path(args.audio_path or "")
    if not audio_path:
        audio_path = clean_input_path(input("Audio file path: "))
    if not audio_path:
        print("No audio path provided.", file=sys.stderr)
        return 2

    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    y, sr = load_audio(path)
    features = extract_features(path, y, sr)
    result = classify(features)

    if args.json:
        print(json.dumps(result_to_jsonable(result), indent=2, ensure_ascii=False))
    else:
        print_human(result, path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
