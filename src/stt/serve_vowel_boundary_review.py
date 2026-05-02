#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import unicodedata
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from serve_gold_set_review import (
    default_ipa_metadata_path,
    maybe_convert_windows_to_wsl,
    maybe_convert_wsl_to_windows,
    normalize_for_search,
    normalize_space,
    parse_json_body,
    path_windows,
    read_jsonl,
    send_audio,
    send_file,
    send_json,
    send_text,
    write_jsonl,
)
from vowel_segment_tools import (
    DEFAULT_PRAAT_EXE,
    VOWEL_RANGES,
    VOWEL_REF_POINTS,
    clamp01,
    count_ipa_vowels,
    detect_vowel_segments_from_formant_rows,
    ensure_praat_script,
    frame_edges_from_centers,
    load_settings,
    parse_formant_tsv,
    read_audio_mono,
    run_praat_formants,
    stage_audio_to_wav,
    tapered_range_score,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


IPA_VOWELS = [
    "i",
    "y",
    "ɨ",
    "ʉ",
    "ɯ",
    "u",
    "ɪ",
    "ʏ",
    "ʊ",
    "e",
    "ø",
    "ɘ",
    "ɵ",
    "ɤ",
    "o",
    "ə",
    "ɛ",
    "œ",
    "ɜ",
    "ɞ",
    "ʌ",
    "ɔ",
    "æ",
    "ɐ",
    "a",
    "ɶ",
    "ɑ",
    "ɒ",
]
VOWEL_PATTERN = re.compile("|".join(sorted(map(re.escape, IPA_VOWELS), key=len, reverse=True)))
VOWEL_DISPLAY_ORDER = ["i", "ɪ", "e", "ɛ", "æ", "ɨ", "u", "o", "a"]


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def default_settings_path() -> Path | None:
    candidates = [
        Path(r"%PRAAT_SCRIPTS_DIR%\vowel_finder_settings.json"),
        Path("$PRAAT_SCRIPTS_DIR/vowel_finder_settings.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def default_confirmed_jsonls() -> list[str]:
    candidates: list[Path] = []
    for base in [
        Path(r"%USERPROFILE%\Downloads"),
        Path("$HOME/Downloads"),
        script_dir() / "gold-set-review-ak-isolation",
    ]:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob("gold-set-confirmed*.jsonl")))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def default_output_dir() -> Path:
    return script_dir() / "vowel-boundary-review-ak-single-vowel"


def default_calibration_jsonls() -> list[str]:
    candidates: list[Path] = []
    for base in [
        Path(r"%USERPROFILE%\Downloads"),
        Path("$HOME/Downloads"),
        default_output_dir(),
    ]:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob("vowel-boundary-annotations*.jsonl"), key=lambda path: path.stat().st_mtime))
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a local browser UI for hand-annotating vowel start/stop boundaries "
            "on confirmed A.K. gold-set recordings."
        )
    )
    parser.add_argument(
        "--confirmed-jsonl",
        nargs="+",
        default=default_confirmed_jsonls(),
        help="One or more confirmed gold-set JSONL files to merge.",
    )
    parser.add_argument("--ipa-metadata-path", default=str(default_ipa_metadata_path()))
    parser.add_argument("--output-dir", default=str(default_output_dir()))
    parser.add_argument("--settings-path", default=str(default_settings_path() or ""))
    parser.add_argument("--praat-exe", default=DEFAULT_PRAAT_EXE)
    parser.add_argument(
        "--calibration-annotation-jsonl",
        nargs="*",
        default=default_calibration_jsonls(),
        help="Optional annotated boundary JSONL files used to learn vowel-specific timing priors.",
    )
    parser.add_argument("--speaker-user-id", type=int, default=8)
    parser.add_argument("--recording-type", default="isolation")
    parser.add_argument(
        "--ignore-vowels",
        nargs="*",
        default=["ʊ"],
        help="IPA vowels to ignore when selecting boundary-review candidates.",
    )
    parser.add_argument(
        "--include-multivowel",
        action="store_true",
        help="Include confirmed items with multiple expected vowel tokens.",
    )
    parser.add_argument(
        "--per-vowel-cap",
        type=int,
        default=3,
        help="Maximum number of single-vowel items to include per vowel symbol.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Optional hard cap after candidate selection.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def extract_vowels(ipa_text: str, ignored_vowels: set[str] | None = None) -> list[str]:
    ignored_vowels = ignored_vowels or set()
    return [vowel for vowel in VOWEL_PATTERN.findall(normalize_space(ipa_text)) if vowel not in ignored_vowels]


def merge_confirmed_rows(paths: list[Path]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_recordings: set[int] = set()
    for path in paths:
        for row in read_jsonl(path):
            recording_id = int(row.get("recording_id") or 0)
            if not recording_id or recording_id in seen_recordings:
                continue
            seen_recordings.add(recording_id)
            merged.append(row)
    return merged


def compute_waveform_peaks(audio: Any, bins: int = 900) -> list[list[float]]:
    if len(audio) == 0:
        return []
    bins = max(50, min(int(bins), len(audio)))
    edges = [int(round(index * len(audio) / bins)) for index in range(bins + 1)]
    peaks: list[list[float]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            right = min(left + 1, len(audio))
        frame = audio[left:right]
        if len(frame) == 0:
            peaks.append([0.0, 0.0])
            continue
        peaks.append([float(frame.min()), float(frame.max())])
    return peaks


def parse_ignore_vowels(raw_values: list[str]) -> set[str]:
    ignored: set[str] = set()
    for raw in raw_values:
        for part in re.split(r"[\s,]+", str(raw or "").strip()):
            if part:
                ignored.add(part)
    return ignored


def smooth_track(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0 or window <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    padded = np.pad(np.asarray(values, dtype=np.float64), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_formant_stability_track(frame_rows: list[dict[str, Any]]) -> np.ndarray:
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


def compute_spectral_shape_tracks(
    audio: np.ndarray,
    sr: int,
    frame_rows: list[dict[str, Any]],
    window_s: float = 0.025,
    max_freq_hz: float = 5000.0,
) -> dict[str, list[float]]:
    if len(audio) == 0 or sr <= 0 or not frame_rows:
        return {"stripe": [], "flatness": [], "similarity": [], "peakiness": []}

    frame_length = max(256, int(2 ** round(math.log2(max(256, sr * window_s)))))
    half = frame_length // 2
    window = np.hanning(frame_length)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / float(sr))
    keep = freqs <= max_freq_hz
    spectra: list[np.ndarray] = []

    for frame in frame_rows:
        center = int(round(float(frame["time_s"]) * sr))
        left = center - half
        right = left + frame_length
        src_left = max(0, left)
        src_right = min(len(audio), right)
        padded = np.zeros(frame_length, dtype=np.float64)
        dest_left = src_left - left
        dest_right = dest_left + max(0, src_right - src_left)
        if src_right > src_left:
            padded[dest_left:dest_right] = audio[src_left:src_right].astype(np.float64)
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

    stripe = smooth_track((0.42 * flatness) + (0.28 * similarity) + (0.30 * peakiness), window=5)
    return {
        "stripe": [round(float(value), 4) for value in stripe],
        "flatness": [round(float(value), 4) for value in flatness],
        "similarity": [round(float(value), 4) for value in similarity],
        "peakiness": [round(float(value), 4) for value in peakiness],
    }


def rough_ipa_tokens(ipa_text: str) -> list[str]:
    text = normalize_space(ipa_text)
    if not text:
        return []
    tokens: list[str] = []
    for char in text:
        if char.isspace() or char in {"/", "|", ".", "ˈ", "ˌ", "ː", "ˑ", "̆", "‿"}:
            continue
        if unicodedata.category(char).startswith("M"):
            continue
        if char in {"ʰ", "ʷ", "ʲ", "˞", "ˠ", "ˤ"}:
            continue
        tokens.append(char)
    return tokens


def ipa_position_prior(ipa_text: str, expected_vowel: str, ignored_vowels: set[str]) -> tuple[float, float]:
    tokens = [token for token in rough_ipa_tokens(ipa_text) if token not in ignored_vowels]
    if not tokens:
        return 0.5, 0.22
    vowel_positions = [index for index, token in enumerate(tokens) if token == expected_vowel]
    if not vowel_positions:
        vowel_positions = [index for index, token in enumerate(tokens) if token in IPA_VOWELS and token not in ignored_vowels]
    if not vowel_positions:
        return 0.5, 0.22
    position = vowel_positions[0]
    center_fraction = (position + 0.5) / max(1, len(tokens))
    slot_fraction = 1.0 / max(1, len(tokens))
    width_fraction = max(0.10, min(0.24, (0.52 * slot_fraction) + 0.02))
    return float(center_fraction), float(width_fraction)


def expected_vowel_match_score(expected_vowel: str, f1_value: float, f2_value: float) -> float:
    if not (math.isfinite(f1_value) and math.isfinite(f2_value)):
        return 0.0
    ranges = VOWEL_RANGES.get(expected_vowel)
    if not ranges:
        return 0.0
    f1_score = tapered_range_score(float(f1_value), float(ranges["F1"][0]), float(ranges["F1"][1]), 120.0)
    f2_score = tapered_range_score(float(f2_value), float(ranges["F2"][0]), float(ranges["F2"][1]), 260.0)
    ref_f1, ref_f2 = VOWEL_REF_POINTS.get(expected_vowel, ((ranges["F1"][0] + ranges["F1"][1]) / 2.0, (ranges["F2"][0] + ranges["F2"][1]) / 2.0))
    distance = math.hypot(float(f1_value) - ref_f1, float(f2_value) - ref_f2)
    distance_score = clamp01(1.0 - (distance / 1200.0))
    return float((0.4 * f1_score) + (0.4 * f2_score) + (0.2 * distance_score))


def choose_best_detector_segment(expected_vowel: str, detector_segments: list[dict[str, Any]], center_fraction: float, duration_s: float) -> dict[str, Any] | None:
    if not detector_segments:
        return None
    best_segment = None
    best_score = None
    for segment in detector_segments:
        center_score = 0.0
        if duration_s > 0.0:
            center_position = float(segment.get("center_s") or 0.0) / duration_s
            center_score = clamp01(1.0 - (abs(center_position - center_fraction) / 0.35))
        formant_score = expected_vowel_match_score(
            expected_vowel,
            float(segment.get("median_f1") or math.nan),
            float(segment.get("median_f2") or math.nan),
        )
        peak_score = clamp01(float(segment.get("peak_score") or 0.0))
        score = (0.30 * center_score) + (0.35 * formant_score) + (0.35 * peak_score)
        if best_score is None or score > best_score:
            best_score = score
            best_segment = segment
    return best_segment


def suggest_single_vowel_segment(
    item: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    detector_segments: list[dict[str, Any]],
    duration_s: float,
    ignored_vowels: set[str],
    extra_tracks: dict[str, Any] | None = None,
    timing_prior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_vowel = item["expected_vowels"][0]
    center_fraction, width_fraction = ipa_position_prior(item["recording_ipa"], expected_vowel, ignored_vowels)
    if not frame_rows:
        center_time = center_fraction * max(duration_s, 0.001)
        half_width = max(0.04, min(0.16, width_fraction * max(duration_s, 0.3) * 0.65))
        return {
            "vowel": expected_vowel,
            "start_s": round(max(0.0, center_time - half_width), 4),
            "end_s": round(min(duration_s, center_time + half_width), 4),
            "source": "ipa_prior_only",
            "ipa_guess": expected_vowel,
        }

    times = np.asarray([float(frame["time_s"]) for frame in frame_rows], dtype=np.float64)
    frame_edges = frame_edges_from_centers(times)
    score_track = np.asarray([float(frame.get("score") or 0.0) for frame in frame_rows], dtype=np.float64)
    voicing_track = np.asarray([clamp01((float(frame.get("voicing") or 0.0) - 0.10) / 0.45) for frame in frame_rows], dtype=np.float64)
    db_track = np.asarray([float(frame.get("relative_db_score") or 0.0) for frame in frame_rows], dtype=np.float64)
    formant_track = np.asarray(
        [
            expected_vowel_match_score(expected_vowel, float(frame.get("F1") or math.nan), float(frame.get("F2") or math.nan))
            for frame in frame_rows
        ],
        dtype=np.float64,
    )
    stability_track = compute_formant_stability_track(frame_rows)
    stripe_track = np.asarray((extra_tracks or {}).get("stripe") or [], dtype=np.float64)
    if len(stripe_track) != len(frame_rows):
        stripe_track = np.zeros(len(frame_rows), dtype=np.float64)
    if duration_s > 0.0:
        normalized_times = times / duration_s
    else:
        normalized_times = np.full_like(times, 0.5)
    position_track = np.exp(-0.5 * np.square((normalized_times - center_fraction) / max(0.08, width_fraction * 0.9)))
    core_track = smooth_track(
        (0.30 * formant_track)
        + (0.18 * voicing_track)
        + (0.12 * score_track)
        + (0.12 * position_track)
        + (0.16 * stability_track)
        + (0.12 * stripe_track),
        window=5,
    )

    detector_segment = choose_best_detector_segment(expected_vowel, detector_segments, center_fraction, duration_s)
    detector_bonus = np.zeros(len(frame_rows), dtype=np.float64)
    if detector_segment:
        detector_bonus = np.asarray(
            [
                1.0 if float(detector_segment.get("start_s") or 0.0) <= time_value <= float(detector_segment.get("end_s") or 0.0) else 0.0
                for time_value in times
            ],
            dtype=np.float64,
        )

    heuristic_track = (
        (0.21 * score_track)
        + (0.18 * voicing_track)
        + (0.12 * db_track)
        + (0.21 * formant_track)
        + (0.08 * position_track)
        + (0.04 * detector_bonus)
        + (0.08 * stability_track)
        + (0.08 * stripe_track)
    )
    heuristic_track = smooth_track(heuristic_track, window=5)

    search_mask = np.ones(len(frame_rows), dtype=bool)
    if duration_s > 0.0:
        left_limit_s = max(0.0, (center_fraction - (1.25 * width_fraction)) * duration_s - 0.02)
        right_limit_s = min(duration_s, (center_fraction + (1.25 * width_fraction)) * duration_s + 0.02)
        search_mask = (times >= left_limit_s) & (times <= right_limit_s)
        if not np.any(search_mask):
            search_mask = np.ones(len(frame_rows), dtype=bool)
    peak_track = (0.65 * heuristic_track) + (0.35 * core_track)
    search_scores = np.where(search_mask, peak_track, -1.0)
    peak_index = int(np.argmax(search_scores))
    peak_value = float(heuristic_track[peak_index])
    peak_core_value = float(core_track[peak_index])

    boundary_threshold = max(0.22, peak_core_value * 0.67)
    support_threshold = max(0.20, peak_value * 0.58)
    formant_floor = max(0.16, float(formant_track[peak_index]) * 0.45)
    stripe_floor = max(0.14, float(stripe_track[peak_index]) * 0.48)
    stability_floor = max(0.14, float(stability_track[peak_index]) * 0.44)
    start_index = peak_index
    end_index = peak_index

    def support_score(index: int) -> float:
        return float(
            (0.34 * heuristic_track[index])
            + (0.24 * core_track[index])
            + (0.12 * voicing_track[index])
            + (0.10 * formant_track[index])
            + (0.10 * stability_track[index])
            + (0.10 * stripe_track[index])
        )

    while start_index > 0:
        candidate = start_index - 1
        if duration_s > 0.0 and not search_mask[candidate] and (times[candidate] < times[peak_index]):
            break
        if (
            core_track[candidate] >= boundary_threshold
            or (
                support_score(candidate) >= support_threshold
                and (formant_track[candidate] >= formant_floor or stripe_track[candidate] >= stripe_floor)
                and stability_track[candidate] >= stability_floor
                and voicing_track[candidate] >= 0.14
            )
        ):
            start_index = candidate
            continue
        break
    while end_index + 1 < len(frame_rows):
        candidate = end_index + 1
        if duration_s > 0.0 and not search_mask[candidate] and (times[candidate] > times[peak_index]):
            break
        if (
            core_track[candidate] >= boundary_threshold
            or (
                support_score(candidate) >= support_threshold
                and (formant_track[candidate] >= formant_floor or stripe_track[candidate] >= stripe_floor)
                and stability_track[candidate] >= stability_floor
                and voicing_track[candidate] >= 0.14
            )
        ):
            end_index = candidate
            continue
        break

    start_s = float(frame_edges[start_index])
    end_s = float(frame_edges[end_index + 1])
    center_time = float(times[peak_index])
    prior_half_width = max(0.05, min(0.18, width_fraction * max(duration_s, 0.35) * 0.95))
    prior_start_s = max(0.0, center_time - prior_half_width - 0.03)
    prior_end_s = min(duration_s, center_time + prior_half_width + 0.03)
    start_s = max(start_s, prior_start_s)
    end_s = min(end_s, prior_end_s)

    min_duration_s = max(0.045, min(0.14, width_fraction * max(duration_s, 0.35) * 0.9))
    max_duration_s = max(min_duration_s + 0.02, min(0.30, (prior_half_width * 2.0) * 0.88))
    if (end_s - start_s) < min_duration_s:
        half_width = min_duration_s / 2.0
        start_s = max(0.0, center_time - half_width)
        end_s = min(duration_s, center_time + half_width)
    elif (end_s - start_s) > max_duration_s:
        half_width = max_duration_s / 2.0
        start_s = max(start_s, center_time - half_width)
        end_s = min(end_s, center_time + half_width)

    if detector_segment:
        detector_start = float(detector_segment.get("start_s") or start_s)
        detector_end = float(detector_segment.get("end_s") or end_s)
        overlap = max(0.0, min(end_s, detector_end) - max(start_s, detector_start))
        if overlap > 0.0:
            start_s = (0.82 * start_s) + (0.18 * detector_start)
            end_s = (0.82 * end_s) + (0.18 * detector_end)

    source = "ipa_signal"
    if timing_prior:
        prior_left_s = max(0.0, float(timing_prior.get("peak_minus_start_s") or 0.0))
        prior_right_s = float(timing_prior.get("end_minus_peak_s") or 0.0)
        prior_duration_s = max(0.03, float(timing_prior.get("duration_s") or 0.03))
        prior_start_s = max(0.0, center_time - prior_left_s)
        prior_end_s = min(duration_s, center_time + prior_right_s)
        if prior_end_s <= prior_start_s:
            prior_end_s = min(duration_s, prior_start_s + prior_duration_s)
        start_s = (0.55 * start_s) + (0.45 * prior_start_s)
        end_s = (0.15 * end_s) + (0.85 * prior_end_s)
        max_duration_s = min(duration_s, max(prior_duration_s * 1.35, 0.04))
        if (end_s - start_s) > max_duration_s:
            end_s = min(duration_s, start_s + max_duration_s)
        source = "ipa_signal_calibrated"

    start_s = max(0.0, min(start_s, duration_s))
    end_s = max(start_s + 0.01, min(end_s, duration_s))
    return {
        "vowel": expected_vowel,
        "start_s": round(float(start_s), 4),
        "end_s": round(float(end_s), 4),
        "source": source,
        "ipa_guess": detector_segment.get("ipa_guess") if detector_segment else expected_vowel,
        "center_fraction": round(center_fraction, 4),
        "peak_time_s": round(float(times[peak_index]), 4),
    }


def resize_matrix_rows(matrix: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((target_rows, target_cols), dtype=np.float64)
    row_edges = np.linspace(0, matrix.shape[0], target_rows + 1, dtype=int)
    col_edges = np.linspace(0, matrix.shape[1], target_cols + 1, dtype=int)
    output = np.zeros((target_rows, target_cols), dtype=np.float64)
    for row_index in range(target_rows):
        row_slice = matrix[row_edges[row_index] : max(row_edges[row_index + 1], row_edges[row_index] + 1), :]
        for col_index in range(target_cols):
            block = row_slice[:, col_edges[col_index] : max(col_edges[col_index + 1], col_edges[col_index] + 1)]
            output[row_index, col_index] = float(np.mean(block)) if block.size else 0.0
    return output


def compute_spectrogram(audio: np.ndarray, sr: int, max_freq_hz: float = 5000.0, target_rows: int = 96, target_cols: int = 360) -> dict[str, Any]:
    if len(audio) == 0 or sr <= 0:
        return {"levels": [], "max_freq_hz": max_freq_hz}
    frame_length = max(256, int(2 ** round(math.log2(max(256, sr * 0.02)))))
    hop_length = max(64, int(round(sr * 0.005)))
    if len(audio) < frame_length:
        padded = np.pad(audio.astype(np.float64), (0, frame_length - len(audio)))
    else:
        padded = audio.astype(np.float64)
    window = np.hanning(frame_length)
    frames = []
    for start in range(0, len(padded) - frame_length + 1, hop_length):
        frames.append(padded[start : start + frame_length] * window)
    if not frames:
        frames = [np.pad(padded, (0, max(0, frame_length - len(padded))))[:frame_length] * window]
    spectra = np.abs(np.fft.rfft(np.asarray(frames, dtype=np.float64), axis=1)).T
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / float(sr))
    keep = freqs <= max_freq_hz
    spectra = spectra[keep, :]
    if spectra.size == 0:
        return {"levels": [], "max_freq_hz": max_freq_hz}
    spectrogram_db = 20.0 * np.log10(spectra + 1e-6)
    high = float(np.percentile(spectrogram_db, 99.5))
    low = float(max(high - 70.0, np.percentile(spectrogram_db, 15.0)))
    norm = np.clip((spectrogram_db - low) / max(1e-6, high - low), 0.0, 1.0)
    reduced = resize_matrix_rows(norm, target_rows, min(target_cols, norm.shape[1]))
    levels = np.rint(reduced * 255.0).astype(np.uint8)
    return {
        "levels": levels.tolist(),
        "max_freq_hz": float(max_freq_hz),
    }


def build_visual_tracks(
    frame_rows: list[dict[str, Any]],
    expected_vowel: str,
    ignored_vowels: set[str],
    recording_ipa: str,
    duration_s: float,
    extra_tracks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not frame_rows:
        return {
            "times_s": [],
            "score": [],
            "heuristic": [],
            "core": [],
            "stability": [],
            "stripe": [],
            "voicing": [],
            "db": [],
            "f1_hz": [],
            "f2_hz": [],
            "f3_hz": [],
            "center_fraction": 0.5,
            "width_fraction": 0.22,
        }
    center_fraction, width_fraction = ipa_position_prior(recording_ipa, expected_vowel, ignored_vowels)
    times = np.asarray([float(frame["time_s"]) for frame in frame_rows], dtype=np.float64)
    score = np.asarray([float(frame.get("score") or 0.0) for frame in frame_rows], dtype=np.float64)
    voicing = np.asarray([clamp01((float(frame.get("voicing") or 0.0) - 0.10) / 0.45) for frame in frame_rows], dtype=np.float64)
    db = np.asarray([float(frame.get("relative_db_score") or 0.0) for frame in frame_rows], dtype=np.float64)
    formant = np.asarray(
        [expected_vowel_match_score(expected_vowel, float(frame.get("F1") or math.nan), float(frame.get("F2") or math.nan)) for frame in frame_rows],
        dtype=np.float64,
    )
    stability = compute_formant_stability_track(frame_rows)
    stripe = np.asarray((extra_tracks or {}).get("stripe") or [], dtype=np.float64)
    if len(stripe) != len(frame_rows):
        stripe = np.zeros(len(frame_rows), dtype=np.float64)
    if duration_s > 0.0:
        position = np.exp(-0.5 * np.square(((times / duration_s) - center_fraction) / max(0.08, width_fraction * 0.9)))
    else:
        position = np.ones_like(times)
    core = smooth_track((0.30 * formant) + (0.18 * voicing) + (0.12 * score) + (0.12 * position) + (0.16 * stability) + (0.12 * stripe), window=5)
    heuristic = smooth_track((0.21 * score) + (0.18 * voicing) + (0.12 * db) + (0.21 * formant) + (0.08 * position) + (0.10 * stability) + (0.10 * stripe), window=5)
    return {
        "times_s": [round(float(value), 4) for value in times],
        "score": [round(float(value), 4) for value in score],
        "heuristic": [round(float(value), 4) for value in heuristic],
        "core": [round(float(value), 4) for value in core],
        "stability": [round(float(value), 4) for value in stability],
        "stripe": [round(float(value), 4) for value in stripe],
        "voicing": [round(float(value), 4) for value in voicing],
        "db": [round(float(value), 4) for value in db],
        "f1_hz": [None if not math.isfinite(float(frame.get("F1") or math.nan)) else round(float(frame["F1"]), 2) for frame in frame_rows],
        "f2_hz": [None if not math.isfinite(float(frame.get("F2") or math.nan)) else round(float(frame["F2"]), 2) for frame in frame_rows],
        "f3_hz": [None if not math.isfinite(float(frame.get("F3") or math.nan)) else round(float(frame["F3"]), 2) for frame in frame_rows],
        "center_fraction": round(center_fraction, 4),
        "width_fraction": round(width_fraction, 4),
    }


def sort_key_for_candidate(item: dict[str, Any]) -> tuple[Any, ...]:
    tier = str(item.get("candidate_tier") or "")
    audio_ext = str(item.get("audio_ext") or "")
    return (
        tier != "preferred",
        audio_ext != ".wav",
        len(str(item.get("recording_ipa") or "")),
        len(str(item.get("orthography_text") or "")),
        int(item.get("recording_id") or 0),
    )


class BoundaryReviewStore:
    def __init__(
        self,
        confirmed_paths: list[Path],
        calibration_annotation_paths: list[Path],
        ipa_metadata_path: Path,
        output_dir: Path,
        settings_path: Path | None,
        praat_exe: str,
        speaker_user_id: int,
        recording_type: str,
        ignored_vowels: set[str],
        include_multivowel: bool,
        per_vowel_cap: int,
        max_items: int,
    ):
        self.confirmed_paths = confirmed_paths
        self.calibration_annotation_paths = calibration_annotation_paths
        self.ipa_metadata_path = ipa_metadata_path
        self.output_dir = output_dir
        self.settings_path = settings_path
        self.settings = load_settings(settings_path)
        self.praat_exe = praat_exe
        self.speaker_user_id = int(speaker_user_id)
        self.recording_type = normalize_space(recording_type).lower()
        self.ignored_vowels = set(ignored_vowels)
        self.include_multivowel = bool(include_multivowel)
        self.per_vowel_cap = max(1, int(per_vowel_cap or 1))
        self.max_items = max(0, int(max_items or 0))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_path = script_dir() / "vowel_boundary_review.html"
        self.cache_dir = self.output_dir / ".analysis-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.praat_script_path = ensure_praat_script(self.cache_dir / "extract_formants_for_boundary_review.praat")
        self.merged_confirmed_path = self.output_dir / "gold-set-confirmed-merged.jsonl"
        self.candidates_path = self.output_dir / "vowel-boundary-candidates.jsonl"
        self.candidates_tsv_path = self.output_dir / "vowel-boundary-candidates.tsv"
        self.decisions_path = self.output_dir / "vowel-boundary-review-decisions.jsonl"
        self.annotations_path = self.output_dir / "vowel-boundary-annotations.jsonl"
        self.skipped_path = self.output_dir / "vowel-boundary-skipped.jsonl"
        self.summary_path = self.output_dir / "vowel-boundary-review-summary.json"
        self.timing_priors_path = self.output_dir / "vowel-boundary-timing-priors.json"

        self.confirmed_rows = merge_confirmed_rows(self.confirmed_paths)
        self.items = self._build_items()
        self.index_by_recording = {int(item["recording_id"]): idx for idx, item in enumerate(self.items)}
        self.decisions_by_recording = self._load_existing_decisions()
        self.analysis_cache: dict[int, dict[str, Any]] = {}
        self.boundary_timing_priors: dict[str, dict[str, Any]] = {}
        self._apply_decisions_to_items()
        self.boundary_timing_priors = self._build_boundary_timing_priors(self.calibration_annotation_paths)
        self.analysis_cache.clear()
        self._write_outputs()

    def _resolve_audio_reference(self, raw_path: str, rel_audio: str = "") -> Path | None:
        raw_path = str(raw_path or "").strip()
        rel_audio = str(rel_audio or "").strip()
        candidates: list[Path] = []
        if raw_path:
            candidates.append(Path(raw_path))
            converted = maybe_convert_windows_to_wsl(raw_path)
            if converted:
                candidates.append(Path(converted))
            converted = maybe_convert_wsl_to_windows(raw_path)
            if converted:
                candidates.append(Path(converted))
        if rel_audio:
            candidates.append(self.ipa_metadata_path.parent / rel_audio)
            converted = maybe_convert_windows_to_wsl(rel_audio)
            if converted:
                candidates.append(Path(converted))
            converted = maybe_convert_wsl_to_windows(rel_audio)
            if converted:
                candidates.append(Path(converted))

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                try:
                    return candidate.resolve()
                except OSError:
                    return candidate
        return None

    def _normalize_confirmed_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if int(row.get("speaker_user_id") or 0) != self.speaker_user_id:
            return None
        recording_type = normalize_space(str(row.get("recording_type") or "")).lower()
        if self.recording_type and self.recording_type not in recording_type:
            return None
        expected_vowels = extract_vowels(str(row.get("recording_ipa") or ""), ignored_vowels=self.ignored_vowels)
        if not expected_vowels:
            return None
        if not self.include_multivowel and len(expected_vowels) != 1:
            return None
        audio_path = self._resolve_audio_reference(
            raw_path=str(row.get("audio_posix") or ""),
            rel_audio=str(row.get("audio_relpath") or ""),
        )
        if audio_path is None:
            return None
        audio_ext = audio_path.suffix.lower()
        normalized = {
            "recording_id": int(row.get("recording_id") or 0),
            "word_id": int(row.get("word_id") or 0),
            "speaker_user_id": int(row.get("speaker_user_id") or 0),
            "speaker_name": str(row.get("speaker_name") or ""),
            "recording_type": str(row.get("recording_type") or ""),
            "word_title": str(row.get("word_title") or ""),
            "word_translation": str(row.get("word_translation") or ""),
            "orthography_text": str(row.get("orthography_text") or ""),
            "orthography_source": str(row.get("orthography_source") or ""),
            "recording_translation": str(row.get("recording_translation") or ""),
            "recording_ipa": normalize_space(str(row.get("recording_ipa") or "")),
            "audio_relpath": str(row.get("audio_relpath") or ""),
            "audio_posix": str(audio_path),
            "audio_filename": str(row.get("audio_filename") or audio_path.name),
            "audio_ext": audio_ext,
            "candidate_score": int(row.get("candidate_score") or 0),
            "candidate_tier": str(row.get("candidate_tier") or ""),
            "review_note": str(row.get("review_note") or ""),
            "review_saved_at": str(row.get("review_saved_at") or ""),
            "expected_vowels": expected_vowels,
            "expected_vowel_count": len(expected_vowels),
            "search_blob": normalize_for_search(
                " ".join(
                    [
                        str(row.get("word_title") or ""),
                        str(row.get("word_translation") or ""),
                        str(row.get("orthography_text") or ""),
                        str(row.get("recording_ipa") or ""),
                        " ".join(expected_vowels),
                    ]
                )
            ),
            "saved_decision": None,
        }
        return normalized

    def _build_items(self) -> list[dict[str, Any]]:
        normalized_rows = []
        for row in self.confirmed_rows:
            normalized = self._normalize_confirmed_row(row)
            if normalized:
                normalized_rows.append(normalized)

        if self.include_multivowel:
            items = sorted(normalized_rows, key=sort_key_for_candidate)
        else:
            by_vowel: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in normalized_rows:
                by_vowel[row["expected_vowels"][0]].append(row)
            items = []
            remaining_vowels = [
                vowel for vowel in VOWEL_DISPLAY_ORDER if vowel in by_vowel
            ] + sorted(vowel for vowel in by_vowel if vowel not in VOWEL_DISPLAY_ORDER)
            for vowel in remaining_vowels:
                candidates = sorted(by_vowel[vowel], key=sort_key_for_candidate)
                items.extend(candidates[: self.per_vowel_cap])

        if self.max_items:
            items = items[: self.max_items]
        return items

    def _load_existing_decisions(self) -> dict[int, dict[str, Any]]:
        if not self.decisions_path.is_file():
            return {}
        decisions: dict[int, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            recording_id = int(row.get("recording_id") or 0)
            if recording_id:
                decisions[recording_id] = row
        return decisions

    def _apply_decisions_to_items(self) -> None:
        for item in self.items:
            item["saved_decision"] = self.decisions_by_recording.get(int(item["recording_id"]))

    def _sorted_decisions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.items:
            decision = self.decisions_by_recording.get(int(item["recording_id"]))
            if decision:
                rows.append(decision)
        return rows

    def _annotated_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.items:
            decision = item.get("saved_decision") or {}
            if decision.get("status") != "annotated":
                continue
            rows.append(
                {
                    "recording_id": int(item["recording_id"]),
                    "word_id": int(item["word_id"]),
                    "speaker_user_id": int(item["speaker_user_id"]),
                    "speaker_name": item["speaker_name"],
                    "recording_type": item["recording_type"],
                    "word_title": item["word_title"],
                    "word_translation": item["word_translation"],
                    "orthography_text": item["orthography_text"],
                    "orthography_source": item["orthography_source"],
                    "recording_translation": item["recording_translation"],
                    "recording_ipa": item["recording_ipa"],
                    "expected_vowels": item["expected_vowels"],
                    "audio_relpath": item["audio_relpath"],
                    "audio_posix": item["audio_posix"],
                    "audio_filename": item["audio_filename"],
                    "segments": decision.get("segments") or [],
                    "note": str(decision.get("note") or ""),
                    "saved_at": str(decision.get("saved_at") or ""),
                }
            )
        return rows

    def _skipped_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in self.items:
            decision = item.get("saved_decision") or {}
            if decision.get("status") != "skipped":
                continue
            rows.append(
                {
                    "recording_id": int(item["recording_id"]),
                    "word_id": int(item["word_id"]),
                    "orthography_text": item["orthography_text"],
                    "recording_ipa": item["recording_ipa"],
                    "expected_vowels": item["expected_vowels"],
                    "audio_posix": item["audio_posix"],
                    "note": str(decision.get("note") or ""),
                    "saved_at": str(decision.get("saved_at") or ""),
                }
            )
        return rows

    def _write_candidates_tsv(self) -> None:
        with self.candidates_tsv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "recording_id",
                    "word_id",
                    "speaker_name",
                    "word_title",
                    "orthography_text",
                    "recording_ipa",
                    "expected_vowels",
                    "candidate_tier",
                    "audio_filename",
                    "audio_ext",
                    "saved_status",
                    "saved_note",
                ]
            )
            for item in self.items:
                decision = item.get("saved_decision") or {}
                writer.writerow(
                    [
                        item["recording_id"],
                        item["word_id"],
                        item["speaker_name"],
                        item["word_title"],
                        item["orthography_text"],
                        item["recording_ipa"],
                        " ".join(item["expected_vowels"]),
                        item["candidate_tier"],
                        item["audio_filename"],
                        item["audio_ext"],
                        decision.get("status", ""),
                        decision.get("note", ""),
                    ]
                )

    def _write_outputs(self) -> None:
        write_jsonl(self.merged_confirmed_path, self.confirmed_rows)
        write_jsonl(self.candidates_path, self.items)
        self._write_candidates_tsv()
        write_jsonl(self.decisions_path, self._sorted_decisions())
        write_jsonl(self.annotations_path, self._annotated_rows())
        write_jsonl(self.skipped_path, self._skipped_rows())

        annotated = self._annotated_rows()
        skipped = self._skipped_rows()
        by_vowel = Counter()
        for item in self.items:
            if len(item["expected_vowels"]) == 1:
                by_vowel[item["expected_vowels"][0]] += 1
        summary = {
            "confirmed_jsonl_paths": [path_windows(path) for path in self.confirmed_paths],
            "calibration_annotation_paths": [path_windows(path) for path in self.calibration_annotation_paths],
            "merged_confirmed_jsonl": path_windows(self.merged_confirmed_path),
            "candidates_path": path_windows(self.candidates_path),
            "candidates_tsv_path": path_windows(self.candidates_tsv_path),
            "annotations_path": path_windows(self.annotations_path),
            "skipped_path": path_windows(self.skipped_path),
            "decisions_path": path_windows(self.decisions_path),
            "timing_priors_path": path_windows(self.timing_priors_path),
            "candidate_items": len(self.items),
            "annotated_items": len(annotated),
            "skipped_items": len(skipped),
            "pending_items": len(self.items) - len(annotated) - len(skipped),
            "ignored_vowels": sorted(self.ignored_vowels),
            "include_multivowel": self.include_multivowel,
            "recording_type": self.recording_type,
            "per_vowel_cap": self.per_vowel_cap,
            "max_items": self.max_items,
            "single_vowel_distribution": dict(sorted(by_vowel.items())),
            "settings_path": path_windows(self.settings_path) if self.settings_path else "",
            "praat_exe": str(self.praat_exe),
            "timing_priors": self.boundary_timing_priors,
        }
        self.timing_priors_path.write_text(json.dumps(self.boundary_timing_priors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _build_boundary_timing_priors(self, paths: list[Path]) -> dict[str, dict[str, Any]]:
        latest_rows: dict[int, dict[str, Any]] = {}
        for path in paths:
            if not path.is_file():
                continue
            for row in read_jsonl(path):
                recording_id = int(row.get("recording_id") or 0)
                if recording_id:
                    latest_rows[recording_id] = row

        by_vowel: dict[str, list[dict[str, float]]] = defaultdict(list)
        for recording_id, row in latest_rows.items():
            if recording_id not in self.index_by_recording:
                continue
            expected_vowels = row.get("expected_vowels") or []
            segments = row.get("segments") or []
            if len(expected_vowels) != 1 or len(segments) != 1:
                continue
            detail = self.get_item_detail(recording_id)
            suggestion = (detail.get("suggested_segments") or [None])[0]
            if not suggestion or suggestion.get("peak_time_s") is None:
                continue
            peak_time_s = float(suggestion["peak_time_s"])
            hand_segment = segments[0]
            start_s = float(hand_segment.get("start_s"))
            end_s = float(hand_segment.get("end_s"))
            if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s):
                continue
            vowel = str(expected_vowels[0])
            by_vowel[vowel].append(
                {
                    "peak_minus_start_s": peak_time_s - start_s,
                    "end_minus_peak_s": end_s - peak_time_s,
                    "duration_s": end_s - start_s,
                }
            )

        priors: dict[str, dict[str, Any]] = {}
        for vowel, rows in by_vowel.items():
            if not rows:
                continue
            priors[vowel] = {
                "count": len(rows),
                "peak_minus_start_s": round(float(np.median([row["peak_minus_start_s"] for row in rows])), 4),
                "end_minus_peak_s": round(float(np.median([row["end_minus_peak_s"] for row in rows])), 4),
                "duration_s": round(float(np.median([row["duration_s"] for row in rows])), 4),
            }
        return dict(sorted(priors.items()))

    def client_state(self) -> dict[str, Any]:
        annotated = len(self._annotated_rows())
        skipped = len(self._skipped_rows())
        return {
            "candidate_items": len(self.items),
            "annotated_items": annotated,
            "skipped_items": skipped,
            "pending_items": len(self.items) - annotated - skipped,
            "paths": {
                "merged_confirmed_jsonl": path_windows(self.merged_confirmed_path),
                "candidates_path": path_windows(self.candidates_path),
                "annotations_path": path_windows(self.annotations_path),
                "skipped_path": path_windows(self.skipped_path),
                "decisions_path": path_windows(self.decisions_path),
                "timing_priors_path": path_windows(self.timing_priors_path),
                "summary_path": path_windows(self.summary_path),
                "settings_path": path_windows(self.settings_path) if self.settings_path else "",
            },
            "ignored_vowels": sorted(self.ignored_vowels),
            "timing_priors": self.boundary_timing_priors,
            "items": self.items,
        }

    def _suggest_segments(self, item: dict[str, Any], detector_segments: list[dict], duration_s: float) -> list[dict]:
        expected = list(item["expected_vowels"])
        if not expected:
            return []
        if len(expected) == 1:
            return [
                suggest_single_vowel_segment(
                    item=item,
                    frame_rows=self.analysis_cache.get(int(item["recording_id"]), {}).get("_frame_rows", []),
                    detector_segments=detector_segments,
                    duration_s=duration_s,
                    ignored_vowels=self.ignored_vowels,
                    extra_tracks=self.analysis_cache.get(int(item["recording_id"]), {}).get("_spectral_tracks"),
                    timing_prior=self.boundary_timing_priors.get(expected[0]),
                )
            ]

        chosen_segments = sorted(detector_segments, key=lambda row: float(row.get("start_s") or 0.0))[: len(expected)]
        output = []
        for index, vowel in enumerate(expected):
            if index < len(chosen_segments):
                chosen = chosen_segments[index]
                output.append(
                    {
                        "vowel": vowel,
                        "start_s": round(float(chosen["start_s"]), 4),
                        "end_s": round(float(chosen["end_s"]), 4),
                        "source": "detector",
                        "ipa_guess": chosen.get("ipa_guess"),
                    }
                )
            else:
                output.append({"vowel": vowel, "start_s": None, "end_s": None, "source": "blank"})
        return output

    def _analysis_paths(self, recording_id: int) -> tuple[Path, Path]:
        base_dir = self.cache_dir / str(recording_id)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / "audio.wav", base_dir / "formants.tsv"

    def get_item_detail(self, recording_id: int) -> dict[str, Any]:
        recording_id = int(recording_id)
        cached = self.analysis_cache.get(recording_id)
        if cached:
            return cached

        idx = self.index_by_recording[recording_id]
        item = self.items[idx]
        audio_path = self.audio_path_for_recording(recording_id)
        if audio_path is None:
            raise FileNotFoundError(f"Audio not found for recording {recording_id}")

        wav_path, tsv_path = self._analysis_paths(recording_id)
        stage_audio_to_wav(audio_path, wav_path)
        try:
            run_praat_formants(
                praat_exe=self.praat_exe,
                script_path=self.praat_script_path,
                wav_path=wav_path,
                tsv_path=tsv_path,
                settings=self.settings,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Praat executable not found: {self.praat_exe}") from exc
        audio, sr = read_audio_mono(wav_path)
        formant_rows = parse_formant_tsv(tsv_path)
        frame_rows, segments = detect_vowel_segments_from_formant_rows(formant_rows, audio, sr, self.settings)
        duration_s = len(audio) / float(sr) if sr else 0.0
        spectral_tracks = compute_spectral_shape_tracks(audio, sr, frame_rows)
        self.analysis_cache[recording_id] = {"_frame_rows": frame_rows, "_spectral_tracks": spectral_tracks}
        suggested_segments = self._suggest_segments(item, segments, duration_s)
        tracks = build_visual_tracks(
            frame_rows=frame_rows,
            expected_vowel=item["expected_vowels"][0] if len(item["expected_vowels"]) == 1 else item["expected_vowels"][0],
            ignored_vowels=self.ignored_vowels,
            recording_ipa=item["recording_ipa"],
            duration_s=duration_s,
            extra_tracks=spectral_tracks,
        )

        detail = {
            "recording_id": recording_id,
            "duration_s": round(float(duration_s), 4),
            "sample_rate": int(sr),
            "waveform_peaks": compute_waveform_peaks(audio),
            "spectrogram": compute_spectrogram(audio, sr),
            "tracks": tracks,
            "detector_segments": segments,
            "suggested_segments": suggested_segments,
            "expected_vowels": item["expected_vowels"],
        }
        self.analysis_cache[recording_id] = detail
        return detail

    def audio_path_for_recording(self, recording_id: int) -> Path | None:
        idx = self.index_by_recording.get(int(recording_id))
        if idx is None:
            return None
        item = self.items[idx]
        return self._resolve_audio_reference(
            raw_path=str(item.get("audio_posix") or ""),
            rel_audio=str(item.get("audio_relpath") or ""),
        )

    def set_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        recording_id = int(payload.get("recording_id") or 0)
        idx = self.index_by_recording[recording_id]
        item = self.items[idx]
        status = normalize_space(str(payload.get("status") or "")).lower()
        if status not in {"annotated", "skipped"}:
            raise ValueError("Status must be `annotated` or `skipped`.")

        note = normalize_space(str(payload.get("note") or ""))
        segments = payload.get("segments") or []
        cleaned_segments = []
        if status == "annotated":
            if not isinstance(segments, list) or len(segments) != len(item["expected_vowels"]):
                raise ValueError("Annotated items must include one segment per expected vowel.")
            for index, (segment, vowel) in enumerate(zip(segments, item["expected_vowels"]), start=1):
                start_s = float(segment.get("start_s"))
                end_s = float(segment.get("end_s"))
                if not math.isfinite(start_s) or not math.isfinite(end_s):
                    raise ValueError("Segment boundaries must be finite numbers.")
                if start_s < 0.0 or end_s <= start_s:
                    raise ValueError("Each segment must have 0 <= start < end.")
                cleaned_segments.append(
                    {
                        "segment_index": index,
                        "vowel": vowel,
                        "start_s": round(start_s, 4),
                        "end_s": round(end_s, 4),
                        "duration_s": round(end_s - start_s, 4),
                    }
                )

        decision = {
            "recording_id": recording_id,
            "word_id": int(item["word_id"]),
            "status": status,
            "note": note,
            "segments": cleaned_segments,
            "saved_at": normalize_space(str(payload.get("saved_at") or "")) or "",
        }
        if not decision["saved_at"]:
            from datetime import datetime, timezone

            decision["saved_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.decisions_by_recording[recording_id] = decision
        item["saved_decision"] = decision
        self._write_outputs()
        return {
            "ok": True,
            "decision": decision,
            "candidate_items": len(self.items),
            "annotated_items": len(self._annotated_rows()),
            "skipped_items": len(self._skipped_rows()),
        }

    def clear_decision(self, recording_id: int) -> dict[str, Any]:
        recording_id = int(recording_id)
        self.decisions_by_recording.pop(recording_id, None)
        self.items[self.index_by_recording[recording_id]]["saved_decision"] = None
        self._write_outputs()
        return {
            "ok": True,
            "candidate_items": len(self.items),
            "annotated_items": len(self._annotated_rows()),
            "skipped_items": len(self._skipped_rows()),
        }


def build_handler(store: BoundaryReviewStore):
    class BoundaryReviewHandler(BaseHTTPRequestHandler):
        server_version = "VowelBoundaryReview/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stdout.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path

            if route in ("/", "/index.html"):
                send_file(self, store.html_path, "text/html; charset=utf-8")
                return
            if route == "/api/bootstrap":
                send_json(self, store.client_state())
                return
            if route.startswith("/api/item/"):
                try:
                    recording_id = int(route.rsplit("/", 1)[-1])
                    payload = store.get_item_detail(recording_id)
                except (ValueError, KeyError):
                    send_json(self, {"ok": False, "error": "Invalid recording id."}, status=400)
                    return
                except FileNotFoundError as exc:
                    send_json(self, {"ok": False, "error": str(exc)}, status=404)
                    return
                except Exception as exc:
                    send_json(self, {"ok": False, "error": str(exc)}, status=500)
                    return
                send_json(self, {"ok": True, "detail": payload})
                return
            if route == "/api/download-annotations":
                data = store.annotations_path.read_bytes() if store.annotations_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{store.annotations_path.name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route == "/api/download-candidates":
                data = store.candidates_path.read_bytes() if store.candidates_path.exists() else b""
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{store.candidates_path.name}"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if route.startswith("/audio/"):
                try:
                    recording_id = int(route.rsplit("/", 1)[-1])
                except ValueError:
                    send_text(self, "Invalid recording id.", status=400)
                    return
                audio_path = store.audio_path_for_recording(recording_id)
                if audio_path is None:
                    send_text(self, "Audio not found.", status=404)
                    return
                send_audio(self, audio_path)
                return
            send_text(self, "Not found.", status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            route = parsed.path
            try:
                payload = parse_json_body(self)
            except json.JSONDecodeError:
                send_json(self, {"ok": False, "error": "Invalid JSON body."}, status=400)
                return

            if route == "/api/save-decision":
                try:
                    result = store.set_decision(payload)
                except (KeyError, ValueError) as exc:
                    send_json(self, {"ok": False, "error": str(exc)}, status=400)
                    return
                send_json(self, result)
                return

            if route == "/api/clear-decision":
                try:
                    recording_id = int(payload.get("recording_id"))
                except (TypeError, ValueError):
                    send_json(self, {"ok": False, "error": "Invalid recording id."}, status=400)
                    return
                send_json(self, store.clear_decision(recording_id))
                return

            send_json(self, {"ok": False, "error": "Not found."}, status=404)

    return BoundaryReviewHandler


def main() -> None:
    args = parse_args()
    confirmed_paths = [Path(value).expanduser().resolve() for value in args.confirmed_jsonl]
    calibration_annotation_paths = [Path(value).expanduser().resolve() for value in args.calibration_annotation_jsonl]
    ipa_metadata_path = Path(args.ipa_metadata_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    settings_path = Path(args.settings_path).expanduser().resolve() if args.settings_path else None

    if not confirmed_paths:
        raise SystemExit("No confirmed JSONL files provided.")
    for confirmed_path in confirmed_paths:
        if not confirmed_path.is_file():
            raise SystemExit(f"Confirmed JSONL not found: {confirmed_path}")
    if not ipa_metadata_path.is_file():
        raise SystemExit(f"IPA metadata JSONL not found: {ipa_metadata_path}")

    store = BoundaryReviewStore(
        confirmed_paths=confirmed_paths,
        calibration_annotation_paths=calibration_annotation_paths,
        ipa_metadata_path=ipa_metadata_path,
        output_dir=output_dir,
        settings_path=settings_path,
        praat_exe=str(args.praat_exe),
        speaker_user_id=int(args.speaker_user_id),
        recording_type=str(args.recording_type),
        ignored_vowels=parse_ignore_vowels(list(args.ignore_vowels)),
        include_multivowel=bool(args.include_multivowel),
        per_vowel_cap=int(args.per_vowel_cap or 1),
        max_items=int(args.max_items or 0),
    )
    handler = build_handler(store)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    url = f"http://{args.host}:{args.port}/"
    print(f"Vowel boundary review UI: {url}")
    for confirmed_path in confirmed_paths:
        print(f"Confirmed seed: {path_windows(confirmed_path)}")
    for calibration_path in calibration_annotation_paths:
        if calibration_path.is_file():
            print(f"Calibration annotations: {path_windows(calibration_path)}")
    print(f"Output dir: {path_windows(output_dir)}")
    print(f"Candidates: {len(store.items)}")
    print(f"Annotated: {len(store._annotated_rows())}")
    print(f"Skipped: {len(store._skipped_rows())}")

    if args.open_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping vowel boundary review server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
