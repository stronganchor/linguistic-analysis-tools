#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from vowel_segment_tools import (
    DEFAULT_SETTINGS as VOWEL_SEGMENT_DEFAULT_SETTINGS,
    detect_vowel_segments_from_formant_rows,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_PRAAT_EXE = "/mnt/c/Program Files (x86)/Praat/Praat.exe"
DEFAULT_SETTINGS = dict(VOWEL_SEGMENT_DEFAULT_SETTINGS)

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

BROAD_VOWEL_REFS = {
    "i": (350.0, 2200.0),
    "e": (575.0, 1900.0),
    "central": (475.0, 1500.0),
    "o": (500.0, 1000.0),
    "a": (900.0, 1550.0),
}

BROAD_VOWEL_SCALES = {
    "i": (140.0, 420.0),
    "e": (180.0, 350.0),
    "central": (180.0, 260.0),
    "o": (180.0, 260.0),
    "a": (200.0, 260.0),
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

CANONICAL_REPLACEMENTS = [
    ("t̪͡ʃ", "C"),
    ("t͡ʃ", "C"),
    ("c͡ç", "C"),
    ("tʃ", "C"),
    ("d̪͡ʒ", "J"),
    ("d͡ʒ", "J"),
    ("ʒ", "J"),
    ("ʃ", "S"),
    ("j", "Y"),
    ("qʰʷ", "XW"),
    ("kʰʷ", "XW"),
    ("qʷ", "XW"),
    ("kʷ", "XW"),
    ("gʷ", "XW"),
    ("χʷ", "XW"),
    ("xʷ", "XW"),
    ("ħʷ", "XW"),
    ("qʰ", "X"),
    ("kʰ", "X"),
    ("q", "X"),
    ("χ", "X"),
    ("ħ", "X"),
    ("x", "X"),
    ("w", "W"),
    ("ɨ", "i"),
    ("ɪ", "i"),
    ("ɛ", "e"),
    ("æ", "a"),
    ("ɾ", "r"),
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Use Praat-backed acoustic analysis to review unresolved LL Tools IPA "
            "model disagreements and emit a conservative additional update JSONL."
        )
    )
    parser.add_argument(
        "--audit-path",
        default=str(
            root_dir
            / "zazacaogren"
            / "missing-ipa-non-cerez-ipa-review-april-2026"
            / "missing-non-cerez-resolution-audit.jsonl"
        ),
    )
    parser.add_argument(
        "--prior-update-path",
        default=str(
            root_dir
            / "zazacaogren"
            / "missing-ipa-non-cerez-ipa-review-april-2026"
            / "missing-non-cerez-recording_ipa-update.jsonl"
        ),
    )
    parser.add_argument(
        "--mms-path",
        default=str(
            root_dir
            / "zazacaogren"
            / "missing-ipa-non-cerez-model-runs-april-2026"
            / "mms-recording_ipa-missing-non-cerez.jsonl"
        ),
    )
    parser.add_argument(
        "--whisper-path",
        default=str(
            root_dir
            / "zazacaogren"
            / "missing-ipa-non-cerez-model-runs-april-2026"
            / "whisper-medium-recording_ipa-missing-non-cerez.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "zazacaogren" / "missing-ipa-non-cerez-acoustic-review-april-2026"),
    )
    parser.add_argument(
        "--praat-exe",
        default=DEFAULT_PRAAT_EXE,
        help="Path to Praat.exe on the Windows filesystem.",
    )
    parser.add_argument(
        "--vowel-finder-settings",
        default=str(
            Path("$PRAAT_SCRIPTS_DIR/vowel_finder_settings.json")
        ),
    )
    parser.add_argument(
        "--staging-dir",
        default=str(root_dir / ".praat-acoustic-review-cache"),
        help="Windows-accessible directory for staged WAV files and Praat TSVs.",
    )
    parser.add_argument("--jobs", type=int, default=2, help="Parallel Praat workers.")
    parser.add_argument(
        "--min-calibration-precision",
        type=float,
        default=0.9,
        help="Minimum in-sample precision required for accepting a threshold.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_settings(path: Path) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        for key, default_value in settings.items():
            if key not in loaded:
                continue
            if isinstance(default_value, int) and not isinstance(default_value, bool):
                settings[key] = int(loaded[key])
            elif isinstance(default_value, float):
                settings[key] = float(loaded[key])
            else:
                settings[key] = loaded[key]
    return settings


def ipa_to_canonical(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    for source, target in CANONICAL_REPLACEMENTS:
        text = text.replace(source, target)
    text = text.replace("-", " ")
    text = text.replace("’", "")
    text = text.replace("'", "")
    text = text.replace("`", "")
    text = text.replace("ˈ", "")
    text = text.replace("ˌ", "")
    text = text.replace("ː", "")
    text = text.replace("ʰ", "")
    text = text.replace("ʲ", "")
    text = text.replace("ʷ", "")
    text = text.replace("̪", "")
    text = text.replace("̥", "")
    text = text.replace("̞", "")
    text = text.replace("̆", "")
    text = text.replace("̃", "")
    text = text.replace("̯", "")
    text = text.replace("ⁿ", "")
    cleaned = []
    for char in text:
        if char.isascii() and (char.isalpha() or char == " "):
            cleaned.append(char)
        elif char == " ":
            cleaned.append(char)
    text = "".join(cleaned)
    return " ".join(text.split())


def consonant_skeleton(value: str) -> str:
    return "".join(char for char in value if char not in "aeiou ")


def broad_vowel_sequence(value: str) -> list[str]:
    text = unicodedata.normalize("NFC", value or "")
    sequence = []
    for char in text:
        mapped = IPA_TO_BROAD_VOWEL.get(char)
        if mapped:
            sequence.append(mapped)
    return sequence


def common_prefix_len(left: list[str], right: list[str]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def common_suffix_len(left: list[str], right: list[str], prefix_len: int) -> int:
    limit = min(len(left), len(right)) - prefix_len
    count = 0
    while count < limit and left[-1 - count] == right[-1 - count]:
        count += 1
    return count


def ensure_praat_script(path: Path) -> Path:
    path.write_text(PRAAT_SCRIPT_TEXT, encoding="utf-8")
    return path


def wsl_to_windows(path: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()


def stage_audio_to_wav(audio_path: Path, wav_path: Path) -> None:
    if wav_path.exists():
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(str(audio_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
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
    command = [
        praat_exe,
        "--run",
        wsl_to_windows(script_path),
        wsl_to_windows(wav_path),
        wsl_to_windows(tsv_path),
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
            try:
                rows.append(
                    {
                        "time_s": float(row["time_s"]),
                        "F1": float(row["F1"]),
                        "F2": float(row["F2"]),
                        "F3": float(row["F3"]),
                        "F4": float(row["F4"]),
                    }
                )
            except (ValueError, TypeError):
                continue
    return rows


def nearest_broad_label(f1: float, f2: float) -> tuple[str, float]:
    best_label = ""
    best_score = 999.0
    for label, (ref_f1, ref_f2) in BROAD_VOWEL_REFS.items():
        scale_f1, scale_f2 = BROAD_VOWEL_SCALES[label]
        score = math.hypot((f1 - ref_f1) / scale_f1, (f2 - ref_f2) / scale_f2)
        if score < best_score:
            best_score = score
            best_label = label
    return best_label, best_score


def derive_observed_segments(formant_rows: list[dict]) -> list[dict]:
    if not formant_rows:
        return []
    valid_frames = []
    for row in formant_rows:
        f1 = row["F1"]
        f2 = row["F2"]
        if not (180.0 <= f1 <= 1100.0 and 500.0 <= f2 <= 3200.0 and f2 > f1 + 120.0):
            continue
        label, label_score = nearest_broad_label(f1, f2)
        valid_frames.append(
            {
                "time_s": row["time_s"],
                "F1": f1,
                "F2": f2,
                "label": label,
                "label_score": label_score,
            }
        )
    if not valid_frames:
        return []

    smoothed = []
    for index, frame in enumerate(valid_frames):
        neighbors = valid_frames[max(0, index - 2) : min(len(valid_frames), index + 3)]
        counts = Counter(item["label"] for item in neighbors)
        top_count = max(counts.values())
        contenders = [label for label, count in counts.items() if count == top_count]
        label = frame["label"] if frame["label"] in contenders else sorted(contenders)[0]
        smoothed.append({**frame, "label": label})

    raw_segments = []
    current = [smoothed[0]]
    for frame in smoothed[1:]:
        last = current[-1]
        if frame["time_s"] - last["time_s"] <= 0.03 and frame["label"] == last["label"]:
            current.append(frame)
        else:
            raw_segments.append(current)
            current = [frame]
    raw_segments.append(current)

    filtered = []
    for segment in raw_segments:
        if len(segment) < 3:
            continue
        label = Counter(item["label"] for item in segment).most_common(1)[0][0]
        filtered.append(
            {
                "label": label,
                "start_s": segment[0]["time_s"],
                "end_s": segment[-1]["time_s"],
                "frames": len(segment),
                "median_f1": float(np.median([item["F1"] for item in segment])),
                "median_f2": float(np.median([item["F2"] for item in segment])),
            }
        )

    merged = []
    for segment in filtered:
        if (
            merged
            and segment["label"] == merged[-1]["label"]
            and segment["start_s"] - merged[-1]["end_s"] <= 0.04
        ):
            previous = merged.pop()
            merged_segment = {
                "label": previous["label"],
                "start_s": previous["start_s"],
                "end_s": segment["end_s"],
                "frames": previous["frames"] + segment["frames"],
                "median_f1": float(np.median([previous["median_f1"], segment["median_f1"]])),
                "median_f2": float(np.median([previous["median_f2"], segment["median_f2"]])),
            }
            merged.append(merged_segment)
        else:
            merged.append(segment)
    return merged


def candidate_segment_cost(segment: dict, target_label: str) -> float:
    ref_f1, ref_f2 = BROAD_VOWEL_REFS[target_label]
    scale_f1, scale_f2 = BROAD_VOWEL_SCALES[target_label]
    return math.hypot(
        (segment["median_f1"] - ref_f1) / scale_f1,
        (segment["median_f2"] - ref_f2) / scale_f2,
    )


def score_observation_against_candidate(observed: list[dict], expected: list[str]) -> float:
    if not expected:
        return 999.0
    if not observed:
        return 999.0

    insert_delete_penalty = 0.6
    rows = len(observed)
    cols = len(expected)
    dp = [[999999.0] * (cols + 1) for _ in range(rows + 1)]
    dp[0][0] = 0.0
    for index in range(1, rows + 1):
        dp[index][0] = dp[index - 1][0] + insert_delete_penalty
    for index in range(1, cols + 1):
        dp[0][index] = dp[0][index - 1] + insert_delete_penalty

    for obs_index in range(1, rows + 1):
        for exp_index in range(1, cols + 1):
            dp[obs_index][exp_index] = min(
                dp[obs_index - 1][exp_index] + insert_delete_penalty,
                dp[obs_index][exp_index - 1] + insert_delete_penalty,
                dp[obs_index - 1][exp_index - 1]
                + candidate_segment_cost(observed[obs_index - 1], expected[exp_index - 1]),
            )

    return dp[rows][cols] / max(rows, cols, 1)


def combined_candidate_score(
    observed: list[dict],
    left_sequence: list[str],
    right_sequence: list[str],
    candidate_sequence: list[str],
) -> tuple[float, dict]:
    full_score = score_observation_against_candidate(observed, candidate_sequence)
    prefix_len = common_prefix_len(left_sequence, right_sequence)
    suffix_len = common_suffix_len(left_sequence, right_sequence, prefix_len)
    details = {
        "full_score": round(full_score, 6),
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
    }

    total_len = len(candidate_sequence)
    diff_end_index = total_len - suffix_len
    local_score = None
    if suffix_len <= 1 and diff_end_index > 0:
        focus_expected = candidate_sequence[max(0, prefix_len - 1) :]
        focus_observed = observed[-max(3, 2 * len(focus_expected)) :]
        if focus_expected and focus_observed:
            local_score = score_observation_against_candidate(focus_observed, focus_expected)
            details["focus_type"] = "tail"
            details["focus_expected"] = focus_expected
    elif prefix_len <= 1:
        focus_expected = candidate_sequence[: min(total_len, max(2, diff_end_index + 1))]
        focus_observed = observed[: max(3, 2 * len(focus_expected))]
        if focus_expected and focus_observed:
            local_score = score_observation_against_candidate(focus_observed, focus_expected)
            details["focus_type"] = "head"
            details["focus_expected"] = focus_expected

    if local_score is None:
        details["combined_score"] = round(full_score, 6)
        return full_score, details

    combined = 0.65 * full_score + 0.35 * local_score
    details["local_score"] = round(local_score, 6)
    details["combined_score"] = round(combined, 6)
    return combined, details


def build_audio_map(mms_rows: list[dict], whisper_rows: list[dict]) -> dict[int, dict]:
    audio_map = {}
    for row in mms_rows + whisper_rows:
        recording_id = int(row["recording_id"])
        audio_map.setdefault(
            recording_id,
            {
                "audio_posix": row.get("audio_posix", ""),
                "audio_relpath": row.get("audio_relpath", ""),
                "source_bundle": row.get("source_bundle", ""),
            },
        )
    return audio_map


def choose_threshold(records: list[dict], min_precision: float) -> dict:
    candidate_thresholds = sorted(
        {round(record["margin"], 6) for record in records if record["pick_matches_truth"]},
        reverse=True,
    )
    best = {
        "threshold": 999.0,
        "coverage": 0,
        "precision": 0.0,
        "correct": 0,
        "picked": 0,
    }
    for threshold in candidate_thresholds:
        picked = [record for record in records if record["margin"] >= threshold]
        if not picked:
            continue
        correct = sum(1 for record in picked if record["pick_matches_truth"])
        precision = correct / len(picked)
        if precision < min_precision:
            continue
        if len(picked) > best["coverage"] or (
            len(picked) == best["coverage"] and precision > best["precision"]
        ):
            best = {
                "threshold": threshold,
                "coverage": len(picked),
                "precision": precision,
                "correct": correct,
                "picked": len(picked),
            }
    return best


def analyze_recording(
    recording_id: int,
    audio_posix: str,
    staging_dir: Path,
    praat_exe: str,
    script_path: Path,
    settings: dict,
) -> tuple[int, dict]:
    audio_path = Path(audio_posix)
    wav_path = staging_dir / "wav" / f"{recording_id}.wav"
    tsv_path = staging_dir / "formants" / f"{recording_id}.tsv"
    try:
        stage_audio_to_wav(audio_path, wav_path)
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        run_praat_formants(praat_exe, script_path, wav_path, tsv_path, settings)
        formants = parse_formant_tsv(tsv_path)
        audio, sample_rate = sf.read(str(wav_path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        frame_rows, observed = detect_vowel_segments_from_formant_rows(
            formants,
            audio.astype(np.float32),
            int(sample_rate),
            settings,
        )
        return recording_id, {
            "ok": True,
            "observed_segments": observed,
            "formant_rows": len(formants),
            "vowel_frame_rows": len(frame_rows),
        }
    except Exception as exc:  # noqa: BLE001
        return recording_id, {"ok": False, "error": str(exc)}


def main() -> int:
    args = parse_args()
    audit_path = Path(args.audit_path)
    prior_update_path = Path(args.prior_update_path)
    mms_path = Path(args.mms_path)
    whisper_path = Path(args.whisper_path)
    output_dir = Path(args.output_dir)
    staging_dir = Path(args.staging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings(Path(args.vowel_finder_settings))

    praat_exe = Path(args.praat_exe)
    if not praat_exe.exists():
        raise FileNotFoundError(f"Praat.exe not found at {praat_exe}")
    script_path = ensure_praat_script(staging_dir / "export_formants_batch.praat")

    audit_rows = load_jsonl(audit_path)
    prior_update_rows = load_jsonl(prior_update_path)
    mms_rows = load_jsonl(mms_path)
    whisper_rows = load_jsonl(whisper_path)
    audio_map = build_audio_map(mms_rows, whisper_rows)

    calibration_rows = []
    unresolved_rows = []
    for row in audit_rows:
        left_sequence = broad_vowel_sequence(row.get("mms_recording_ipa", ""))
        right_sequence = broad_vowel_sequence(row.get("whisper_medium_recording_ipa", ""))
        same_skeleton = consonant_skeleton(ipa_to_canonical(row.get("mms_recording_ipa", ""))) == consonant_skeleton(
            ipa_to_canonical(row.get("whisper_medium_recording_ipa", ""))
        )
        selected_model = row.get("selected_model", "")

        if (
            row.get("resolution_status") == "resolved"
            and selected_model in {"mms", "whisper_medium"}
            and row.get("resolution_type") != "high_conf_canonical_equivalence"
            and same_skeleton
        ):
            calibration_rows.append(row)

        if row.get("resolution_status") != "unresolved":
            continue
        if same_skeleton or row.get("resolution_type") == "unresolved_weak_support":
            unresolved_rows.append(row)

    needed_recording_ids = sorted(
        {
            int(row["recording_id"])
            for row in calibration_rows + unresolved_rows
            if int(row["recording_id"]) in audio_map
        }
    )
    print(
        json.dumps(
            {
                "calibration_rows": len(calibration_rows),
                "eligible_unresolved_rows": len(unresolved_rows),
                "needed_recordings": len(needed_recording_ids),
            },
            ensure_ascii=False,
        )
    )

    analysis_by_recording = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        future_map = {
            pool.submit(
                analyze_recording,
                recording_id,
                audio_map[recording_id]["audio_posix"],
                staging_dir,
                str(praat_exe),
                script_path,
                settings,
            ): recording_id
            for recording_id in needed_recording_ids
        }
        completed = 0
        for future in as_completed(future_map):
            recording_id, result = future.result()
            analysis_by_recording[recording_id] = result
            completed += 1
            if completed % 25 == 0 or completed == len(needed_recording_ids):
                print(
                    json.dumps(
                        {
                            "progress": completed,
                            "total": len(needed_recording_ids),
                            "current_recording_id": recording_id,
                            "ok": result.get("ok", False),
                        },
                        ensure_ascii=False,
                    )
                )

    calibration_scored = []
    for row in calibration_rows:
        recording_id = int(row["recording_id"])
        analysis = analysis_by_recording.get(recording_id, {})
        if not analysis.get("ok"):
            continue
        observed = analysis["observed_segments"]
        if len(observed) < 2:
            continue

        left_sequence = broad_vowel_sequence(row["mms_recording_ipa"])
        right_sequence = broad_vowel_sequence(row["whisper_medium_recording_ipa"])
        left_score, left_details = combined_candidate_score(observed, left_sequence, right_sequence, left_sequence)
        right_score, right_details = combined_candidate_score(observed, left_sequence, right_sequence, right_sequence)
        acoustic_pick = "mms" if left_score < right_score else "whisper_medium"
        calibration_scored.append(
            {
                "recording_id": recording_id,
                "word_title": row.get("word_title", ""),
                "selected_model": row["selected_model"],
                "acoustic_pick": acoustic_pick,
                "pick_matches_truth": acoustic_pick == row["selected_model"],
                "left_score": left_score,
                "right_score": right_score,
                "margin": abs(left_score - right_score),
                "observed_segment_count": len(observed),
                "left_details": left_details,
                "right_details": right_details,
            }
        )

    threshold_info = choose_threshold(calibration_scored, args.min_calibration_precision)
    weak_threshold = threshold_info["threshold"] * 0.8 if threshold_info["coverage"] else 999.0

    acoustic_audit_rows = []
    additional_update_rows = []
    newly_resolved = 0
    unresolved_after_acoustics = 0
    reason_counts = Counter()
    for row in unresolved_rows:
        recording_id = int(row["recording_id"])
        analysis = analysis_by_recording.get(recording_id, {})
        output_row = dict(row)
        output_row["acoustic_status"] = "not_scored"
        output_row["acoustic_selected_model"] = ""
        output_row["acoustic_margin"] = None
        output_row["acoustic_threshold_used"] = None
        output_row["acoustic_reason"] = ""
        output_row["acoustic_segment_count"] = 0
        output_row["acoustic_observed_segments"] = []

        if not analysis.get("ok"):
            output_row["acoustic_status"] = "analysis_error"
            output_row["acoustic_reason"] = analysis.get("error", "unknown")
            acoustic_audit_rows.append(output_row)
            reason_counts[output_row["acoustic_status"]] += 1
            unresolved_after_acoustics += 1
            continue

        observed = analysis["observed_segments"]
        output_row["acoustic_segment_count"] = len(observed)
        output_row["acoustic_observed_segments"] = observed
        if len(observed) < 2:
            output_row["acoustic_status"] = "too_few_segments"
            output_row["acoustic_reason"] = "Observed fewer than two usable vowel segments."
            acoustic_audit_rows.append(output_row)
            reason_counts[output_row["acoustic_status"]] += 1
            unresolved_after_acoustics += 1
            continue

        left_sequence = broad_vowel_sequence(row["mms_recording_ipa"])
        right_sequence = broad_vowel_sequence(row["whisper_medium_recording_ipa"])
        if not left_sequence or not right_sequence:
            output_row["acoustic_status"] = "no_vowel_sequence"
            output_row["acoustic_reason"] = "Could not derive broad vowel sequences from one or both candidates."
            acoustic_audit_rows.append(output_row)
            reason_counts[output_row["acoustic_status"]] += 1
            unresolved_after_acoustics += 1
            continue

        left_score, left_details = combined_candidate_score(observed, left_sequence, right_sequence, left_sequence)
        right_score, right_details = combined_candidate_score(observed, left_sequence, right_sequence, right_sequence)
        acoustic_pick = "mms" if left_score < right_score else "whisper_medium"
        margin = abs(left_score - right_score)
        output_row["acoustic_left_score"] = round(left_score, 6)
        output_row["acoustic_right_score"] = round(right_score, 6)
        output_row["acoustic_selected_model"] = acoustic_pick
        output_row["acoustic_margin"] = round(margin, 6)
        output_row["acoustic_left_details"] = left_details
        output_row["acoustic_right_details"] = right_details

        same_skeleton = consonant_skeleton(ipa_to_canonical(row["mms_recording_ipa"])) == consonant_skeleton(
            ipa_to_canonical(row["whisper_medium_recording_ipa"])
        )

        decision = None
        if same_skeleton and margin >= threshold_info["threshold"]:
            decision = "same_skeleton_margin"
            output_row["acoustic_threshold_used"] = round(threshold_info["threshold"], 6)
        elif row.get("resolution_type") == "unresolved_weak_support" and acoustic_pick == row.get("selected_model"):
            if margin >= weak_threshold:
                decision = "weak_support_confirmed_by_acoustics"
                output_row["acoustic_threshold_used"] = round(weak_threshold, 6)

        if decision is None:
            output_row["acoustic_status"] = "left_unresolved"
            output_row["acoustic_reason"] = "Acoustic margin did not clear the calibrated threshold."
            acoustic_audit_rows.append(output_row)
            reason_counts[output_row["acoustic_status"]] += 1
            unresolved_after_acoustics += 1
            continue

        chosen_ipa = row["mms_recording_ipa"] if acoustic_pick == "mms" else row["whisper_medium_recording_ipa"]
        output_row["acoustic_status"] = "resolved"
        output_row["acoustic_reason"] = decision
        output_row["chosen_recording_ipa"] = chosen_ipa
        acoustic_audit_rows.append(output_row)
        reason_counts[output_row["acoustic_status"]] += 1
        newly_resolved += 1

        additional_update_rows.append(
            {
                "word_id": row["word_id"],
                "recording_id": row["recording_id"],
                "recording_ipa": chosen_ipa,
                "selected_model": acoustic_pick,
                "resolution_type": f"acoustic_{decision}",
                "word_title": row.get("word_title", ""),
                "word_translation": row.get("word_translation", ""),
                "evidence_sources": [
                    "praat_formants",
                    "vowel_finder_broad_vowel_space",
                    acoustic_pick,
                ],
            }
        )

    prior_keys = {(int(row["word_id"]), int(row["recording_id"])) for row in prior_update_rows}
    merged_update_rows = list(prior_update_rows)
    for row in additional_update_rows:
        key = (int(row["word_id"]), int(row["recording_id"]))
        if key not in prior_keys:
            merged_update_rows.append(row)

    calibration_summary = {
        "rows_scored": len(calibration_scored),
        "correct_without_threshold": sum(1 for row in calibration_scored if row["pick_matches_truth"]),
        "raw_accuracy": (
            sum(1 for row in calibration_scored if row["pick_matches_truth"]) / len(calibration_scored)
            if calibration_scored
            else 0.0
        ),
        "same_skeleton_threshold": threshold_info,
        "weak_support_confirmation_threshold": weak_threshold if threshold_info["coverage"] else None,
    }

    summary = {
        "eligible_unresolved_rows": len(unresolved_rows),
        "newly_resolved_by_acoustics": newly_resolved,
        "left_unresolved_after_acoustics": unresolved_after_acoustics,
        "additional_update_rows": len(additional_update_rows),
        "merged_update_rows": len(merged_update_rows),
        "analysis_errors": reason_counts["analysis_error"],
        "too_few_segments": reason_counts["too_few_segments"],
        "reason_counts": dict(reason_counts),
        "calibration": calibration_summary,
    }

    acoustic_audit_path = output_dir / "missing-non-cerez-acoustic-audit.jsonl"
    additional_update_path = output_dir / "missing-non-cerez-recording_ipa-acoustic-update.jsonl"
    merged_update_path = output_dir / "missing-non-cerez-recording_ipa-merged-update.jsonl"
    calibration_path = output_dir / "missing-non-cerez-acoustic-calibration.jsonl"
    summary_path = output_dir / "missing-non-cerez-acoustic-summary.json"
    notes_path = output_dir / "missing-non-cerez-acoustic-notes.md"

    write_jsonl(acoustic_audit_path, acoustic_audit_rows)
    write_jsonl(additional_update_path, additional_update_rows)
    write_jsonl(merged_update_path, merged_update_rows)
    write_jsonl(calibration_path, calibration_scored)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    notes = [
        "# Acoustic Review Notes",
        "",
        "This pass uses Praat Burg formants plus a broad vowel-space scorer derived from the local `vowel_finder.py` reference system.",
        "",
        "Decision policy:",
        f"- Calibrate a same-skeleton acoustic margin on already-resolved disagreement rows; require at least {args.min_calibration_precision:.0%} in-sample precision.",
        "- Accept unresolved same-skeleton rows only when the acoustic margin clears that threshold.",
        "- Accept unresolved weak-support rows only when the acoustic pick matches the prior weak lexical lean and clears a reduced confirmation threshold.",
        "",
        "Outputs:",
        f"- `{acoustic_audit_path.name}`: row-by-row acoustic audit for eligible unresolved cases.",
        f"- `{additional_update_path.name}`: new acoustic-only LL Tools update rows.",
        f"- `{merged_update_path.name}`: prior conservative update plus acoustic additions.",
        f"- `{calibration_path.name}`: calibration rows with acoustic margins and correctness.",
        f"- `{summary_path.name}`: aggregate counts and calibrated thresholds.",
        "",
        "Summary:",
        f"- Eligible unresolved rows: {len(unresolved_rows)}",
        f"- Newly resolved by acoustics: {newly_resolved}",
        f"- Remaining unresolved after acoustics: {unresolved_after_acoustics}",
        f"- Calibration rows scored: {len(calibration_scored)}",
    ]
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
