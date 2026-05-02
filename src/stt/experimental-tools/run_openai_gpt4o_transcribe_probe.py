#!/usr/bin/env python3
"""
Probe OpenAI gpt-4o-transcribe as an extra Zazaki STT witness.

The runner keeps raw outputs auditable and avoids printing API key values. It
prefers OPENAI_API_KEY from the current process, then reads the same variable
from Windows User/Machine env vars when running under WSL.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
STT_DIR = SCRIPT_DIR.parent
ROOT = STT_DIR.parent

AK_GOLD_JSONL = STT_DIR / "gold-set-review-ak-isolation-targeted-core32" / "gold-set-confirmed.jsonl"
AK_TEXT_CSV = (
    STT_DIR
    / "extracted_bundles_text"
    / "ll-tools-stt-training-genc-palu-recording_text-20260409-071931"
    / "metadata.csv"
)
CK_REVIEWED_TSV = (
    STT_DIR
    / "training_data"
    / "ck-reviewed-mined-ipa-20260413-1810-cumulative"
    / "manifest_ipa_reviewed_new_only.tsv"
)
UNRESOLVED_AK_NOTES = {
    "2772": "Agiz question unresolved family mismatch",
    "14599": "Kalem question unresolved onset mismatch",
    "7213": "Kumru question unresolved onset mismatch",
    "12064": "11 question unresolved jundas/jondas alternation",
}

AK_REVIEWED_IDS = [
    "15947",  # Tûm / t̪ʰum
    "15037",  # Gırıgıl / gɨɾɨ̯gɨl
    "15053",  # Dbûm / d̪͡bum
    "13779",  # Mwiz / mʷɪz
    "11770",  # Şow / ʃow
    "14751",  # Tpırp / t̪͡ʙ̥̆ɨɾp
    "3648",  # Yo ho çada?
    "4242",  # Mirçik ha perena
    "14232",  # Qıc bire xwi vina
]

CK_REVIEWED_SOURCE_IDS = [
    "phase1a_0204",
    "phase1b_processed_0744",
    "phase1a_0546",
    "phase1a_0803",
]

PROMPTS = {
    "none": None,
    "zazaki_orthography": (
        "The audio is Central Zazaki, also called Zaza or Dimili, in a Genç-Palu style. "
        "Transcribe the actual Zazaki speech. Do not translate into Turkish or English. "
        "Use a practical Latin Zazaki spelling with letters such as ç, ş, ı, ê, û, â when appropriate. "
        "Return only the transcription."
    ),
    "no_translate_soundalike": (
        "This is a short Zazaki language-learning recording. Write what you hear, not the Turkish or English meaning. "
        "If the word is unfamiliar, output the closest sound-alike spelling rather than a known Turkish word. "
        "Return only the spoken words."
    ),
    "phonetic_ipa_like": (
        "The audio is Central Zazaki. Return a close phonetic transcription. IPA symbols are allowed where useful "
        "for vowels or consonants; keep aspiration, uvular/fricative, pharyngeal, and affricate cues if audible. "
        "Do not translate. Return only the transcription."
    ),
}


def read_windows_env_var(names: list[str]) -> str:
    powershell = (
        "$names=@("
        + ",".join("'" + name.replace("'", "''") + "'" for name in names)
        + ");"
        + "foreach($scope in @('Process','User','Machine')){"
        + "foreach($n in $names){"
        + "$v=[Environment]::GetEnvironmentVariable($n,$scope);"
        + "if($v){[Console]::Out.Write($v); exit 0}"
        + "}} exit 1"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", powershell],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return ""
    if completed.returncode == 0:
        return completed.stdout.strip()
    return ""


def get_openai_api_key() -> str:
    for name in ("OPENAI_API_KEY", "OpenAI_API_Key", "OPENAI_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return read_windows_env_var(["OPENAI_API_KEY", "OpenAI_API_Key", "OPENAI_KEY"])


def windows_to_posix(path: str) -> str:
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", normalized)
    if match:
        drive = match.group(1).lower()
        return f"/mnt/{drive}/{match.group(2)}"
    return normalized


def to_windows_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text.replace("/", "\\")


def audio_path(row: dict[str, Any], base: Path | None = None) -> Path:
    for key in ("audio_posix", "audio", "audio_windows"):
        raw = str(row.get(key) or "").strip()
        if not raw:
            continue
        posix = windows_to_posix(raw)
        candidate = Path(posix)
        if candidate.exists():
            return candidate
        if base and not candidate.is_absolute():
            joined = base / candidate
            if joined.exists():
                return joined
    rel = str(row.get("audio_relpath") or "").strip()
    if rel and base:
        candidate = base / rel
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No audio file found for row {row}")


def has_riff_header(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"RIFF"
    except OSError:
        return False


def prepare_audio_for_api(path: Path, out_dir: Path, sample_id: str) -> tuple[Path, bool]:
    """Fix mislabeled files like WAV data stored under .mp3 names."""
    if path.suffix.lower() != ".mp3" or not has_riff_header(path):
        return path, False
    fixed_dir = out_dir / "_api_audio_wav"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id)
    fixed_path = fixed_dir / f"{safe_id}.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copyfile(path, fixed_path)
        return fixed_path, True
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(fixed_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg failed")
    return fixed_path, True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ak_samples() -> list[dict[str, Any]]:
    by_recording: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(AK_GOLD_JSONL):
        row = dict(row)
        row["sample_group"] = "ak_reviewed_gold_isolation"
        row["reference_orthography"] = row.get("orthography_text") or row.get("recording_text") or ""
        row["reference_ipa"] = row.get("recording_ipa") or ""
        by_recording[str(row.get("recording_id"))] = row

    text_base = AK_TEXT_CSV.parent
    with AK_TEXT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("speaker_user_id") or "") != "8":
                continue
            row = dict(row)
            row["sample_group"] = "ak_reviewed_text_export"
            row["reference_orthography"] = row.get("recording_text") or row.get("text") or ""
            row["reference_ipa"] = row.get("recording_ipa") or ""
            row["audio_posix"] = str(text_base / row.get("audio", ""))
            by_recording.setdefault(str(row.get("recording_id")), row)

    samples = []
    for recording_id in AK_REVIEWED_IDS:
        row = by_recording.get(recording_id)
        if not row:
            raise KeyError(f"Missing A.K. reviewed sample recording_id={recording_id}")
        row = dict(row)
        row["sample_id"] = f"ak_reviewed_{recording_id}"
        row["evaluation_bucket"] = "ak_reviewed"
        samples.append(row)

    for recording_id, note in UNRESOLVED_AK_NOTES.items():
        row = by_recording.get(recording_id)
        if not row:
            continue
        row = dict(row)
        row["sample_id"] = f"ak_watchlist_{recording_id}"
        row["sample_group"] = "ak_watchlist_unresolved"
        row["evaluation_bucket"] = "ak_watchlist"
        row["watchlist_note"] = note
        samples.append(row)
    return samples


def load_ck_samples() -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    with CK_REVIEWED_TSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            by_source[str(row.get("source_id") or "")] = dict(row)

    samples = []
    for source_id in CK_REVIEWED_SOURCE_IDS:
        row = by_source.get(source_id)
        if not row:
            raise KeyError(f"Missing C.K. reviewed sample source_id={source_id}")
        row = dict(row)
        row["sample_id"] = f"ck_reviewed_{source_id}"
        row["sample_group"] = "ck_reviewed_mined"
        row["evaluation_bucket"] = "ck_reviewed"
        row["recording_id"] = row.get("source_id") or source_id
        row["recording_type"] = row.get("recording_type") or "reviewed_mined"
        row["speaker_name"] = "C.K."
        row["reference_orthography"] = row.get("recording_text") or row.get("word_title") or ""
        row["reference_ipa"] = row.get("target_ipa") or ""
        samples.append(row)
    return samples


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = value.replace("ı", "i")
    value = re.sub(r"[^\w\sçşğüöâêûî']", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def norm_ipa(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[\s\.\,\?\!\-_'`]+", "", value)
    return value


def ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def classify_output(text: str, reference_orthography: str, reference_ipa: str) -> dict[str, Any]:
    output_text_norm = norm_text(text)
    ref_text_norm = norm_text(reference_orthography)
    output_ipa_norm = norm_ipa(text)
    ref_ipa_norm = norm_ipa(reference_ipa)
    return {
        "orthography_ratio": round(ratio(output_text_norm, ref_text_norm), 4),
        "ipa_ratio": round(ratio(output_ipa_norm, ref_ipa_norm), 4),
        "output_norm": output_text_norm,
        "reference_orthography_norm": ref_text_norm,
        "reference_ipa_norm": ref_ipa_norm,
        "empty_output": not output_text_norm,
        "looks_like_translation": bool(
            re.search(
                r"\b(the|a|an|this|that|is|are|bir|bu|şu|ve|için|olarak|ile)\b",
                output_text_norm,
            )
        ),
    }


def transcribe(client: OpenAI, path: Path, prompt: str | None, model: str, timeout_pause: float) -> str:
    with path.open("rb") as audio_file:
        kwargs: dict[str, Any] = {
            "model": model,
            "file": audio_file,
        }
        if prompt:
            kwargs["prompt"] = prompt
        result = client.audio.transcriptions.create(**kwargs)
    if timeout_pause > 0:
        time.sleep(timeout_pause)
    text = getattr(result, "text", None)
    if text is not None:
        return str(text).strip()
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(result).strip()


def write_tsv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        if row.get("ok"):
            buckets[(row["evaluation_bucket"], row["prompt_name"])].append(row)

    summary = []
    for (bucket, prompt_name), rows in sorted(buckets.items()):
        orth = [float(row["orthography_ratio"]) for row in rows]
        ipa = [float(row["ipa_ratio"]) for row in rows]
        summary.append(
            {
                "evaluation_bucket": bucket,
                "prompt_name": prompt_name,
                "n": len(rows),
                "avg_orthography_ratio": round(sum(orth) / len(orth), 4),
                "avg_ipa_ratio": round(sum(ipa) / len(ipa), 4),
                "orthography_ratio_ge_0_80": sum(1 for value in orth if value >= 0.8),
                "orthography_ratio_ge_0_60": sum(1 for value in orth if value >= 0.6),
                "ipa_ratio_ge_0_70": sum(1 for value in ipa if value >= 0.7),
                "translation_like_outputs": sum(1 for row in rows if row.get("looks_like_translation")),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-transcribe")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--include-ck", action="store_true")
    parser.add_argument(
        "--prompt",
        action="append",
        choices=sorted(PROMPTS),
        help="Prompt names to run. Defaults to all prompt styles.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        choices=["ak_reviewed", "ak_watchlist", "ck_reviewed"],
        help="Evaluation bucket to run. Defaults to A.K. reviewed and watchlist; add --include-ck for C.K.",
    )
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    parser.add_argument("--limit-samples", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = get_openai_api_key()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY was not found in process or Windows env vars.")

    client = OpenAI(api_key=api_key)
    prompts = args.prompt or list(PROMPTS)

    samples = load_ak_samples()
    if args.include_ck:
        samples.extend(load_ck_samples())
    allowed_buckets = set(args.bucket or ["ak_reviewed", "ak_watchlist"] + (["ck_reviewed"] if args.include_ck else []))
    samples = [sample for sample in samples if sample.get("evaluation_bucket") in allowed_buckets]
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else STT_DIR / "transcription_outputs" / f"openai_gpt4o_transcribe_probe_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_jsonl = out_dir / "openai_gpt4o_transcribe_results.jsonl"
    results: list[dict[str, Any]] = []
    with raw_jsonl.open("w", encoding="utf-8") as raw_handle:
        for sample in samples:
            try:
                path = audio_path(sample)
                api_path, container_fixed = prepare_audio_for_api(path, out_dir, str(sample.get("sample_id") or "sample"))
            except Exception as exc:
                row = {
                    "sample_id": sample.get("sample_id"),
                    "recording_id": sample.get("recording_id"),
                    "ok": False,
                    "error": f"audio_path_error: {exc}",
                }
                raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.append(row)
                continue

            for prompt_name in prompts:
                prompt = PROMPTS[prompt_name]
                started = time.time()
                row = {
                    "sample_id": sample.get("sample_id"),
                    "sample_group": sample.get("sample_group"),
                    "evaluation_bucket": sample.get("evaluation_bucket"),
                    "watchlist_note": sample.get("watchlist_note", ""),
                    "recording_id": str(sample.get("recording_id") or ""),
                    "word_id": str(sample.get("word_id") or ""),
                    "speaker_user_id": str(sample.get("speaker_user_id") or ""),
                    "speaker_name": sample.get("speaker_name") or "",
                    "recording_type": sample.get("recording_type") or sample.get("recording_types") or "",
                    "word_title": sample.get("word_title") or "",
                    "reference_orthography": sample.get("reference_orthography") or "",
                    "reference_ipa": sample.get("reference_ipa") or "",
                    "audio_windows": to_windows_path(path),
                    "audio_posix": str(path),
                    "api_audio_windows": to_windows_path(api_path),
                    "api_audio_posix": str(api_path),
                    "audio_container_fixed": container_fixed,
                    "prompt_name": prompt_name,
                    "prompt": prompt or "",
                    "model": args.model,
                }
                try:
                    text = transcribe(client, api_path, prompt, args.model, args.pause_seconds)
                    row.update(classify_output(text, row["reference_orthography"], row["reference_ipa"]))
                    row["transcription"] = text
                    row["ok"] = True
                except Exception as exc:
                    row["ok"] = False
                    row["error"] = f"{type(exc).__name__}: {exc}"
                row["elapsed_sec"] = round(time.time() - started, 3)
                raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_handle.flush()
                results.append(row)
                status = "ok" if row.get("ok") else "ERR"
                print(
                    f"{status}\t{row['sample_id']}\t{prompt_name}\t"
                    f"{row.get('orthography_ratio', '')}\t{row.get('transcription', row.get('error', ''))}",
                    flush=True,
                )

    detail_headers = [
        "evaluation_bucket",
        "sample_id",
        "recording_id",
        "speaker_name",
        "recording_type",
        "word_title",
        "reference_orthography",
        "reference_ipa",
        "prompt_name",
        "orthography_ratio",
        "ipa_ratio",
        "looks_like_translation",
        "transcription",
        "watchlist_note",
        "audio_windows",
        "api_audio_windows",
        "audio_container_fixed",
        "ok",
        "error",
    ]
    write_tsv(out_dir / "openai_gpt4o_transcribe_results.tsv", results, detail_headers)
    summary_rows = summarize(results)
    write_tsv(
        out_dir / "openai_gpt4o_transcribe_summary.tsv",
        summary_rows,
        [
            "evaluation_bucket",
            "prompt_name",
            "n",
            "avg_orthography_ratio",
            "avg_ipa_ratio",
            "orthography_ratio_ge_0_80",
            "orthography_ratio_ge_0_60",
            "ipa_ratio_ge_0_70",
            "translation_like_outputs",
        ],
    )
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# OpenAI gpt-4o-transcribe Zazaki probe",
                "",
                f"- model: `{args.model}`",
                f"- created: `{datetime.now().isoformat(timespec='seconds')}`",
                f"- samples: `{len(samples)}`",
                f"- prompt styles: `{', '.join(prompts)}`",
                "",
                "Files:",
                "",
                "- `openai_gpt4o_transcribe_results.jsonl`: full raw per-call output and metadata",
                "- `openai_gpt4o_transcribe_results.tsv`: compact per-call comparison table",
                "- `openai_gpt4o_transcribe_summary.tsv`: prompt/bucket aggregate scores",
                "",
                "Similarity scores are rough string-comparison diagnostics, not approval signals.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OUTPUT_DIR\t{to_windows_path(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
