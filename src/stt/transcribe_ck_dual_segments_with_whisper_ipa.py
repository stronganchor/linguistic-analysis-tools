#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import librosa
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from ipa_rule_checks import basic_repair_ipa_text
from prepare_lltools_ipa_dataset import normalize_ipa_text
from review_lltools_ipa_comparison import ipa_to_canonical


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TARGET_SR = 16000
SCRIPT_DIR = Path(__file__).resolve().parent


def default_input_tsv() -> str:
    return str(
        SCRIPT_DIR
        / "transcription_outputs"
        / "ck_corpus_dual_ctc_20260411-131711"
        / "ck_dual_transcriptions.tsv"
    )


def default_model_dir() -> str:
    return str(
        SCRIPT_DIR
        / "models"
        / "high-accuracy"
        / "whisper-medium-ipa-ck-ak-reviewed-v002-bf16-20260412-1631"
    )


def default_processor_dir() -> str:
    return str(
        SCRIPT_DIR
        / "models"
        / "high-accuracy"
        / "whisper-medium-ipa-nostress-genc-palu-warmstart-v004-20260406"
    )


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def choose_device(value: str) -> str:
    requested = (value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_torch_dtype(device_name: str, precision: str):
    requested = (precision or "").strip().lower()
    if device_name != "cuda":
        return None
    if requested in ("", "auto", "float16", "fp16", "half"):
        return torch.float16
    if requested in ("bfloat16", "bf16"):
        return torch.bfloat16
    if requested in ("float32", "fp32"):
        return torch.float32
    raise SystemExit(f"Unsupported precision: {precision}")


def resolve_feature_dtype(model) -> torch.dtype | None:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add a Whisper-medium IPA witness to the existing C.K. dual transcription table, "
            "optionally excluding source ids."
        )
    )
    parser.add_argument("--input-tsv", default=default_input_tsv())
    parser.add_argument("--model-source", default=default_model_dir())
    parser.add_argument("--processor-source", default=default_processor_dir())
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "transcription_outputs"))
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--exclude-source-id",
        action="append",
        dest="exclude_source_ids",
        default=[],
        help="Repeatable source_id exclusion filter.",
    )
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name.strip() or f"ck_corpus_whisper_witness_{timestamp_slug()}"
    run_dir = Path(args.output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_rows(path: Path, exclude_source_ids: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader if str(row.get("source_id") or "") not in exclude_source_ids]
    if not rows:
        raise RuntimeError("No rows remain after filtering.")
    return rows


def load_model_bundle(
    model_source: Path,
    processor_source: Path,
    device_name: str,
    precision: str,
    num_beams: int,
) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    torch_dtype = choose_torch_dtype(device_name, precision)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype

    feature_extractor = WhisperFeatureExtractor.from_pretrained(str(processor_source))
    tokenizer = WhisperTokenizer.from_pretrained(str(processor_source))
    processor = WhisperProcessor.from_pretrained(str(processor_source))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_source), **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.num_beams = max(1, num_beams)

    device = torch.device(device_name)
    model.to(device).eval()
    return {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "feature_dtype": resolve_feature_dtype(model),
    }


def load_audio(audio_path: str) -> Any:
    audio_array, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    return audio_array


def decode_rows(
    rows: list[dict[str, str]],
    bundle: dict[str, Any],
    *,
    batch_size: int,
    num_beams: int,
    max_length: int,
    output_path: Path,
) -> list[dict[str, str]]:
    feature_extractor = bundle["feature_extractor"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    feature_dtype = bundle["feature_dtype"]

    predictions: list[dict[str, str]] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            audio_arrays = [load_audio(str(row["clip_audio_posix"])) for row in chunk]
            batch = feature_extractor(
                audio_arrays,
                sampling_rate=TARGET_SR,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = batch["input_features"].to(device)
            if feature_dtype is not None:
                input_features = input_features.to(dtype=feature_dtype)

            generate_kwargs = {
                "input_features": input_features,
                "max_length": max(32, max_length),
                "num_beams": max(1, num_beams),
            }
            if "attention_mask" in batch:
                generate_kwargs["attention_mask"] = batch["attention_mask"].to(device)

            generated = model.generate(**generate_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for row, raw_prediction in zip(chunk, decoded):
                prediction = normalize_ipa_text(raw_prediction)
                repaired = basic_repair_ipa_text(prediction)
                predictions.append(
                    {
                        "segment_id": str(row["segment_id"]),
                        "whisper_predicted_ipa": prediction,
                        "whisper_predicted_ipa_repaired": repaired,
                        "whisper_predicted_ipa_canonical": ipa_to_canonical(repaired or prediction) if (repaired or prediction) else "",
                    }
                )

            if ((start // batch_size) + 1) % 25 == 0 or (start + batch_size) >= len(rows):
                write_tsv(
                    output_path,
                    predictions,
                    [
                        "segment_id",
                        "whisper_predicted_ipa",
                        "whisper_predicted_ipa_repaired",
                        "whisper_predicted_ipa_canonical",
                    ],
                )
            print(f"Whisper decoded {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    return predictions


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_combined_rows(
    base_rows: list[dict[str, str]],
    whisper_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    whisper_map = {str(row["segment_id"]): row for row in whisper_rows}
    combined: list[dict[str, str]] = []
    for row in base_rows:
        witness = whisper_map.get(str(row["segment_id"]), {})
        merged = dict(row)
        merged["whisper_predicted_ipa"] = str(witness.get("whisper_predicted_ipa") or "")
        merged["whisper_predicted_ipa_repaired"] = str(witness.get("whisper_predicted_ipa_repaired") or "")
        merged["whisper_predicted_ipa_canonical"] = str(witness.get("whisper_predicted_ipa_canonical") or "")
        combined.append(merged)
    return combined


def build_joined_source_rows(combined_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined_rows:
        grouped[str(row["source_id"])].append(row)

    joined_rows: list[dict[str, Any]] = []
    for source_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["segment_index"]))
        first = rows[0]
        joined_rows.append(
            {
                "source_id": source_id,
                "source_title": first["source_title"],
                "source_group": first["source_group"],
                "source_kind": first["source_kind"],
                "source_audio_windows": first["source_audio_windows"],
                "segment_count": len(rows),
                "total_segment_sec": round(sum(float(row["duration_sec"]) for row in rows), 3),
                "joined_mms_orthography": " ".join(row["predicted_orthography"] for row in rows if row["predicted_orthography"]),
                "joined_mms_ipa": " ".join(row["predicted_ipa"] for row in rows if row["predicted_ipa"]),
                "joined_whisper_ipa": " ".join(
                    row["whisper_predicted_ipa_repaired"] or row["whisper_predicted_ipa"]
                    for row in rows
                    if (row.get("whisper_predicted_ipa_repaired") or row.get("whisper_predicted_ipa"))
                ),
            }
        )
    return sorted(joined_rows, key=lambda row: (str(row["source_group"]), str(row["source_title"])))


def main() -> None:
    global args
    args = parse_args()
    input_tsv = Path(args.input_tsv).expanduser().resolve()
    model_source = Path(args.model_source).expanduser().resolve()
    processor_source = Path(args.processor_source).expanduser().resolve()
    run_dir = make_run_dir(args)
    device_name = choose_device(args.device)
    exclude_source_ids = {item.strip() for item in args.exclude_source_ids if item.strip()}

    print(f"Run dir: {run_dir}", flush=True)
    print(f"Device: {device_name}", flush=True)
    print(f"Input TSV: {input_tsv}", flush=True)
    print(f"Excluded source ids: {sorted(exclude_source_ids)}", flush=True)

    base_rows = read_rows(input_tsv, exclude_source_ids)
    bundle = load_model_bundle(
        model_source,
        processor_source,
        device_name,
        args.precision,
        max(1, args.num_beams),
    )

    whisper_path = run_dir / "ck_whisper_ipa_predictions.tsv"
    combined_tsv = run_dir / "ck_triple_transcriptions.tsv"
    combined_jsonl = run_dir / "ck_triple_transcriptions.jsonl"
    source_inventory_tsv = run_dir / "ck_source_inventory_remaining.tsv"
    joined_sources_tsv = run_dir / "ck_joined_by_source.tsv"
    summary_path = run_dir / "summary.json"
    readme_path = run_dir / "README.md"

    whisper_rows = decode_rows(
        base_rows,
        bundle,
        batch_size=max(1, args.batch_size),
        num_beams=max(1, args.num_beams),
        max_length=max(32, args.max_length),
        output_path=whisper_path,
    )
    combined_rows = build_combined_rows(base_rows, whisper_rows)

    write_tsv(
        combined_tsv,
        combined_rows,
        list(base_rows[0].keys())
        + [
            "whisper_predicted_ipa",
            "whisper_predicted_ipa_repaired",
            "whisper_predicted_ipa_canonical",
        ],
    )
    write_jsonl(combined_jsonl, combined_rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in combined_rows:
        grouped[str(row["source_id"])].append(row)
    source_summary_rows = []
    for source_id, rows in grouped.items():
        first = rows[0]
        source_summary_rows.append(
            {
                "source_id": source_id,
                "source_title": first["source_title"],
                "source_group": first["source_group"],
                "source_kind": first["source_kind"],
                "source_audio_posix": first["source_audio_posix"],
                "source_audio_windows": first["source_audio_windows"],
                "source_audio_relpath": first["source_audio_relpath"],
                "segment_count": len(rows),
                "total_segment_sec": round(sum(float(item["duration_sec"]) for item in rows), 3),
            }
        )
    write_tsv(
        source_inventory_tsv,
        sorted(source_summary_rows, key=lambda row: (str(row["source_group"]), str(row["source_title"]))),
        [
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "source_audio_posix",
            "source_audio_windows",
            "source_audio_relpath",
            "segment_count",
            "total_segment_sec",
        ],
    )

    joined_rows = build_joined_source_rows(combined_rows)
    write_tsv(
        joined_sources_tsv,
        joined_rows,
        [
            "source_id",
            "source_title",
            "source_group",
            "source_kind",
            "source_audio_windows",
            "segment_count",
            "total_segment_sec",
            "joined_mms_orthography",
            "joined_mms_ipa",
            "joined_whisper_ipa",
        ],
    )

    summary = {
        "input_tsv": str(input_tsv),
        "model_source": str(model_source),
        "processor_source": str(processor_source),
        "device": device_name,
        "precision": args.precision,
        "num_beams": max(1, args.num_beams),
        "max_length": max(32, args.max_length),
        "excluded_source_ids": sorted(exclude_source_ids),
        "segment_count": len(combined_rows),
        "source_count": len(source_summary_rows),
        "source_group_counts": dict(Counter(str(row["source_group"]) for row in combined_rows)),
        "outputs": {
            "whisper_predictions_tsv": str(whisper_path),
            "triple_tsv": str(combined_tsv),
            "triple_jsonl": str(combined_jsonl),
            "source_inventory_tsv": str(source_inventory_tsv),
            "joined_by_source_tsv": str(joined_sources_tsv),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# C.K. Triple Transcription Run",
                "",
                "This folder adds a Whisper-medium IPA witness to the existing C.K. MMS dual transcription table.",
                "",
                "Excluded source ids:",
                *[f"- `{item}`" for item in sorted(exclude_source_ids)],
                "",
                "Main files:",
                "- `ck_whisper_ipa_predictions.tsv`: Whisper witness keyed by `segment_id`",
                "- `ck_triple_transcriptions.tsv`: existing MMS orthography + MMS IPA + Whisper IPA witness",
                "- `ck_joined_by_source.tsv`: joined view by remaining source recording",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
