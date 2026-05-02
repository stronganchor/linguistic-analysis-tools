import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from model_paths import resolve_stt_models_dir, stt_model_path
from prepare_lltools_ipa_dataset import normalize_ipa_text


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def default_model_source() -> str:
    script_dir = Path(__file__).resolve().parent
    models_dir = resolve_stt_models_dir(script_dir)
    discovered_models: list[Path] = []

    if models_dir.is_dir():
        for candidate in models_dir.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name.lower().startswith("checkpoint-"):
                continue
            if not (candidate / "config.json").is_file():
                continue
            discovered_models.append(candidate)

    if discovered_models:
        newest_model = max(
            discovered_models,
            key=lambda path: (
                getattr(path.stat(), "st_ctime", 0),
                path.stat().st_mtime,
                path.name.lower(),
            ),
        )
        return str(newest_model)

    candidates = [
        stt_model_path(script_dir, "whisper-small-ipa-doga1-from-zazaki-ft002"),
        stt_model_path(script_dir, "whisper-small-ipa-doga1-001"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return "openai/whisper-small"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe one or more audio files to IPA with a local Whisper model."
    )
    parser.add_argument(
        "--model-source",
        default=default_model_source(),
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--audio",
        action="append",
        dest="audio_paths",
        default=[],
        help="Audio file to transcribe. Repeat for multiple files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device override: auto, cuda, or cpu.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        help="Model precision for CUDA inference: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of files to decode at once.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search width. Higher can improve accuracy at the cost of speed.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Generation max_length for Whisper decoding.",
    )
    parser.add_argument(
        "--output-path",
        default="",
        help="Optional TSV or JSON output path. Extension controls the format.",
    )
    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Include clip duration in the output.",
    )
    return parser.parse_args()


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


def clean_user_path(raw_path: str) -> str:
    cleaned = (raw_path or "").strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def maybe_prompt_for_audio_paths(args: argparse.Namespace) -> argparse.Namespace:
    if len(sys.argv) > 1 or args.audio_paths:
        return args

    print("No arguments provided.")
    print(f"Using default model: {args.model_source}")
    raw_path = clean_user_path(input("Enter the path to the audio file: "))
    if not raw_path:
        raise SystemExit("No audio path entered.")
    args.audio_paths = [raw_path]
    return args


def validate_audio_paths(audio_paths: list[str]) -> list[Path]:
    if not audio_paths:
        raise SystemExit("Provide at least one --audio path.")

    resolved = []
    for raw_path in audio_paths:
        path = Path(clean_user_path(raw_path)).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Missing audio file: {path}")
        resolved.append(path)
    return resolved


def transcribe_paths(
    model_source: str,
    audio_paths: list[Path],
    device_name: str,
    precision: str,
    batch_size: int,
    num_beams: int,
    max_length: int,
    show_timings: bool,
) -> list[dict[str, str]]:
    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source)
    tokenizer = WhisperTokenizer.from_pretrained(model_source)
    processor = WhisperProcessor.from_pretrained(model_source)
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(model_source, **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []

    device = torch.device(device_name)
    model.to(device)
    model.eval()
    feature_dtype = resolve_feature_dtype(model)

    rows: list[dict[str, str]] = []

    with torch.no_grad():
        for start in range(0, len(audio_paths), batch_size):
            chunk_paths = audio_paths[start : start + batch_size]
            audio_arrays = []
            durations = []

            for audio_path in chunk_paths:
                audio_array, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
                audio_arrays.append(audio_array)
                durations.append(len(audio_array) / TARGET_SR)

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

            generated_ids = model.generate(**generate_kwargs)
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for audio_path, duration_seconds, predicted in zip(chunk_paths, durations, predictions):
                row = {
                    "audio_path": str(audio_path),
                    "predicted_ipa": normalize_ipa_text(predicted),
                }
                if show_timings:
                    row["duration_seconds"] = f"{duration_seconds:.3f}"
                rows.append(row)

            print(f"Decoded {min(start + batch_size, len(audio_paths))}/{len(audio_paths)} file(s)")

    return rows


def write_output(path: Path, rows: list[dict[str, str]]) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)
        return

    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = maybe_prompt_for_audio_paths(parse_args())
    audio_paths = validate_audio_paths(args.audio_paths)
    device_name = choose_device(args.device)

    print(f"Using device: {device_name}")
    print(f"Model:        {args.model_source}")

    rows = transcribe_paths(
        model_source=args.model_source,
        audio_paths=audio_paths,
        device_name=device_name,
        precision=args.precision,
        batch_size=max(1, args.batch_size),
        num_beams=max(1, args.num_beams),
        max_length=max(32, args.max_length),
        show_timings=args.show_timings,
    )

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
        write_output(output_path, rows)
        print(f"Wrote output:  {output_path}")

    for row in rows:
        print("")
        print(f"Audio: {row['audio_path']}")
        if "duration_seconds" in row:
            print(f"Length: {row['duration_seconds']}s")
        print(f"IPA:   {row['predicted_ipa']}")


if __name__ == "__main__":
    main()
