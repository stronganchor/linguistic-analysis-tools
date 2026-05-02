#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

PYCTC_UNK_LABEL = "⁇"
PRIVATE_USE_START = 0xE000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Wav2Vec2 CTC inference speed on eval clips.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=["auto", "greedy", "beam"],
    )
    parser.add_argument("--beam-width", type=int, default=64)
    return parser.parse_args()


def resolve_audio_path(row: dict) -> Path:
    raw_value = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw_value = str(raw_value).strip()
    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail
    return Path(raw_value).expanduser()


def load_eval_rows(manifest_path: Path, limit: int) -> list[dict]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split") != "eval":
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def audio_duration_seconds(audio_path: Path) -> float:
    info = torchaudio.info(str(audio_path))
    if info.sample_rate <= 0:
        return 0.0
    return float(info.num_frames) / float(info.sample_rate)


def synchronize_if_needed(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_local_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "lltools_ctc_config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def placeholder_label(token_id: int) -> str:
    return chr(PRIVATE_USE_START + token_id)


def map_ctc_label(token: str, token_id: int, word_delimiter_token: str) -> str:
    if token == "[PAD]":
        return ""
    if token == "[UNK]":
        return PYCTC_UNK_LABEL
    if token == word_delimiter_token:
        return " "
    if token.startswith("<") and token.endswith(">"):
        return placeholder_label(token_id)
    return token


def build_pyctc_labels(
    vocab: dict[str, int],
    word_delimiter_token: str,
    vocab_size: int | None = None,
    added_tokens_decoder: dict | None = None,
) -> list[str]:
    max_vocab_id = max(vocab.values()) if vocab else -1
    max_added_id = -1
    if added_tokens_decoder:
        for token_id in added_tokens_decoder.keys():
            try:
                max_added_id = max(max_added_id, int(token_id))
            except (TypeError, ValueError):
                continue

    label_count = max(max_vocab_id, max_added_id, (vocab_size or 0) - 1) + 1
    labels = [placeholder_label(token_id) for token_id in range(label_count)]

    for token, token_id in vocab.items():
        labels[token_id] = map_ctc_label(token, token_id, word_delimiter_token)

    if added_tokens_decoder:
        for token_id_raw, token_meta in added_tokens_decoder.items():
            try:
                token_id = int(token_id_raw)
            except (TypeError, ValueError):
                continue
            if token_id < 0 or token_id >= len(labels):
                continue
            if any(existing_id == token_id for existing_id in vocab.values()):
                continue
            token = str((token_meta or {}).get("content") or "")
            labels[token_id] = map_ctc_label(token, token_id, word_delimiter_token)
    return labels


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    rows = load_eval_rows(manifest_path, args.limit)
    if not rows:
        raise RuntimeError("No eval rows found")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(args.device).eval()
    target_sr = int(feature_extractor.sampling_rate)
    local_model_dir = Path(args.model).expanduser().resolve()
    model_config = load_local_model_config(local_model_dir) if local_model_dir.exists() else {}
    decode_mode = args.decoder
    if decode_mode == "auto":
        decode_mode = str(model_config.get("preferred_decoder") or "greedy").strip().lower()

    decoder = None
    if decode_mode == "beam":
        from pyctcdecode import build_ctcdecoder

        vocab = json.loads((local_model_dir / "vocab.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads((local_model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        decoder = build_ctcdecoder(
            build_pyctc_labels(
                vocab,
                tokenizer_config.get("word_delimiter_token", "|"),
                vocab_size=int(model.config.vocab_size),
                added_tokens_decoder=tokenizer_config.get("added_tokens_decoder"),
            )
        )

    clip_times: list[float] = []
    total_audio_seconds = 0.0

    for row in rows:
        audio_path = resolve_audio_path(row)
        total_audio_seconds += audio_duration_seconds(audio_path)

        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        features = feature_extractor(
            waveform.numpy(),
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        input_values = features.input_values.to(args.device)
        attention_mask = getattr(features, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        synchronize_if_needed(args.device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            if decoder is not None:
                _ = decoder.decode(logits[0].float().cpu().numpy(), beam_width=args.beam_width)
            else:
                _ = torch.argmax(logits, dim=-1)
        synchronize_if_needed(args.device)
        elapsed = time.perf_counter() - started
        clip_times.append(elapsed)

    total_elapsed = sum(clip_times)
    benchmark = {
        "clips": len(rows),
        "total_audio_seconds": total_audio_seconds,
        "total_elapsed_seconds": total_elapsed,
        "avg_seconds_per_clip": total_elapsed / len(rows),
        "median_seconds_per_clip": statistics.median(clip_times),
        "realtime_factor": total_elapsed / total_audio_seconds if total_audio_seconds else 0.0,
        "decoder": decode_mode,
    }
    output_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(benchmark, ensure_ascii=False, indent=2))
    print(f"Wrote benchmark: {output_path}")


if __name__ == "__main__":
    main()
