#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC

from prepare_lltools_ipa_dataset import normalize_ipa_text

PYCTC_UNK_LABEL = "⁇"
PRIVATE_USE_START = 0xE000


def compact_ipa_text(value: str) -> str:
    return re.sub(r"\s+", "", normalize_ipa_text(value))


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def compute_cer(pairs: list[tuple[str, str]]) -> float:
    total_edits = 0
    total_chars = 0
    for ref, hyp in pairs:
        total_edits += levenshtein_distance(ref, hyp)
        total_chars += len(ref)
    return (total_edits / total_chars) if total_chars else 0.0


def load_eval_rows(manifest_path: Path) -> list[dict]:
    rows = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("split") != "eval":
                continue
            rows.append(row)
    return rows


def resolve_audio_path(row: dict) -> Path:
    raw_value = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw_value = str(raw_value).strip()
    if not raw_value:
        raise RuntimeError("Missing audio path in manifest row")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail

    return Path(raw_value).expanduser()


def decode_ctc_ids(
    predicted_ids: torch.Tensor,
    id_to_token: dict[int, str],
    blank_id: int,
    word_delimiter_token: str,
) -> str:
    pieces: list[str] = []
    prev = None
    for item in predicted_ids.tolist():
        if item == prev:
            continue
        prev = item
        if item == blank_id:
            continue
        token = id_to_token.get(item, "")
        if not token or token.startswith("<"):
            continue
        if token == word_delimiter_token:
            pieces.append(" ")
        else:
            pieces.append(token)
    return normalize_ipa_text("".join(pieces))


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


def clean_beam_decoded_text(value: str) -> str:
    value = value.replace(PYCTC_UNK_LABEL, "")
    return re.sub(r"[\ue000-\uf8ff]", "", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face CTC phone-recognition model on manifest_ipa.tsv."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest_ipa.tsv")
    parser.add_argument(
        "--model",
        default="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write eval metrics and predictions",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit of eval rows for smoke testing",
    )
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=["auto", "greedy", "beam"],
        help="CTC decoding strategy. Auto uses lltools_ctc_config.json when present.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=64,
        help="Beam width when using pyctcdecode beam search.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_eval_rows(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError(f"No eval rows found in {manifest_path}")

    model_source = args.model
    local_model_dir = Path(model_source).expanduser().resolve()
    is_local_model = local_model_dir.exists()

    print(f"Loading model: {model_source}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
    model = Wav2Vec2ForCTC.from_pretrained(model_source)
    model = model.to(args.device).eval()

    if is_local_model:
        vocab_path = local_model_dir / "vocab.json"
        tokenizer_config_path = local_model_dir / "tokenizer_config.json"
        model_config = load_local_model_config(local_model_dir)
    else:
        from huggingface_hub import hf_hub_download

        vocab_path = hf_hub_download(args.model, "vocab.json")
        tokenizer_config_path = hf_hub_download(args.model, "tokenizer_config.json")
        model_config = {}

    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    tokenizer_config = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))

    id_to_token = {token_id: token for token, token_id in vocab.items()}
    blank_id = int(model.config.pad_token_id)
    word_delimiter_token = tokenizer_config.get("word_delimiter_token", "|")
    target_sr = int(feature_extractor.sampling_rate)
    redentalize_output = bool(model_config.get("redentalize_output"))
    decode_mode = args.decoder
    beam_width = int(model_config.get("beam_width") or args.beam_width)
    decoder = None
    if decode_mode == "auto":
        decode_mode = str(model_config.get("preferred_decoder") or "greedy").strip().lower()
    if decode_mode == "beam":
        from pyctcdecode import build_ctcdecoder

        decoder = build_ctcdecoder(
            build_pyctc_labels(
                vocab,
                word_delimiter_token,
                vocab_size=int(model.config.vocab_size),
                added_tokens_decoder=tokenizer_config.get("added_tokens_decoder"),
            )
        )

    pairs_normalized: list[tuple[str, str]] = []
    pairs_compact: list[tuple[str, str]] = []
    prediction_rows: list[dict[str, str | float]] = []

    started = time.time()
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row)
        target = normalize_ipa_text(str(row.get("target_ipa") or ""))
        compact_target = compact_ipa_text(target)

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

        with torch.no_grad():
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
        if decoder is not None:
            prediction = normalize_ipa_text(
                clean_beam_decoded_text(
                    decoder.decode(logits[0].float().cpu().numpy(), beam_width=beam_width)
                ),
                redentalize_stops=redentalize_output,
            )
        else:
            predicted_ids = torch.argmax(logits, dim=-1)[0].cpu()
            prediction = decode_ctc_ids(
                predicted_ids=predicted_ids,
                id_to_token=id_to_token,
                blank_id=blank_id,
                word_delimiter_token=word_delimiter_token,
            )
            prediction = normalize_ipa_text(
                prediction,
                redentalize_stops=redentalize_output,
            )
        compact_prediction = compact_ipa_text(prediction)

        pairs_normalized.append((target, prediction))
        pairs_compact.append((compact_target, compact_prediction))
        prediction_rows.append(
            {
                "audio": str(audio_path),
                "reference_ipa": target,
                "predicted_ipa": prediction,
                "reference_compact": compact_target,
                "predicted_compact": compact_prediction,
            }
        )

        if index % 10 == 0 or index == len(rows):
            print(f"{index}/{len(rows)}")

    elapsed = time.time() - started
    metrics = {
        "model": args.model,
        "eval_examples": len(rows),
        "test_cer": compute_cer(pairs_normalized),
        "test_exact_match": sum(1 for ref, hyp in pairs_normalized if ref == hyp) / len(pairs_normalized),
        "compact_test_cer": compute_cer(pairs_compact),
        "compact_test_exact_match": sum(1 for ref, hyp in pairs_compact if ref == hyp) / len(pairs_compact),
        "runtime_seconds": elapsed,
        "avg_seconds_per_clip": elapsed / len(rows),
        "decoder": decode_mode,
        "beam_width": beam_width if decode_mode == "beam" else None,
        "redentalize_output": redentalize_output,
    }

    metrics_path = output_dir / "eval_metrics.json"
    predictions_path = output_dir / "eval_predictions.tsv"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio",
                "reference_ipa",
                "predicted_ipa",
                "reference_compact",
                "predicted_compact",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
