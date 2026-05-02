#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor

from prepare_lltools_ipa_dataset import normalize_ipa_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remap a Wav2Vec2 CTC checkpoint to a new manifest-derived vocab while preserving matching lm_head rows."
    )
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strip-dental-stops", action="store_true")
    parser.add_argument("--redentalize-output", action="store_true")
    return parser.parse_args()


def load_manifest_targets(manifest_path: Path, strip_dental_stops: bool) -> list[str]:
    targets: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            target = normalize_ipa_text(
                str(row.get("target_ipa") or ""),
                strip_dental_stops=strip_dental_stops,
            )
            if target:
                targets.append(target)
    if not targets:
        raise RuntimeError(f"No targets found in {manifest_path}")
    return targets


def build_vocab(targets: list[str], output_dir: Path) -> tuple[Path, dict[str, int]]:
    chars = set()
    for target in targets:
        chars.update(target)

    chars.discard(" ")
    vocab_list = sorted(chars)
    vocab = {char: idx for idx, char in enumerate(vocab_list)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab_path, vocab


def load_source_vocab(source_dir: Path) -> dict[str, int]:
    vocab_path = source_dir / "vocab.json"
    return json.loads(vocab_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_model).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_manifest_targets(manifest_path, args.strip_dental_stops)
    vocab_path, target_vocab = build_vocab(targets, output_dir)

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(source_dir))
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(output_dir))

    source_model = Wav2Vec2ForCTC.from_pretrained(str(source_dir))
    remapped_model = Wav2Vec2ForCTC.from_pretrained(
        str(source_dir),
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
    )

    source_vocab = load_source_vocab(source_dir)
    copied_tokens: list[str] = []
    missing_tokens: list[str] = []

    with torch.no_grad():
        for token, new_index in target_vocab.items():
            old_index = source_vocab.get(token)
            if old_index is None:
                missing_tokens.append(token)
                continue
            remapped_model.lm_head.weight[new_index].copy_(source_model.lm_head.weight[old_index])
            remapped_model.lm_head.bias[new_index].copy_(source_model.lm_head.bias[old_index])
            copied_tokens.append(token)

    remapped_model.config.pad_token_id = tokenizer.pad_token_id
    remapped_model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    (output_dir / "lltools_ctc_config.json").write_text(
        json.dumps(
            {
                "strip_dental_stops": bool(args.strip_dental_stops),
                "redentalize_output": bool(args.redentalize_output),
                "preferred_decoder": "beam",
                "beam_width": 64,
                "remapped_from": str(source_dir),
                "copied_token_count": len(copied_tokens),
                "missing_tokens": missing_tokens,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source_model": str(source_dir),
                "output_dir": str(output_dir),
                "new_vocab_size": len(target_vocab),
                "copied_token_count": len(copied_tokens),
                "missing_tokens": missing_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
