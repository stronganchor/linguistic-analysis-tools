#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from prepare_lltools_ipa_dataset import normalize_ipa_text


def resolve_audio_path(row: dict[str, Any]) -> Path:
    raw_value = row.get("audio_posix") or row.get("audio") or row.get("audio_windows") or ""
    raw_value = str(raw_value).strip()
    if not raw_value:
        raise RuntimeError("Missing audio path in manifest row")

    if re.match(r"^[A-Za-z]:[\\/]", raw_value):
        drive = raw_value[0].lower()
        tail = raw_value[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / tail

    return Path(raw_value).expanduser()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 on IPA targets from manifest_ipa.tsv")
    parser.add_argument("--manifest", required=True, help="Path to manifest_ipa.tsv")
    parser.add_argument(
        "--base-model",
        default="$LINUX_STT_ENV/alt-models/hf-models/wav2vec2-xlsr-53-espeak-cv-ft",
        help="Base Wav2Vec2 model path or HF id",
    )
    parser.add_argument(
        "--target-lang",
        default="",
        help="Optional MMS target language adapter code, e.g. zza",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--run-name", default="wav2vec2-ipa-finetune", help="Trainer run name")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for larger models such as MMS.",
    )
    parser.add_argument(
        "--strip-dental-stops",
        action="store_true",
        help="Train on targets with t̪/d̪ normalized to plain t/d.",
    )
    parser.add_argument(
        "--redentalize-output",
        action="store_true",
        help="Post-process decoded outputs back to t̪/d̪ for metrics and serving.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training")
    return parser.parse_args()


def load_rows(
    manifest_path: Path,
    *,
    strip_dental_stops: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            target = normalize_ipa_text(
                str(row.get("target_ipa") or ""),
                strip_dental_stops=strip_dental_stops,
            )
            if not target:
                continue
            item = {
                "audio_path": str(resolve_audio_path(row)),
                "target_text": target,
            }
            if row.get("split") == "train":
                train_rows.append(item)
            elif row.get("split") == "eval":
                eval_rows.append(item)
    if not train_rows or not eval_rows:
        raise RuntimeError("Manifest is missing train or eval rows")
    return train_rows, eval_rows


def build_vocab(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    chars = set()
    for row in train_rows + eval_rows:
        chars.update(row["target_text"])

    chars.discard(" ")
    vocab_list = sorted(chars)
    vocab = {char: idx for idx, char in enumerate(vocab_list)}
    vocab["|"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)

    vocab_path = output_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    return vocab_path


class IPACTCDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], processor: Wav2Vec2Processor, target_sr: int):
        self.rows = rows
        self.processor = processor
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio_path = row["audio_path"]
        target_text = row["target_text"]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)

        input_values = self.processor.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.target_sr,
        ).input_values[0]

        labels = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
        ).input_ids

        return {
            "input_values": input_values,
            "labels": labels,
        }


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool | str = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, eval_rows = load_rows(
        manifest_path,
        strip_dental_stops=args.strip_dental_stops,
    )

    vocab_path = build_vocab(train_rows, eval_rows, output_dir)

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
        do_lower_case=False,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(output_dir)

    target_sr = int(feature_extractor.sampling_rate)
    train_dataset = IPACTCDataset(train_rows, processor, target_sr)
    eval_dataset = IPACTCDataset(eval_rows, processor, target_sr)

    model_kwargs = {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "ctc_loss_reduction": "mean",
        "ctc_zero_infinity": True,
        "ignore_mismatched_sizes": True,
    }
    if args.target_lang.strip():
        model_kwargs["target_lang"] = args.target_lang.strip()

    model = Wav2Vec2ForCTC.from_pretrained(args.base_model, **model_kwargs)

    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pairs = [
            (
                normalize_ipa_text(
                    ref.replace("|", " "),
                    redentalize_stops=args.redentalize_output,
                ),
                normalize_ipa_text(
                    hyp.replace("|", " "),
                    redentalize_stops=args.redentalize_output,
                ),
            )
            for ref, hyp in zip(label_str, pred_str)
        ]
        cer = compute_cer(pairs)
        exact = sum(1 for ref, hyp in pairs if ref == hyp) / len(pairs)
        return {"cer": cer, "exact_match": exact}

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        group_by_length=True,
        dataloader_num_workers=args.dataloader_workers,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    started = time.time()
    trainer.train()
    train_elapsed = time.time() - started
    trainer.save_model(str(output_dir))
    processor.save_pretrained(output_dir)

    summary = {
        "base_model": args.base_model,
        "target_lang": args.target_lang.strip() or None,
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_seconds": train_elapsed,
        "feature_encoder_frozen": bool(args.freeze_feature_encoder),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "vocab_size": len(tokenizer),
        "target_sampling_rate": target_sr,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_cer": trainer.state.best_metric,
        "strip_dental_stops": bool(args.strip_dental_stops),
        "redentalize_output": bool(args.redentalize_output),
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "lltools_ctc_config.json").write_text(
        json.dumps(
            {
                "strip_dental_stops": bool(args.strip_dental_stops),
                "redentalize_output": bool(args.redentalize_output),
                "preferred_decoder": "beam",
                "beam_width": 64,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
