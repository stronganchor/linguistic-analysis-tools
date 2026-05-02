import argparse
import csv
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper on an audio->text manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to a tab-separated training manifest.")
    parser.add_argument(
        "--model-source",
        default="openai/whisper-small",
        help="Hugging Face model id or local Whisper model directory.",
    )
    parser.add_argument(
        "--processor-source",
        default="",
        help="Optional separate tokenizer/processor source. Defaults to --model-source.",
    )
    parser.add_argument(
        "--target-field",
        default="target_ipa",
        help="Manifest column containing the training target text.",
    )
    parser.add_argument(
        "--target-mode",
        choices=["ipa", "plain"],
        default="ipa",
        help="Normalization mode for the target text column.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints and the final model should be written.",
    )
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--generation-num-beams", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Model dtype: auto, float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the base model with bitsandbytes 8-bit quantization.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with bitsandbytes 4-bit quantization (QLoRA-style).",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        default="bfloat16",
        help="Compute dtype for 4-bit loading: auto, float16, bfloat16, float32.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        default="nf4",
        help="Quantization type for 4-bit loading: nf4 or fp4.",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        action="store_true",
        help="Enable nested quantization when loading in 4-bit.",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Wrap Whisper with LoRA adapters instead of full fine-tuning.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,v_proj",
        help="Comma-separated module names to target with LoRA.",
    )
    parser.add_argument(
        "--lora-modules-to-save",
        default="",
        help="Comma-separated module names to keep fully trainable and save alongside LoRA adapters.",
    )
    parser.add_argument(
        "--save-merged-model",
        action="store_true",
        help="If using LoRA, merge adapters into a standalone model after training.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze Whisper encoder weights and train only decoder-side parameters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and preprocess the dataset, then exit before training.",
    )
    return parser.parse_args()


def normalize_target_text(value: str, mode: str) -> str:
    if mode == "plain":
        return normalize_plain_text(value)
    return normalize_ipa_text(value)


def choose_torch_dtype(value: str):
    requested = str(value or "").strip().lower()
    if requested in ("", "auto"):
        return torch.float16 if torch.cuda.is_available() else None
    if requested in ("float16", "fp16", "half"):
        return torch.float16
    if requested in ("bfloat16", "bf16"):
        return torch.bfloat16
    if requested in ("float32", "fp32"):
        return torch.float32
    raise SystemExit(f"Unsupported torch dtype: {value}")


def choose_trainer_precision(torch_dtype):
    fp16 = False
    bf16 = False
    if not torch.cuda.is_available():
        return fp16, bf16
    if torch_dtype == torch.bfloat16:
        bf16 = True
    elif torch_dtype == torch.float16 or torch_dtype is None:
        fp16 = True
    return fp16, bf16


def build_quantization_config(args: argparse.Namespace):
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one of --load-in-4bit or --load-in-8bit.")
    if not args.load_in_4bit and not args.load_in_8bit:
        return None
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "Quantized loading requires transformers with BitsAndBytesConfig support."
        )

    quant_kwargs = {
        "load_in_4bit": bool(args.load_in_4bit),
        "load_in_8bit": bool(args.load_in_8bit),
    }
    if args.load_in_4bit:
        quant_kwargs["bnb_4bit_quant_type"] = str(args.bnb_4bit_quant_type or "nf4").strip().lower() or "nf4"
        quant_kwargs["bnb_4bit_use_double_quant"] = bool(args.bnb_4bit_use_double_quant)
        compute_dtype = choose_torch_dtype(args.bnb_4bit_compute_dtype)
        if compute_dtype is None:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        quant_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
    return BitsAndBytesConfig(**quant_kwargs)


def read_manifest(path: Path, *, target_field: str, target_mode: str) -> DatasetDict:
    train_rows = []
    eval_rows = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            audio_candidates = [
                str(row.get("audio") or "").strip(),
                str(row.get("audio_windows") or "").strip(),
                str(row.get("audio_posix") or "").strip(),
            ]
            audio_path = ""
            for candidate in audio_candidates:
                if candidate and os.path.isfile(candidate):
                    audio_path = candidate
                    break
            target = normalize_target_text(str(row.get(target_field) or ""), target_mode)
            split = str(row.get("split") or "").strip().lower() or "train"

            if not audio_path or not target:
                continue
            if not os.path.isfile(audio_path):
                print(f"[WARN] missing audio file, skipping: {audio_path}")
                continue

            payload = {
                "audio": audio_path,
                "sentence": target,
                "recording_type": str(row.get("recording_type") or "").strip(),
                "word_title": str(row.get("word_title") or "").strip(),
                "source_bundle": str(row.get("source_bundle") or "").strip(),
            }
            if split == "eval":
                eval_rows.append(payload)
            else:
                train_rows.append(payload)

    if not train_rows:
        raise RuntimeError("No training rows found in manifest.")
    if not eval_rows:
        eval_rows = train_rows[-1:]

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "eval": Dataset.from_list(eval_rows),
        }
    )


def maybe_limit_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def prepare_batch(feature_extractor, tokenizer, *, target_mode: str):
    def _fn(example):
        audio_path = example["audio"]
        text = normalize_target_text(example["sentence"], target_mode)

        audio_array, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        input_features = feature_extractor(
            audio_array,
            sampling_rate=TARGET_SR,
        ).input_features[0]
        label_ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()

        return {
            "input_features": input_features,
            "labels": label_ids,
        }

    return _fn


def data_collator(features):
    feat_tensors = []
    max_t = 0

    for item in features:
        feat = torch.tensor(item["input_features"], dtype=torch.float32)
        feat_tensors.append(feat)
        max_t = max(max_t, feat.shape[-1])

    padded_feats = []
    for feat in feat_tensors:
        if feat.shape[-1] < max_t:
            feat = torch.nn.functional.pad(feat, (0, max_t - feat.shape[-1]), value=0.0)
        padded_feats.append(feat)

    input_features = torch.stack(padded_feats)

    label_lists = [item["labels"] for item in features]
    max_l = max(len(values) for values in label_lists)
    padded_labels = []
    for values in label_lists:
        if len(values) < max_l:
            values = values + [-100] * (max_l - len(values))
        padded_labels.append(torch.tensor(values, dtype=torch.long))

    return {
        "input_features": input_features,
        "labels": torch.stack(padded_labels),
    }


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    prev = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        cur = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def compute_cer(predictions: list[str], references: list[str], *, target_mode: str) -> float:
    total_distance = 0
    total_chars = 0
    for pred, ref in zip(predictions, references):
        pred = normalize_target_text(pred, target_mode)
        ref = normalize_target_text(ref, target_mode)
        total_distance += levenshtein_distance(pred, ref)
        total_chars += max(1, len(ref))
    return total_distance / max(1, total_chars)


def print_trainable_parameter_summary(model) -> None:
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
    ratio = (trainable_params / total_params * 100.0) if total_params else 0.0
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {total_params:,} || "
        f"trainable%: {ratio:.4f}"
    )


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = None
        input_features = inputs.get("input_features")
        if (
            model_dtype is not None
            and input_features is not None
            and torch.is_floating_point(input_features)
        ):
            inputs["input_features"] = input_features.to(dtype=model_dtype)
        return inputs


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading manifest: {manifest_path}")
    dataset = read_manifest(
        manifest_path,
        target_field=str(args.target_field or "target_ipa"),
        target_mode=args.target_mode,
    )
    dataset["train"] = maybe_limit_dataset(dataset["train"], args.max_train_samples)
    dataset["eval"] = maybe_limit_dataset(dataset["eval"], args.max_eval_samples)

    print(f"Train rows: {len(dataset['train'])}")
    print(f"Eval rows:  {len(dataset['eval'])}")

    processor_source = str(args.processor_source or args.model_source)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(processor_source)
    tokenizer = WhisperTokenizer.from_pretrained(processor_source)
    processor = WhisperProcessor.from_pretrained(processor_source)
    model_kwargs = {}
    torch_dtype = choose_torch_dtype(args.torch_dtype)
    quantization_config = build_quantization_config(args)
    if quantization_config is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit/8-bit model loading requires CUDA.")
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(args.model_source, **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.num_beams = max(1, args.generation_num_beams)

    if args.freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False

    if args.gradient_checkpointing:
        model.config.use_cache = False

    if args.use_lora:
        if (
            get_peft_model is None
            or LoraConfig is None
        ):
            raise RuntimeError(
                "LoRA training requires the 'peft' package. Install it in the STT venv first."
            )
        if quantization_config is not None:
            if prepare_model_for_kbit_training is None:
                raise RuntimeError(
                    "Quantized LoRA training requires prepare_model_for_kbit_training from peft."
                )
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(args.gradient_checkpointing),
            )
        lora_targets = [
            item.strip()
            for item in str(args.lora_target_modules or "").split(",")
            if item.strip()
        ]
        lora_modules_to_save = [
            item.strip()
            for item in str(args.lora_modules_to_save or "").split(",")
            if item.strip()
        ]
        if not lora_targets:
            raise RuntimeError("No LoRA target modules were provided.")
        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            inference_mode=False,
            r=max(1, args.lora_r),
            lora_alpha=max(1, args.lora_alpha),
            lora_dropout=max(0.0, float(args.lora_dropout)),
            bias="none",
            target_modules=lora_targets,
            modules_to_save=lora_modules_to_save or None,
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        print_trainable_parameter_summary(model)

    print(f"Preprocessing audio and {args.target_mode} targets...")
    train_proc = dataset["train"].map(
        prepare_batch(feature_extractor, tokenizer, target_mode=args.target_mode),
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    eval_proc = dataset["eval"].map(
        prepare_batch(feature_extractor, tokenizer, target_mode=args.target_mode),
        remove_columns=dataset["eval"].column_names,
        num_proc=1,
    )

    if args.dry_run:
        preview_path = output_dir / "dry_run_preview.json"
        preview = {
            "train_examples": len(train_proc),
            "eval_examples": len(eval_proc),
            "sample_label_text": normalize_target_text(dataset["train"][0]["sentence"], args.target_mode),
            "sample_audio": dataset["train"][0]["audio"],
            "model_source": args.model_source,
            "processor_source": processor_source,
            "target_field": args.target_field,
            "target_mode": args.target_mode,
        }
        with preview_path.open("w", encoding="utf-8") as handle:
            json.dump(preview, handle, ensure_ascii=False, indent=2)
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        print(f"Wrote dry-run preview: {preview_path}")
        return

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        labels = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_text = [normalize_target_text(x, args.target_mode) for x in pred_text]
        label_text = [normalize_target_text(x, args.target_mode) for x in label_text]

        cer = compute_cer(pred_text, label_text, target_mode=args.target_mode)
        exact = sum(1 for pred, ref in zip(pred_text, label_text) if pred == ref) / max(1, len(label_text))
        return {"cer": cer, "exact_match": exact}

    trainer_fp16, trainer_bf16 = choose_trainer_precision(torch_dtype)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
        gradient_checkpointing=bool(args.gradient_checkpointing),
        learning_rate=args.learning_rate,
        warmup_ratio=max(0.0, args.warmup_ratio),
        num_train_epochs=args.num_train_epochs,
        fp16=trainer_fp16,
        bf16=trainer_bf16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=max(1, args.generation_num_beams),
        logging_steps=5,
        report_to=[],
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        optim=str(args.optim),
        max_grad_norm=max(0.0, args.max_grad_norm),
    )

    trainer = WhisperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_proc,
        eval_dataset=eval_proc,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    feature_extractor.save_pretrained(str(output_dir))

    print("Running final evaluation...")
    eval_output = trainer.predict(eval_proc)
    metrics = {k: float(v) for k, v in eval_output.metrics.items()}
    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    pred_ids = eval_output.predictions
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    label_ids = np.where(eval_output.label_ids == -100, tokenizer.pad_token_id, eval_output.label_ids)
    pred_text = [normalize_ipa_text(x) for x in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
    label_text = [normalize_ipa_text(x) for x in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    predictions_path = output_dir / "eval_predictions.tsv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["reference_ipa", "predicted_ipa"])
        writer.writerows(zip(label_text, pred_text))

    print(f"Wrote metrics:     {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(json.dumps(metrics, indent=2))

    if args.use_lora and args.save_merged_model and hasattr(trainer.model, "merge_and_unload"):
        merged_dir = output_dir / "merged-model"
        print(f"Merging LoRA adapters into standalone model at {merged_dir}")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
        feature_extractor.save_pretrained(str(merged_dir))
        print(f"Wrote merged model: {merged_dir}")


if __name__ == "__main__":
    main()
