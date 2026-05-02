#!/usr/bin/env python
"""Evaluate a local VibeVoice ASR LoRA adapter on a VibeVoice-format split."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, 1):
        current = [i] + [0] * len(right)
        for j, right_item in enumerate(right, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_item != right_item),
            )
        previous = current
    return previous[-1]


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^\w\sêÊîÎûÛşŞçÇğĞıİöÖüÜâÂ']", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def collect_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


def load_sample(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    audio_path = json_path.parent / data["audio_path"]
    context = data.get("customized_context") or []
    if isinstance(context, list):
        context_info = "\n".join(str(line) for line in context if str(line).strip())
    else:
        context_info = str(context)
    return {
        "json_path": json_path,
        "audio_path": audio_path,
        "reference": collect_text(data.get("segments", [])),
        "context_info": context_info or None,
    }


def load_model(
    base_model: str,
    lora_path: str,
    attn_implementation: str,
    load_in_4bit: bool,
) -> tuple[Any, VibeVoiceASRProcessor, torch.device]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    processor = VibeVoiceASRProcessor.from_pretrained(
        base_model,
        language_model_pretrained_name="Qwen/Qwen2.5-7B",
    )
    quantization_config = None
    device_map = None
    if torch.cuda.is_available():
        device_map = "auto"
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = "auto"

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if device_map is None:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    device = next(model.parameters()).device
    return model, processor, device


def transcribe(
    model: Any,
    processor: VibeVoiceASRProcessor,
    device: torch.device,
    audio_path: Path,
    context_info: str | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    inputs = processor(
        audio=str(audio_path),
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=context_info,
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.pad_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            do_sample=False,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    raw_text = processor.decode(generated_ids, skip_special_tokens=True)
    try:
        segments = processor.post_process_transcription(raw_text)
    except Exception:
        segments = []
    return {
        "raw_text": raw_text,
        "segments": segments,
        "prediction": collect_text(segments) if segments else raw_text.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-samples", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    json_paths = sorted(eval_dir.glob("*.json"))
    if args.max_samples > 0:
        json_paths = json_paths[: args.max_samples]
    samples = [load_sample(path) for path in json_paths]
    if not samples:
        raise SystemExit(f"No eval samples found in {eval_dir}")

    started = time.time()
    model, processor, device = load_model(
        base_model=args.base_model,
        lora_path=args.lora_path,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    word_errors = 0
    ref_words = 0
    norm_word_errors = 0
    norm_ref_words = 0
    char_errors = 0
    ref_chars = 0
    rows = []

    with output_path.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(samples, 1):
            result = transcribe(
                model=model,
                processor=processor,
                device=device,
                audio_path=sample["audio_path"],
                context_info=sample["context_info"],
                max_new_tokens=args.max_new_tokens,
            )
            reference = sample["reference"]
            prediction = result["prediction"]
            ref_word_list = reference.split()
            pred_word_list = prediction.split()
            norm_reference = normalize_text(reference)
            norm_prediction = normalize_text(prediction)
            norm_ref_word_list = norm_reference.split()
            norm_pred_word_list = norm_prediction.split()

            row_word_errors = edit_distance(ref_word_list, pred_word_list)
            row_norm_word_errors = edit_distance(norm_ref_word_list, norm_pred_word_list)
            row_char_errors = edit_distance(list(reference), list(prediction))

            word_errors += row_word_errors
            ref_words += len(ref_word_list)
            norm_word_errors += row_norm_word_errors
            norm_ref_words += len(norm_ref_word_list)
            char_errors += row_char_errors
            ref_chars += len(reference)

            row = {
                "index": index,
                "json_path": str(sample["json_path"]),
                "audio_path": str(sample["audio_path"]),
                "reference": reference,
                "prediction": prediction,
                "raw_text": result["raw_text"],
                "segments": result["segments"],
                "word_errors": row_word_errors,
                "reference_words": len(ref_word_list),
                "normalized_word_errors": row_norm_word_errors,
                "normalized_reference_words": len(norm_ref_word_list),
                "char_errors": row_char_errors,
                "reference_chars": len(reference),
            }
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"[{index}/{len(samples)}] WER={row_word_errors}/{max(len(ref_word_list), 1)} "
                f"REF={reference} | HYP={prediction}",
                flush=True,
            )

    summary = {
        "samples": len(rows),
        "word_errors": word_errors,
        "reference_words": ref_words,
        "wer": word_errors / ref_words if ref_words else None,
        "normalized_word_errors": norm_word_errors,
        "normalized_reference_words": norm_ref_words,
        "normalized_wer": norm_word_errors / norm_ref_words if norm_ref_words else None,
        "char_errors": char_errors,
        "reference_chars": ref_chars,
        "cer": char_errors / ref_chars if ref_chars else None,
        "runtime_seconds": round(time.time() - started, 3),
        "output_jsonl": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
