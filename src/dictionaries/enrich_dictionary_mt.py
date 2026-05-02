#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import ctranslate2
import sentencepiece as spm
from huggingface_hub import snapshot_download

from _dictionary_pipeline_common import TSV_OUTPUT_DIR, clean_cell, ensure_parent_dir, tsv_rows


DEFAULT_TM_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
DEFAULT_CACHE_DIR = TSV_OUTPUT_DIR / "model_cache"
TR_EN_MODEL = "quickmt/quickmt-tr-en"
EN_DE_MODEL = "quickmt/quickmt-en-de"


@dataclass
class QuickMTModel:
    repo_id: str
    local_dir: Path
    translator: ctranslate2.Translator
    src_spm: spm.SentencePieceProcessor
    tgt_spm: spm.SentencePieceProcessor

    @classmethod
    def load(cls, repo_id: str, cache_dir: Path, device: str) -> "QuickMTModel":
        local_dir = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                cache_dir=str(cache_dir),
                ignore_patterns=["eole-model/*", ".ipynb_checkpoints/*"],
            )
        )
        translator = ctranslate2.Translator(str(local_dir), device=device)
        src_spm = spm.SentencePieceProcessor(model_file=str(local_dir / "src.spm.model"))
        tgt_spm = spm.SentencePieceProcessor(model_file=str(local_dir / "tgt.spm.model"))
        return cls(repo_id=repo_id, local_dir=local_dir, translator=translator, src_spm=src_spm, tgt_spm=tgt_spm)

    def translate_batch(self, texts: list[str], beam_size: int) -> list[str]:
        token_batches = [self.src_spm.encode(text, out_type=str) for text in texts]
        results = self.translator.translate_batch(token_batches, beam_size=beam_size)
        outputs: list[str] = []
        for result in results:
            best = result.hypotheses[0] if result.hypotheses else []
            filtered = [token for token in best if token and not token.startswith("<")]
            outputs.append(clean_cell(self.tgt_spm.decode(filtered)))
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill Harun translation memory with local machine translations.")
    parser.add_argument("--input-path", default=str(DEFAULT_TM_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_TM_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Only translate the first N eligible rows.")
    parser.add_argument("--force", action="store_true", help="Recompute English/German even if the row already has them.")
    parser.add_argument("--download-only", action="store_true", help="Download the MT models and exit.")
    return parser.parse_args()


def load_fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or [])


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fieldnames})


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    cache_dir = Path(args.cache_dir)

    tr_en_model = QuickMTModel.load(TR_EN_MODEL, cache_dir, args.device)
    en_de_model = QuickMTModel.load(EN_DE_MODEL, cache_dir, args.device)

    if args.download_only:
        print(f"Downloaded models to:\n- {tr_en_model.local_dir}\n- {en_de_model.local_dir}")
        return

    rows = tsv_rows(input_path)
    fieldnames = load_fieldnames(input_path)
    for field in [
        "definition_en",
        "definition_de",
        "translation_source_en",
        "translation_source_de",
        "mt_model_tr_en",
        "mt_model_en_de",
        "mt_generated_at",
        "mt_review_status",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    eligible_indexes: list[int] = []
    tr_inputs: list[str] = []
    for index, row in enumerate(rows):
        definition_tr = clean_cell(row.get("definition_tr"))
        definition_en = clean_cell(row.get("definition_en"))
        definition_de = clean_cell(row.get("definition_de"))
        needs_en = args.force or not definition_en
        needs_de = args.force or not definition_de
        if definition_tr and (needs_en or needs_de):
            eligible_indexes.append(index)
            tr_inputs.append(definition_tr)
        if args.limit and len(eligible_indexes) >= args.limit:
            break

    if not eligible_indexes:
        print("No eligible translation-memory rows found.")
        return

    en_outputs = tr_en_model.translate_batch(tr_inputs, beam_size=args.beam_size)
    de_outputs = en_de_model.translate_batch(en_outputs, beam_size=args.beam_size)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    for row_index, definition_en, definition_de in zip(eligible_indexes, en_outputs, de_outputs):
        row = rows[row_index]
        if args.force or not clean_cell(row.get("definition_en")):
            row["definition_en"] = definition_en
            row["translation_source_en"] = f"machine_translation:{TR_EN_MODEL}"
        if args.force or not clean_cell(row.get("definition_de")):
            row["definition_de"] = definition_de
            row["translation_source_de"] = f"machine_translation:{EN_DE_MODEL}"
        row["mt_model_tr_en"] = TR_EN_MODEL
        row["mt_model_en_de"] = EN_DE_MODEL
        row["mt_generated_at"] = generated_at
        row["mt_review_status"] = "pending_review"

    write_rows(output_path, fieldnames, rows)
    print(
        "\n".join(
            [
                f"Updated translation memory: {output_path}",
                f"Rows translated: {len(eligible_indexes)}",
                f"tr->en model: {TR_EN_MODEL}",
                f"en->de model: {EN_DE_MODEL}",
            ]
        )
    )


if __name__ == "__main__":
    main()
