#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from _dictionary_pipeline_common import (
    DEZD_TSV_PATH,
    TSV_OUTPUT_DIR,
    clean_cell,
    compact_text_list,
    ensure_parent_dir,
    normalize_search_text,
    normalize_spaces,
    write_json,
)


HARUN_TM_TSV_PATH = TSV_OUTPUT_DIR / "harun_translation_memory.tsv"
HARUN_REVIEW_TSV_PATH = TSV_OUTPUT_DIR / "harun_entries_review.tsv"
CACHE_PATH = TSV_OUTPUT_DIR / "mt_work" / "definition_mt_cache.jsonl"
SUMMARY_PATH = TSV_OUTPUT_DIR / "mt_work" / "definition_mt_summary.json"

MODEL_TR_EN_DEFAULT = "Helsinki-NLP/opus-mt-tr-en"
MODEL_EN_DE_DEFAULT = "Helsinki-NLP/opus-mt-en-de"
PACKAGE_DIR_DEFAULT = Path("$HOME/.python_mt")

LONG_SPLIT_DELIMS = (" | ", "; ")
VERY_LONG_SPLIT_RE = re.compile(r"(?<=[,)])\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing EN/DE dictionary definitions from existing Turkish definitions using a local MT model.")
    parser.add_argument("--harun-tm-path", default=str(HARUN_TM_TSV_PATH))
    parser.add_argument("--harun-review-path", default=str(HARUN_REVIEW_TSV_PATH))
    parser.add_argument("--dezd-path", default=str(DEZD_TSV_PATH))
    parser.add_argument("--cache-path", default=str(CACHE_PATH))
    parser.add_argument("--summary-path", default=str(SUMMARY_PATH))
    parser.add_argument("--local-package-dir", default=str(PACKAGE_DIR_DEFAULT))
    parser.add_argument("--model-tr-en", default=MODEL_TR_EN_DEFAULT)
    parser.add_argument("--model-en-de", default=MODEL_EN_DE_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--max-input-tokens", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--backup-suffix", default=".pre_mt_fill_20260416.bak")
    return parser.parse_args()


def add_local_package_dir(path: Path) -> None:
    if path.exists():
        sys.path.insert(0, str(path))


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows([{name: clean_cell(row.get(name, "")) for name in fieldnames} for row in rows])


def ensure_fields(fieldnames: list[str], extra_fields: list[str]) -> list[str]:
    output = list(fieldnames)
    for name in extra_fields:
        if name not in output:
            output.append(name)
    return output


def reseed_harun_translation_memory(
    review_rows: list[dict[str, str]],
    existing_tm_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    existing_by_key = {
        clean_cell(row.get("tm_key")): row
        for row in existing_tm_rows
        if clean_cell(row.get("tm_key"))
    }
    buckets: dict[str, dict[str, object]] = {}
    for row in review_rows:
        definition = clean_cell(row.get("definition_tr_display")) or clean_cell(row.get("definition_tr_raw"))
        if not definition:
            continue
        tm_key = normalize_search_text(definition)
        if not tm_key:
            continue
        existing = existing_by_key.get(tm_key, {})
        bucket = buckets.setdefault(
            tm_key,
            {
                "tm_key": tm_key,
                "definition_tr": definition,
                "definition_en": clean_cell(existing.get("definition_en")),
                "definition_de": clean_cell(existing.get("definition_de")),
                "translation_source_en": clean_cell(existing.get("translation_source_en")),
                "translation_source_de": clean_cell(existing.get("translation_source_de")),
                "mt_model_tr_en": clean_cell(existing.get("mt_model_tr_en")),
                "mt_model_en_de": clean_cell(existing.get("mt_model_en_de")),
                "mt_generated_at": clean_cell(existing.get("mt_generated_at")),
                "mt_review_status": clean_cell(existing.get("mt_review_status")),
                "entry_count": 0,
                "verified_entry_count": 0,
                "entry_ids": [],
            },
        )
        bucket["entry_count"] = int(bucket["entry_count"]) + 1
        if clean_cell(row.get("review_status")) == "verified_pdf":
            bucket["verified_entry_count"] = int(bucket["verified_entry_count"]) + 1
        entry_ids = bucket["entry_ids"]
        assert isinstance(entry_ids, list)
        entry_ids.append(clean_cell(row.get("entry_id")))

    rows: list[dict[str, str]] = []
    for tm_key, bucket in buckets.items():
        entry_ids = bucket.pop("entry_ids")
        assert isinstance(entry_ids, list)
        rows.append(
            {
                **bucket,
                "tm_key": tm_key,
                "entry_ids_preview": compact_text_list(entry_ids, limit=8, joiner=" | "),
                "notes": "Reseeded from current harun_entries_review.tsv before MT fill",
            }
        )
    rows.sort(key=lambda row: (-int(clean_cell(row.get("entry_count")) or "0"), clean_cell(row.get("tm_key"))))
    return rows


def backup_once(path: Path, suffix: str) -> Path:
    backup_path = path.with_name(path.name + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_cache(path: Path) -> dict[tuple[str, str], str]:
    cache: dict[tuple[str, str], str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[(record["target_lang"], record["source_text"])] = record["translated_text"]
    return cache


def append_cache(path: Path, records: list[dict[str, str]]) -> None:
    if not records:
        return
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunk_by_length(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def split_long_text(text: str, max_chars: int) -> list[str]:
    normalized = clean_cell(text)
    if len(normalized) <= max_chars:
        return [normalized]
    for delimiter in LONG_SPLIT_DELIMS:
        if delimiter in normalized:
            parts = [clean_cell(part) for part in normalized.split(delimiter)]
            if len(parts) > 1:
                return [part for item in parts for part in split_long_text(item, max_chars) if part]
    parts = [clean_cell(part) for part in VERY_LONG_SPLIT_RE.split(normalized) if clean_cell(part)]
    if len(parts) > 1:
        output: list[str] = []
        for part in parts:
            output.extend(split_long_text(part, max_chars))
        return output
    return chunk_by_length(normalized, max_chars)


@dataclass
class Translator:
    model_id: str
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    num_beams: int

    def __post_init__(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    def translate_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        return [normalize_spaces(text) for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)]


def batched(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def build_translation_cache(
    translator: Translator,
    cache_path: Path,
    cache: dict[tuple[str, str], str],
    texts: set[str],
    *,
    source_lang: str,
    target_lang: str,
    max_chars: int,
    generated_at: str,
) -> Counter:
    stats: Counter = Counter()
    total_direct = len([text for text in texts if len(text) <= max_chars and (target_lang, text) not in cache])

    direct_texts = sorted(
        [text for text in texts if len(text) <= max_chars and (target_lang, text) not in cache],
        key=len,
    )
    pending_records: list[dict[str, str]] = []
    done_direct = 0
    total_batches = max(1, len(direct_texts) // translator.batch_size + (1 if len(direct_texts) % translator.batch_size else 0))
    for batch_index, batch in enumerate(batched(direct_texts, translator.batch_size), start=1):
        outputs = translator.translate_batch(batch)
        for source_text, translated_text in zip(batch, outputs, strict=True):
            cache[(target_lang, source_text)] = translated_text
            pending_records.append(
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_hash": text_hash(source_text),
                    "source_text": source_text,
                    "translated_text": translated_text,
                    "model_id": translator.model_id,
                    "generated_at": generated_at,
                }
            )
            stats["new_cached"] += 1
        append_cache(cache_path, pending_records)
        pending_records.clear()
        done_direct += len(batch)
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == total_batches:
            print(
                f"[{source_lang}->{target_lang}] short-text batches {batch_index}/{total_batches} "
                f"({done_direct}/{total_direct} new cached)",
                flush=True,
            )

    long_texts = sorted([text for text in texts if len(text) > max_chars], key=len, reverse=True)
    for index, text in enumerate(long_texts, start=1):
        normalized = clean_cell(text)
        if (target_lang, normalized) in cache:
            continue
        parts = [part for part in split_long_text(normalized, max_chars) if clean_cell(part)]
        uncached_parts = sorted({part for part in parts if (target_lang, part) not in cache}, key=len)
        if uncached_parts:
            total_part_batches = max(1, len(uncached_parts) // translator.batch_size + (1 if len(uncached_parts) % translator.batch_size else 0))
            done_parts = 0
            for batch_index, batch in enumerate(batched(uncached_parts, translator.batch_size), start=1):
                outputs = translator.translate_batch(batch)
                records: list[dict[str, str]] = []
                for source_text, translated_text in zip(batch, outputs, strict=True):
                    cache[(target_lang, source_text)] = translated_text
                    records.append(
                        {
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "source_hash": text_hash(source_text),
                            "source_text": source_text,
                            "translated_text": translated_text,
                            "model_id": translator.model_id,
                            "generated_at": generated_at,
                        }
                    )
                    stats["new_cached"] += 1
                append_cache(cache_path, records)
                done_parts += len(batch)
                if batch_index == 1 or batch_index % 25 == 0 or batch_index == total_part_batches:
                    print(
                        f"[{source_lang}->{target_lang}] long-text parts {index}/{len(long_texts)} "
                        f"batch {batch_index}/{total_part_batches} ({done_parts}/{len(uncached_parts)} new part translations)",
                        flush=True,
                    )
        translated = " | ".join(cache[(target_lang, part)] for part in parts)
        cache[(target_lang, normalized)] = translated
        append_cache(
            cache_path,
            [
                {
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_hash": text_hash(normalized),
                    "source_text": normalized,
                    "translated_text": translated,
                    "model_id": translator.model_id,
                    "generated_at": generated_at,
                }
            ],
        )
        stats["new_cached"] += 1
        print(f"[{source_lang}->{target_lang}] long texts {index}/{len(long_texts)} cached", flush=True)

    stats["total_available"] = sum(1 for text in texts if (target_lang, text) in cache)
    return stats


def collect_needed_turkish_texts(harun_tm_rows: list[dict[str, str]], dezd_rows: list[dict[str, str]]) -> set[str]:
    texts: set[str] = set()
    for row in harun_tm_rows:
        definition_tr = clean_cell(row.get("definition_tr"))
        if definition_tr and (not clean_cell(row.get("definition_en")) or not clean_cell(row.get("definition_de"))):
            texts.add(definition_tr)
    for row in dezd_rows:
        definition_tr = clean_cell(row.get("definition_full_tr")) or clean_cell(row.get("definition"))
        if definition_tr and (not clean_cell(row.get("definition_full_en")) or not clean_cell(row.get("definition_full_de"))):
            texts.add(definition_tr)
    return texts


def collect_needed_english_texts(harun_tm_rows: list[dict[str, str]], dezd_rows: list[dict[str, str]], en_cache: dict[tuple[str, str], str]) -> set[str]:
    texts: set[str] = set()
    for row in harun_tm_rows:
        if clean_cell(row.get("definition_de")):
            continue
        source_en = clean_cell(row.get("definition_en"))
        if not source_en:
            definition_tr = clean_cell(row.get("definition_tr"))
            source_en = en_cache.get(("en", definition_tr), "")
        if source_en:
            texts.add(source_en)
    for row in dezd_rows:
        if clean_cell(row.get("definition_full_de")):
            continue
        source_en = clean_cell(row.get("definition_full_en"))
        if not source_en:
            definition_tr = clean_cell(row.get("definition_full_tr")) or clean_cell(row.get("definition"))
            source_en = en_cache.get(("en", definition_tr), "")
        if source_en:
            texts.add(source_en)
    return texts


def update_harun_tm_rows(
    rows: list[dict[str, str]],
    en_cache: dict[tuple[str, str], str],
    de_cache: dict[tuple[str, str], str],
    model_tr_en: str,
    model_en_de: str,
    generated_at: str,
) -> Counter:
    stats: Counter = Counter()
    for row in rows:
        definition_tr = clean_cell(row.get("definition_tr"))
        if not definition_tr:
            stats["rows_without_tr"] += 1
            continue
        if not clean_cell(row.get("definition_en")):
            row["definition_en"] = en_cache[("en", definition_tr)]
            row["translation_source_en"] = "machine_translated_tr_en"
            row["mt_model_tr_en"] = model_tr_en
            row["mt_generated_at"] = generated_at
            row["mt_review_status"] = "needs_review"
            stats["filled_en"] += 1
        if not clean_cell(row.get("definition_de")):
            source_en = clean_cell(row.get("definition_en")) or en_cache[("en", definition_tr)]
            row["definition_de"] = de_cache[("de", source_en)]
            row["translation_source_de"] = "machine_translated_en_de"
            row["mt_model_en_de"] = model_en_de
            row["mt_generated_at"] = generated_at
            row["mt_review_status"] = "needs_review"
            stats["filled_de"] += 1
    return stats


def append_note(existing: str, note: str) -> str:
    current = clean_cell(existing)
    return note if not current else f"{current} | {note}"


def update_dezd_rows(
    rows: list[dict[str, str]],
    en_cache: dict[tuple[str, str], str],
    de_cache: dict[tuple[str, str], str],
    model_tr_en: str,
    model_en_de: str,
    generated_at: str,
) -> Counter:
    stats: Counter = Counter()
    for row in rows:
        fallback_tr = clean_cell(row.get("definition"))
        definition_tr = clean_cell(row.get("definition_full_tr"))
        if not definition_tr and fallback_tr:
            row["definition_full_tr"] = fallback_tr
            row["qa_notes"] = append_note(row.get("qa_notes", ""), "promoted fallback Turkish definition into definition_full_tr before MT fill")
            definition_tr = fallback_tr
            stats["filled_tr_from_fallback"] += 1
        if not definition_tr:
            stats["rows_without_tr"] += 1
            continue
        if not clean_cell(row.get("definition_full_en")):
            row["definition_full_en"] = en_cache[("en", definition_tr)]
            row["definition_full_en_source"] = "machine_translated_tr_en"
            row["definition_full_en_mt_model"] = model_tr_en
            row["definition_full_en_review_status"] = "needs_review"
            row["mt_generated_at"] = generated_at
            stats["filled_en"] += 1
        if not clean_cell(row.get("definition_full_de")):
            source_en = clean_cell(row.get("definition_full_en")) or en_cache[("en", definition_tr)]
            row["definition_full_de"] = de_cache[("de", source_en)]
            row["definition_full_de_source"] = "machine_translated_en_de"
            row["definition_full_de_mt_model"] = model_en_de
            row["definition_full_de_review_status"] = "needs_review"
            row["mt_generated_at"] = generated_at
            stats["filled_de"] += 1
    return stats


def main() -> None:
    args = parse_args()
    add_local_package_dir(Path(args.local_package_dir))

    harun_tm_path = Path(args.harun_tm_path)
    harun_review_path = Path(args.harun_review_path)
    dezd_path = Path(args.dezd_path)
    cache_path = Path(args.cache_path)
    summary_path = Path(args.summary_path)

    harun_tm_fields, existing_harun_tm_rows = read_tsv(harun_tm_path)
    _, harun_review_rows = read_tsv(harun_review_path)
    dezd_fields, dezd_rows = read_tsv(dezd_path)
    harun_tm_fields = ensure_fields(
        harun_tm_fields,
        [
            "tm_key",
            "definition_tr",
            "definition_en",
            "definition_de",
            "translation_source_en",
            "translation_source_de",
            "mt_model_tr_en",
            "mt_model_en_de",
            "mt_generated_at",
            "mt_review_status",
            "entry_count",
            "verified_entry_count",
            "entry_ids_preview",
            "notes",
        ],
    )
    harun_tm_rows = reseed_harun_translation_memory(harun_review_rows, existing_harun_tm_rows)
    preserved_exact_tm_matches = sum(
        1
        for row in harun_tm_rows
        if clean_cell(row.get("definition_en")) or clean_cell(row.get("definition_de"))
    )

    needed_texts = collect_needed_turkish_texts(harun_tm_rows, dezd_rows)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cache = load_cache(cache_path)
    print(f"Loaded cache entries: {len(cache)}", flush=True)

    tr_en_translator = Translator(
        model_id=args.model_tr_en,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(f"Loaded translator tr->en: {args.model_tr_en}", flush=True)
    en_de_translator = Translator(
        model_id=args.model_en_de,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(f"Loaded translator en->de: {args.model_en_de}", flush=True)
    print(f"Unique Turkish texts needing EN/DE: {len(needed_texts)}", flush=True)

    translation_stats = {
        "tr_en": build_translation_cache(
            tr_en_translator,
            cache_path,
            cache,
            needed_texts,
            source_lang="tr",
            target_lang="en",
            max_chars=args.max_chars,
            generated_at=generated_at,
        )
    }
    needed_english_texts = collect_needed_english_texts(harun_tm_rows, dezd_rows, cache)
    print(f"Unique English texts needing DE: {len(needed_english_texts)}", flush=True)
    translation_stats["en_de"] = build_translation_cache(
        en_de_translator,
        cache_path,
        cache,
        needed_english_texts,
        source_lang="en",
        target_lang="de",
        max_chars=args.max_chars,
        generated_at=generated_at,
    )

    harun_backup = backup_once(harun_tm_path, args.backup_suffix)
    dezd_backup = backup_once(dezd_path, args.backup_suffix)

    dezd_fields = ensure_fields(
        dezd_fields,
        [
            "definition_full_en_source",
            "definition_full_en_mt_model",
            "definition_full_en_review_status",
            "definition_full_de_source",
            "definition_full_de_mt_model",
            "definition_full_de_review_status",
            "mt_generated_at",
        ],
    )

    harun_stats = update_harun_tm_rows(harun_tm_rows, cache, cache, args.model_tr_en, args.model_en_de, generated_at)
    dezd_stats = update_dezd_rows(dezd_rows, cache, cache, args.model_tr_en, args.model_en_de, generated_at)

    write_tsv(harun_tm_path, harun_tm_fields, harun_tm_rows)
    write_tsv(dezd_path, dezd_fields, dezd_rows)

    summary = {
        "model_tr_en": args.model_tr_en,
        "model_en_de": args.model_en_de,
        "generated_at": generated_at,
        "cache_path": str(cache_path),
        "harun_review_path": str(harun_review_path),
        "harun_tm_path": str(harun_tm_path),
        "dezd_path": str(dezd_path),
        "backups": {
            "harun_tm": str(harun_backup),
            "dezd": str(dezd_backup),
        },
        "harun_tm_reseed": {
            "row_count": len(harun_tm_rows),
            "preserved_exact_tm_matches": preserved_exact_tm_matches,
        },
        "needed_unique_turkish_texts": len(needed_texts),
        "needed_unique_english_texts_for_german": len(needed_english_texts),
        "translation_stats": {target_lang: dict(stats) for target_lang, stats in translation_stats.items()},
        "harun_stats": dict(harun_stats),
        "dezd_stats": dict(dezd_stats),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
