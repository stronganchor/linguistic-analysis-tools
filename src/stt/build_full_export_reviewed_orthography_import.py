#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from build_recording_text_orthography_import import (
    collect_output_review_flags,
    collect_review_flags,
    normalize_space,
    orthography_from_ipa,
)


NUMERIC_TITLE_RE = re.compile(r"[-+]?(?:\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    zip_dir = script_dir.parent / "zazacaogren" / "zip files"
    default_zip = choose_latest_full_export_zip(zip_dir)
    default_output_dir = script_dir.parent / "zazacaogren" / "orthography-text-fill-april-2026"
    default_prefix = (
        f"{default_zip.stem}-reviewed-ipa-orthography-import"
        if default_zip is not None
        else "genc-palu-full-export-reviewed-ipa-orthography-import"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build an LL Tools metadata import from the latest full export zip, "
            "updating recording_text for reviewed recording_ipa rows and backfilling "
            "word_title from reviewed isolation transcriptions when the current title "
            "looks like a placeholder."
        )
    )
    parser.add_argument(
        "--full-export-zip",
        default=str(default_zip) if default_zip is not None else "",
        help="Path to a full LL Tools export zip that contains data.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Folder for import, audit, and summary output.",
    )
    parser.add_argument(
        "--output-prefix",
        default=default_prefix,
        help="Prefix for generated files.",
    )
    return parser.parse_args()


def choose_latest_full_export_zip(zip_dir: Path) -> Path | None:
    if not zip_dir.is_dir():
        return None
    paths = sorted(zip_dir.glob("ll-tools-export-full-*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0] if paths else None


def first_meta_value(meta: dict[str, object] | None, key: str) -> str:
    if not meta:
        return ""
    raw = meta.get(key)
    if isinstance(raw, list):
        for item in raw:
            value = normalize_space(str(item))
            if value:
                return value
        return ""
    return normalize_space(str(raw or ""))


def meta_flag_is_truthy(meta: dict[str, object] | None, key: str) -> bool:
    if not meta:
        return False
    raw = meta.get(key)
    if raw is None:
        return False
    values = raw if isinstance(raw, list) else [raw]
    for item in values:
        if item is None:
            continue
        value = normalize_space(str(item)).casefold()
        if value and value not in {"0", "false", "no", "off"}:
            return True
    return False


def dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def title_backfill_reason(word_title: str, word_slug: str) -> str:
    title = normalize_space(word_title)
    slug = normalize_space(word_slug)
    if not title:
        return "blank"
    if NUMERIC_TITLE_RE.fullmatch(title):
        return "numeric"
    if slug and title.casefold() == slug.casefold():
        return "matches_slug"
    return ""


def choose_title_source_row(rows: list[dict[str, object]]) -> dict[str, object] | None:
    isolation_rows = [row for row in rows if row["recording_type"] == "isolation"]
    if not isolation_rows:
        return None
    return sorted(isolation_rows, key=lambda row: int(str(row["recording_id"])))[0]


def ensure_update_row(
    updates_by_key: dict[tuple[int, int], dict[str, object]],
    audits_by_key: dict[tuple[int, int], dict[str, object]],
    row: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    key = (int(str(row["word_id"])), int(str(row["recording_id"])))
    update_row = updates_by_key.setdefault(
        key,
        {
            "word_id": key[0],
            "recording_id": key[1],
        },
    )
    audit_row = audits_by_key.setdefault(
        key,
        {
            "word_id": key[0],
            "recording_id": key[1],
            "word_slug": str(row["word_slug"]),
            "recording_type": str(row["recording_type"]),
            "recording_ipa": str(row["recording_ipa"]),
            "old_recording_text": "",
            "new_recording_text": "",
            "old_word_title": str(row["word_title"]),
            "new_word_title": "",
            "word_title_reason": "",
        },
    )
    return update_row, audit_row


def main() -> None:
    args = parse_args()
    if not args.full_export_zip:
        raise SystemExit("No full export zip provided and none was found in Z/zazacaogren/zip files.")

    full_export_zip = Path(args.full_export_zip).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(full_export_zip) as archive:
        payload = json.loads(archive.read("data.json"))

    words = payload.get("words") or []
    summary = Counter()
    clean_rows_by_word: dict[str, list[dict[str, object]]] = defaultdict(list)
    flagged_rows: list[dict[str, object]] = []
    updates_by_key: dict[tuple[int, int], dict[str, object]] = {}
    audits_by_key: dict[tuple[int, int], dict[str, object]] = {}

    for word in words:
        word_id = normalize_space(str(word.get("origin_id") or ""))
        word_title = normalize_space(str(word.get("title") or ""))
        word_slug = normalize_space(str(word.get("slug") or ""))
        audio_entries = word.get("audio_entries") or []
        summary["words_seen"] += 1

        for audio_entry in audio_entries:
            summary["audio_entries_seen"] += 1
            entry_meta = audio_entry.get("meta") or {}
            recording_id = normalize_space(str(audio_entry.get("origin_id") or ""))
            recording_ipa = first_meta_value(entry_meta, "recording_ipa")
            if not recording_ipa:
                continue

            summary["rows_with_recording_ipa"] += 1
            if meta_flag_is_truthy(entry_meta, "ll_auto_transcription_needs_review"):
                summary["unreviewed_rows_skipped"] += 1
                continue

            recording_types = audio_entry.get("recording_types") or []
            recording_type = normalize_space(str(recording_types[0] if recording_types else "unknown"))
            recording_text = first_meta_value(entry_meta, "recording_text")
            generated_text = orthography_from_ipa(recording_ipa, recording_type)
            review_flags = dedupe_preserve_order(
                collect_review_flags(recording_ipa) + collect_output_review_flags(generated_text)
            )

            row = {
                "word_id": word_id,
                "word_title": word_title,
                "word_slug": word_slug,
                "recording_id": recording_id,
                "recording_type": recording_type,
                "recording_ipa": recording_ipa,
                "recording_text": recording_text,
                "generated_recording_text": generated_text,
                "review_flags": review_flags,
            }
            summary["reviewed_rows_seen"] += 1

            if review_flags:
                flagged_rows.append(
                    {
                        "word_id": int(word_id),
                        "recording_id": int(recording_id),
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_type": recording_type,
                        "recording_ipa": recording_ipa,
                        "current_recording_text": recording_text,
                        "generated_recording_text": generated_text,
                        "review_flags": review_flags,
                    }
                )
                summary["flagged_rows_excluded"] += 1
                for flag in review_flags:
                    summary[f"flag_{flag}"] += 1
                continue

            clean_rows_by_word[word_id].append(row)

            if generated_text != recording_text:
                update_row, audit_row = ensure_update_row(updates_by_key, audits_by_key, row)
                update_row["recording_text"] = generated_text
                audit_row["old_recording_text"] = recording_text
                audit_row["new_recording_text"] = generated_text
                summary["recording_text_updates"] += 1
                if recording_text:
                    summary["recording_text_corrected"] += 1
                else:
                    summary["recording_text_filled"] += 1

    for word in words:
        word_id = normalize_space(str(word.get("origin_id") or ""))
        word_title = normalize_space(str(word.get("title") or ""))
        word_slug = normalize_space(str(word.get("slug") or ""))
        clean_rows = clean_rows_by_word.get(word_id) or []
        if not clean_rows:
            continue

        backfill_reason = title_backfill_reason(word_title, word_slug)
        if not backfill_reason:
            continue

        title_source_row = choose_title_source_row(clean_rows)
        if title_source_row is None:
            continue

        new_word_title = str(title_source_row["generated_recording_text"])
        if not new_word_title or new_word_title == word_title:
            continue

        update_row, audit_row = ensure_update_row(updates_by_key, audits_by_key, title_source_row)
        update_row["word_title"] = new_word_title
        audit_row["new_word_title"] = new_word_title
        audit_row["word_title_reason"] = backfill_reason
        summary["word_title_updates"] += 1
        summary[f"word_title_updates_{backfill_reason}"] += 1

    prefix = args.output_prefix
    import_rows = [updates_by_key[key] for key in sorted(updates_by_key)]
    audit_rows = [audits_by_key[key] for key in sorted(audits_by_key)]

    summary_data = {
        "source_full_export_zip": str(full_export_zip),
        "bundle_exported_at": payload.get("exported_at"),
        "total_words": summary["words_seen"],
        "total_audio_entries": summary["audio_entries_seen"],
        "rows_with_recording_ipa": summary["rows_with_recording_ipa"],
        "reviewed_rows_seen": summary["reviewed_rows_seen"],
        "unreviewed_rows_skipped": summary["unreviewed_rows_skipped"],
        "flagged_rows_excluded": summary["flagged_rows_excluded"],
        "import_rows": len(import_rows),
        "recording_text_updates": summary["recording_text_updates"],
        "recording_text_filled": summary["recording_text_filled"],
        "recording_text_corrected": summary["recording_text_corrected"],
        "word_title_updates": summary["word_title_updates"],
        "word_title_updates_blank": summary["word_title_updates_blank"],
        "word_title_updates_numeric": summary["word_title_updates_numeric"],
        "word_title_updates_matches_slug": summary["word_title_updates_matches_slug"],
        "word_title_backfill_rule": "blank_or_numeric_or_case_insensitive_slug_match",
        "flag_initial_ʔ_manual_review": summary["flag_initial_ʔ_manual_review"],
        "flag_contains_ʊ_manual_review": summary["flag_contains_ʊ_manual_review"],
        "flag_contains_short_vowel_manual_review": summary["flag_contains_short_vowel_manual_review"],
        "flag_contains_epenthetic_vowel_manual_review": summary["flag_contains_epenthetic_vowel_manual_review"],
        "flag_non_prevelar_ŋ_manual_review": summary["flag_non_prevelar_ŋ_manual_review"],
        "flag_contains_residual_ipa_output_manual_review": summary["flag_contains_residual_ipa_output_manual_review"],
    }

    import_path = output_dir / f"{prefix}.jsonl"
    audit_path = output_dir / f"{prefix}.audit.jsonl"
    flagged_path = output_dir / f"{prefix}.review_flags.jsonl"
    summary_path = output_dir / f"{prefix}.summary.json"

    with import_path.open("w", encoding="utf-8") as handle:
        for row in import_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with flagged_path.open("w", encoding="utf-8") as handle:
        for row in flagged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote import:  {import_path}")
    print(f"Wrote audit:   {audit_path}")
    print(f"Wrote flags:   {flagged_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
