#!/usr/bin/env python3
import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from build_full_export_reviewed_orthography_import import (
    choose_latest_full_export_zip,
    first_meta_value,
    meta_flag_is_truthy,
    title_backfill_reason,
)
from build_recording_text_orthography_import import (
    collect_output_review_flags,
    collect_review_flags,
    normalize_space,
    orthography_from_ipa,
)


EXPECTED_RECORDING_TYPES = ("isolation", "introduction", "question")
SAFE_TITLE_BACKFILL_REASONS = {"blank", "numeric"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    zip_dir = script_dir.parent / "zazacaogren" / "zip files"
    default_zip = choose_latest_full_export_zip(zip_dir)
    default_output_dir = script_dir.parent / "zazacaogren" / "general-qa-april-2026"
    default_prefix = (
        f"{default_zip.stem}-general-qa"
        if default_zip is not None
        else "genc-palu-full-export-general-qa"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Run a conservative QA pass over the latest full LL Tools export, "
            "writing manual-review buckets plus a separate high-confidence import file."
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
        help="Folder for QA output files.",
    )
    parser.add_argument(
        "--output-prefix",
        default=default_prefix,
        help="Prefix for generated files.",
    )
    return parser.parse_args()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def jsonl_write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def conservative_surface_reasons(text: str) -> list[str]:
    lowered = text.lower()
    reasons: list[str] = []
    if "'" in text:
        reasons.append("contains_apostrophe")
    if "tp" in lowered:
        reasons.append("contains_tp_cluster")
    if "db" in lowered:
        reasons.append("contains_db_cluster")
    if "rr" in lowered:
        reasons.append("contains_rr_sequence")
    return reasons


def build_reviewed_row(word: dict[str, object], audio_entry: dict[str, object]) -> dict[str, object] | None:
    entry_meta = audio_entry.get("meta") or {}
    recording_ipa = first_meta_value(entry_meta, "recording_ipa")
    if not recording_ipa:
        return None

    recording_types = audio_entry.get("recording_types") or []
    recording_type = normalize_space(str(recording_types[0] if recording_types else "unknown"))
    recording_text = first_meta_value(entry_meta, "recording_text")
    generated_text = orthography_from_ipa(recording_ipa, recording_type)
    review_flags = dedupe_preserve_order(
        collect_review_flags(recording_ipa) + collect_output_review_flags(generated_text)
    )
    speaker_user_id = first_meta_value(entry_meta, "speaker_user_id")
    return {
        "word_id": int(str(word.get("origin_id") or "0")),
        "word_title": normalize_space(str(word.get("title") or "")),
        "word_slug": normalize_space(str(word.get("slug") or "")),
        "recording_id": int(str(audio_entry.get("origin_id") or "0")),
        "recording_type": recording_type,
        "recording_ipa": recording_ipa,
        "recording_text": recording_text,
        "generated_recording_text": generated_text,
        "review_flags": review_flags,
        "speaker_user_id": speaker_user_id,
    }


def choose_first_clean_isolation(rows: list[dict[str, object]]) -> dict[str, object] | None:
    isolation_rows = [row for row in rows if row["recording_type"] == "isolation"]
    if not isolation_rows:
        return None
    return sorted(isolation_rows, key=lambda row: int(row["recording_id"]))[0]


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

    safe_recording_updates: dict[tuple[int, int], dict[str, object]] = {}
    safe_word_updates: dict[int, dict[str, object]] = {}
    safe_audit_rows: list[dict[str, object]] = []

    missing_ipa_rows: list[dict[str, object]] = []
    unreviewed_ipa_rows: list[dict[str, object]] = []
    orthography_flag_rows: list[dict[str, object]] = []
    conservative_hold_rows: list[dict[str, object]] = []
    title_suggestion_rows: list[dict[str, object]] = []
    title_placeholder_without_clean_isolation_rows: list[dict[str, object]] = []
    structural_issue_rows: list[dict[str, object]] = []

    for word in words:
        word_id = int(str(word.get("origin_id") or "0"))
        word_title = normalize_space(str(word.get("title") or ""))
        word_slug = normalize_space(str(word.get("slug") or ""))
        word_meta = word.get("meta") or {}
        current_word_translation = first_meta_value(word_meta, "word_translation")
        audio_entries = word.get("audio_entries") or []

        summary["total_words"] += 1
        if not audio_entries:
            summary["words_with_no_audio_entries"] += 1

        by_type_ids: dict[str, list[int]] = defaultdict(list)
        word_speaker_ids: set[str] = set()
        reviewed_clean_rows: list[dict[str, object]] = []
        clean_isolation_forms: list[tuple[str, int, str]] = []

        for audio_entry in audio_entries:
            summary["total_audio_entries"] += 1
            recording_id = int(str(audio_entry.get("origin_id") or "0"))
            entry_meta = audio_entry.get("meta") or {}
            recording_types = audio_entry.get("recording_types") or []
            recording_type = normalize_space(str(recording_types[0] if recording_types else "unknown"))
            by_type_ids[recording_type].append(recording_id)

            speaker_user_id = first_meta_value(entry_meta, "speaker_user_id")
            if speaker_user_id:
                word_speaker_ids.add(speaker_user_id)

            recording_ipa = first_meta_value(entry_meta, "recording_ipa")
            recording_text = first_meta_value(entry_meta, "recording_text")
            if not recording_ipa:
                summary["missing_ipa_rows"] += 1
                summary[f"missing_ipa_rows_{recording_type}"] += 1
                missing_ipa_rows.append(
                    {
                        "word_id": word_id,
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_id": recording_id,
                        "recording_type": recording_type,
                        "speaker_user_id": speaker_user_id,
                        "current_recording_text": recording_text,
                    }
                )
                continue

            summary["rows_with_recording_ipa"] += 1

            if meta_flag_is_truthy(entry_meta, "ll_auto_transcription_needs_review"):
                summary["unreviewed_ipa_rows"] += 1
                summary[f"unreviewed_ipa_rows_{recording_type}"] += 1
                unreviewed_ipa_rows.append(
                    {
                        "word_id": word_id,
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_id": recording_id,
                        "recording_type": recording_type,
                        "speaker_user_id": speaker_user_id,
                        "recording_ipa": recording_ipa,
                        "current_recording_text": recording_text,
                    }
                )
                continue

            reviewed_row = build_reviewed_row(word, audio_entry)
            if reviewed_row is None:
                continue

            summary["reviewed_ipa_rows"] += 1
            summary[f"reviewed_ipa_rows_{recording_type}"] += 1

            if reviewed_row["review_flags"]:
                summary["orthography_flag_rows"] += 1
                for flag in reviewed_row["review_flags"]:
                    summary[f"orthography_flag_{flag}"] += 1
                orthography_flag_rows.append(
                    {
                        "word_id": word_id,
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_id": recording_id,
                        "recording_type": recording_type,
                        "speaker_user_id": speaker_user_id,
                        "recording_ipa": reviewed_row["recording_ipa"],
                        "current_recording_text": reviewed_row["recording_text"],
                        "generated_recording_text": reviewed_row["generated_recording_text"],
                        "review_flags": reviewed_row["review_flags"],
                    }
                )
                continue

            reviewed_clean_rows.append(reviewed_row)
            summary["reviewed_clean_rows"] += 1

            if reviewed_row["recording_type"] == "isolation":
                clean_isolation_forms.append(
                    (
                        str(reviewed_row["generated_recording_text"]),
                        int(reviewed_row["recording_id"]),
                        str(reviewed_row["speaker_user_id"]),
                    )
                )

            generated_text = str(reviewed_row["generated_recording_text"])
            current_text = str(reviewed_row["recording_text"])
            if generated_text != current_text:
                hold_reasons: list[str] = []
                if current_text:
                    hold_reasons.append("existing_recording_text_present")
                hold_reasons.extend(conservative_surface_reasons(generated_text))
                if hold_reasons:
                    conservative_hold_rows.append(
                        {
                            "update_type": "recording_text",
                            "word_id": word_id,
                            "word_title": word_title,
                            "word_slug": word_slug,
                            "recording_id": int(reviewed_row["recording_id"]),
                            "recording_type": reviewed_row["recording_type"],
                            "old_value": current_text,
                            "new_value": generated_text,
                            "source_recording_ipa": str(reviewed_row["recording_ipa"]),
                            "hold_reasons": hold_reasons,
                        }
                    )
                    summary["conservative_hold_rows"] += 1
                    if current_text:
                        summary["conservative_hold_recording_text_existing_value"] += 1
                    else:
                        summary["conservative_hold_recording_text_risky_surface"] += 1
                else:
                    safe_recording_updates[(word_id, int(reviewed_row["recording_id"]))] = {
                        "word_id": word_id,
                        "recording_id": int(reviewed_row["recording_id"]),
                        "recording_text": generated_text,
                    }
                    safe_audit_rows.append(
                        {
                            "update_type": "recording_text",
                            "word_id": word_id,
                            "word_title": word_title,
                            "word_slug": word_slug,
                            "recording_id": int(reviewed_row["recording_id"]),
                            "recording_type": reviewed_row["recording_type"],
                            "old_value": current_text,
                            "new_value": generated_text,
                            "reason": "clean_reviewed_ipa_blank_fill_conservative_surface",
                        }
                    )
                    summary["safe_recording_text_updates"] += 1
                    summary["safe_recording_text_fills"] += 1

        clean_isolation_rows = [row for row in reviewed_clean_rows if row["recording_type"] == "isolation"]
        first_clean_isolation = choose_first_clean_isolation(reviewed_clean_rows)
        title_reason = title_backfill_reason(word_title, word_slug)

        if first_clean_isolation is not None and not current_word_translation:
            suggested_translation = str(first_clean_isolation["generated_recording_text"])
            hold_reasons = conservative_surface_reasons(suggested_translation)
            if hold_reasons:
                conservative_hold_rows.append(
                    {
                        "update_type": "word_translation",
                        "word_id": word_id,
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_id": int(first_clean_isolation["recording_id"]),
                        "recording_type": "isolation",
                        "old_value": current_word_translation,
                        "new_value": suggested_translation,
                        "source_recording_ipa": str(first_clean_isolation["recording_ipa"]),
                        "hold_reasons": hold_reasons,
                    }
                )
                summary["conservative_hold_rows"] += 1
                summary["conservative_hold_word_translation_risky_surface"] += 1
            else:
                word_update = safe_word_updates.setdefault(word_id, {"word_id": word_id})
                word_update["word_translation"] = suggested_translation
                safe_audit_rows.append(
                    {
                        "update_type": "word_translation",
                        "word_id": word_id,
                        "word_title": word_title,
                        "word_slug": word_slug,
                        "recording_id": int(first_clean_isolation["recording_id"]),
                        "recording_type": "isolation",
                        "old_value": current_word_translation,
                        "new_value": suggested_translation,
                        "reason": "fill_from_clean_reviewed_isolation_conservative_surface",
                    }
                )
                summary["safe_word_translation_fills"] += 1

        if title_reason in SAFE_TITLE_BACKFILL_REASONS and first_clean_isolation is not None:
            new_title = str(first_clean_isolation["generated_recording_text"])
            if new_title and new_title != word_title:
                hold_reasons = conservative_surface_reasons(new_title)
                if hold_reasons:
                    conservative_hold_rows.append(
                        {
                            "update_type": "word_title",
                            "word_id": word_id,
                            "word_title": word_title,
                            "word_slug": word_slug,
                            "recording_id": int(first_clean_isolation["recording_id"]),
                            "recording_type": "isolation",
                            "old_value": word_title,
                            "new_value": new_title,
                            "source_recording_ipa": str(first_clean_isolation["recording_ipa"]),
                            "hold_reasons": hold_reasons + [f"title_backfill_{title_reason}"],
                        }
                    )
                    summary["conservative_hold_rows"] += 1
                    summary["conservative_hold_word_title_risky_surface"] += 1
                else:
                    word_update = safe_word_updates.setdefault(word_id, {"word_id": word_id})
                    word_update["word_title"] = new_title
                    safe_audit_rows.append(
                        {
                            "update_type": "word_title",
                            "word_id": word_id,
                            "word_title": word_title,
                            "word_slug": word_slug,
                            "recording_id": int(first_clean_isolation["recording_id"]),
                            "recording_type": "isolation",
                            "old_value": word_title,
                            "new_value": new_title,
                            "reason": f"safe_title_backfill_{title_reason}_conservative_surface",
                        }
                    )
                    summary["safe_word_title_updates"] += 1
                    summary[f"safe_word_title_updates_{title_reason}"] += 1

        if title_reason == "matches_slug":
            if first_clean_isolation is not None:
                title_suggestion_rows.append(
                    {
                        "word_id": word_id,
                        "current_word_title": word_title,
                        "word_slug": word_slug,
                        "suggested_word_title": str(first_clean_isolation["generated_recording_text"]),
                        "source_recording_id": int(first_clean_isolation["recording_id"]),
                        "source_recording_ipa": str(first_clean_isolation["recording_ipa"]),
                        "reason": "title_matches_slug_but_not_safe_auto_fix",
                    }
                )
                summary["title_suggestion_rows"] += 1
            else:
                title_placeholder_without_clean_isolation_rows.append(
                    {
                        "word_id": word_id,
                        "current_word_title": word_title,
                        "word_slug": word_slug,
                        "reason": "title_matches_slug_no_clean_reviewed_isolation",
                    }
                )
                summary["title_placeholder_without_clean_isolation_rows"] += 1
        elif title_reason in SAFE_TITLE_BACKFILL_REASONS and first_clean_isolation is None:
            title_placeholder_without_clean_isolation_rows.append(
                {
                    "word_id": word_id,
                    "current_word_title": word_title,
                    "word_slug": word_slug,
                    "reason": f"safe_title_backfill_{title_reason}_but_no_clean_reviewed_isolation",
                }
            )
            summary["title_placeholder_without_clean_isolation_rows"] += 1

        issue_types: list[str] = []
        missing_types = [name for name in EXPECTED_RECORDING_TYPES if name not in by_type_ids]
        duplicate_type_ids = {name: ids for name, ids in by_type_ids.items() if len(ids) > 1}
        clean_iso_texts = sorted({item[0] for item in clean_isolation_forms})

        if len(word_speaker_ids) > 1:
            issue_types.append("mixed_speakers")
            summary["words_with_mixed_speakers"] += 1
        if duplicate_type_ids:
            issue_types.append("duplicate_recording_types")
            summary["words_with_duplicate_recording_types"] += 1
        if missing_types:
            issue_types.append("missing_expected_recording_types")
            summary["words_missing_expected_recording_types"] += 1
        if not audio_entries:
            issue_types.append("no_audio_entries")
        if len(clean_iso_texts) > 1:
            issue_types.append("conflicting_clean_isolation_forms")
            summary["words_with_conflicting_clean_isolation_forms"] += 1

        if issue_types:
            structural_issue_rows.append(
                {
                    "word_id": word_id,
                    "word_title": word_title,
                    "word_slug": word_slug,
                    "issue_types": issue_types,
                    "speaker_user_ids": sorted(word_speaker_ids),
                    "recording_type_counts": {name: len(ids) for name, ids in sorted(by_type_ids.items())},
                    "missing_recording_types": missing_types,
                    "duplicate_recording_ids_by_type": duplicate_type_ids,
                    "clean_isolation_forms": [
                        {
                            "text": text,
                            "recording_id": recording_id,
                            "speaker_user_id": speaker_user_id,
                        }
                        for text, recording_id, speaker_user_id in clean_isolation_forms
                    ],
                }
            )

    safe_import_rows = [safe_recording_updates[key] for key in sorted(safe_recording_updates)]
    safe_import_rows.extend(safe_word_updates[key] for key in sorted(safe_word_updates))

    summary_data = {
        "source_full_export_zip": str(full_export_zip),
        "bundle_exported_at": payload.get("exported_at"),
        "total_words": summary["total_words"],
        "total_audio_entries": summary["total_audio_entries"],
        "rows_with_recording_ipa": summary["rows_with_recording_ipa"],
        "missing_ipa_rows": summary["missing_ipa_rows"],
        "unreviewed_ipa_rows": summary["unreviewed_ipa_rows"],
        "reviewed_ipa_rows": summary["reviewed_ipa_rows"],
        "reviewed_clean_rows": summary["reviewed_clean_rows"],
        "orthography_flag_rows": summary["orthography_flag_rows"],
        "words_with_no_audio_entries": summary["words_with_no_audio_entries"],
        "words_missing_expected_recording_types": summary["words_missing_expected_recording_types"],
        "words_with_duplicate_recording_types": summary["words_with_duplicate_recording_types"],
        "words_with_mixed_speakers": summary["words_with_mixed_speakers"],
        "words_with_conflicting_clean_isolation_forms": summary["words_with_conflicting_clean_isolation_forms"],
        "title_suggestion_rows": summary["title_suggestion_rows"],
        "title_placeholder_without_clean_isolation_rows": summary["title_placeholder_without_clean_isolation_rows"],
        "safe_recording_text_updates": summary["safe_recording_text_updates"],
        "safe_recording_text_fills": summary["safe_recording_text_fills"],
        "safe_word_translation_fills": summary["safe_word_translation_fills"],
        "safe_word_title_updates": summary["safe_word_title_updates"],
        "safe_word_title_updates_blank": summary["safe_word_title_updates_blank"],
        "safe_word_title_updates_numeric": summary["safe_word_title_updates_numeric"],
        "high_confidence_import_rows": len(safe_import_rows),
        "conservative_hold_rows": summary["conservative_hold_rows"],
        "conservative_hold_recording_text_existing_value": summary["conservative_hold_recording_text_existing_value"],
        "conservative_hold_recording_text_risky_surface": summary["conservative_hold_recording_text_risky_surface"],
        "conservative_hold_word_translation_risky_surface": summary["conservative_hold_word_translation_risky_surface"],
        "conservative_hold_word_title_risky_surface": summary["conservative_hold_word_title_risky_surface"],
        "orthography_flag_initial_ʔ_manual_review": summary["orthography_flag_initial_ʔ_manual_review"],
        "orthography_flag_contains_ʊ_manual_review": summary["orthography_flag_contains_ʊ_manual_review"],
        "orthography_flag_contains_short_vowel_manual_review": summary["orthography_flag_contains_short_vowel_manual_review"],
        "orthography_flag_contains_epenthetic_vowel_manual_review": summary["orthography_flag_contains_epenthetic_vowel_manual_review"],
        "orthography_flag_non_prevelar_ŋ_manual_review": summary["orthography_flag_non_prevelar_ŋ_manual_review"],
        "orthography_flag_contains_residual_ipa_output_manual_review": summary["orthography_flag_contains_residual_ipa_output_manual_review"],
    }

    prefix = args.output_prefix
    safe_import_path = output_dir / f"{prefix}-high-confidence-fixes.jsonl"
    safe_audit_path = output_dir / f"{prefix}-high-confidence-fixes.audit.jsonl"
    missing_ipa_path = output_dir / f"{prefix}-manual-review-missing-ipa.jsonl"
    unreviewed_ipa_path = output_dir / f"{prefix}-manual-review-unreviewed-ipa.jsonl"
    orthography_flags_path = output_dir / f"{prefix}-manual-review-orthography-flags.jsonl"
    conservative_holds_path = output_dir / f"{prefix}-manual-review-conservative-holds.jsonl"
    title_suggestions_path = output_dir / f"{prefix}-manual-review-title-suggestions.jsonl"
    title_placeholders_path = output_dir / f"{prefix}-manual-review-title-placeholders-no-clean-isolation.jsonl"
    structural_path = output_dir / f"{prefix}-manual-review-structural-issues.jsonl"
    summary_path = output_dir / f"{prefix}.summary.json"

    jsonl_write(safe_import_path, safe_import_rows)
    jsonl_write(safe_audit_path, safe_audit_rows)
    jsonl_write(missing_ipa_path, missing_ipa_rows)
    jsonl_write(unreviewed_ipa_path, unreviewed_ipa_rows)
    jsonl_write(orthography_flags_path, orthography_flag_rows)
    jsonl_write(conservative_holds_path, conservative_hold_rows)
    jsonl_write(title_suggestions_path, title_suggestion_rows)
    jsonl_write(title_placeholders_path, title_placeholder_without_clean_isolation_rows)
    jsonl_write(structural_path, structural_issue_rows)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote safe import:        {safe_import_path}")
    print(f"Wrote safe audit:         {safe_audit_path}")
    print(f"Wrote missing IPA rows:   {missing_ipa_path}")
    print(f"Wrote unreviewed IPA:     {unreviewed_ipa_path}")
    print(f"Wrote orthography flags:  {orthography_flags_path}")
    print(f"Wrote conservative holds: {conservative_holds_path}")
    print(f"Wrote title suggestions:  {title_suggestions_path}")
    print(f"Wrote title placeholders: {title_placeholders_path}")
    print(f"Wrote structural issues:  {structural_path}")
    print(f"Wrote summary:            {summary_path}")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
