#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from genc_palu_dialect_rules import summarize_genc_palu_dialect_repairs
from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues
from prepare_lltools_ipa_dataset import normalize_ipa_text, normalize_plain_text
from review_lltools_ipa_comparison import ipa_to_canonical, normalize_title_text, orthography_to_canonical


FAMILY_EXCLUDED_ROWS = {
    "2650": "İnek family watchlist",
    "3001": "İnek family watchlist",
    "3002": "İnek family watchlist",
    "15475": "Mısır unu family watchlist",
    "15476": "Mısır unu family watchlist",
    "15477": "Mısır unu family watchlist",
    "11926": "Balıkçı family watchlist",
    "11928": "Balıkçı family watchlist",
    "15765": "Ben de iyiyim watchlist",
    "5523": "Un family watchlist",
    "5524": "Un family watchlist",
    "5525": "Un family watchlist",
    "2772": "Agiz unresolved family mismatch",
    "14599": "Kalem unresolved onset mismatch",
    "7213": "Kumru unresolved onset mismatch",
    "10018": "Altın unresolved i/ɨ alternation",
    "10019": "Altın unresolved i/ɨ alternation",
    "10020": "Altın unresolved i/ɨ alternation",
    "12062": "11 unresolved jundas/jondas alternation",
    "12063": "11 unresolved jundas/jondas alternation",
    "12064": "11 unresolved jundas/jondas alternation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile family-mismatch and missing/unreviewed recording lists, compare them "
            "against a Whisper second-opinion pass, and emit a conservative recording_ipa import."
        )
    )
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--wordset-csv", required=True)
    parser.add_argument("--whisper-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--speaker-id", default="8")
    parser.add_argument("--mms-jsonl", default="")
    parser.add_argument("--review-root", default="")
    return parser.parse_args()


def summarize_ipa(value: str, *, word_title: str, word_translation: str) -> dict[str, object]:
    raw = normalize_ipa_text(value)
    repaired = basic_repair_ipa_text(raw)
    dialect = summarize_genc_palu_dialect_repairs(
        repaired,
        word_title=word_title,
        word_translation=word_translation,
    )
    final = normalize_ipa_text(str(dialect.get("repaired") or repaired))
    issues = summarize_ipa_rule_issues(final)
    return {
        "raw": raw,
        "repaired": repaired,
        "dialect_repaired": final,
        "canonical": ipa_to_canonical(final),
        "issue_count": int(issues["count"]),
        "issue_types": list(issues["types"]),
    }


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_review_ids(root: Path) -> set[str]:
    review_ids: set[str] = set()
    for path in root.rglob("*.jsonl"):
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:4000]
        except OSError:
            continue
        if "review_saved_at" not in head:
            continue
        for row in read_jsonl(path):
            recording_id = normalize_plain_text(str(row.get("recording_id") or ""))
            if recording_id:
                review_ids.add(recording_id)
    return review_ids


def load_wordset_csv(path: Path) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phon = summarize_ipa(
                str(row.get("PhoneticForm") or ""),
                word_title=str(row.get("Lexeme") or ""),
                word_translation=str(row.get("Gloss_tr") or ""),
            )
            if not phon["canonical"]:
                continue
            entry = {
                "phonetic_form": normalize_ipa_text(str(row.get("PhoneticForm") or "")),
                "canonical": str(phon["canonical"]),
            }
            for field in ("Lexeme", "Gloss_tr"):
                key = normalize_title_text(str(row.get(field) or ""))
                if key:
                    index[key].append(entry)
    return index


def load_predictions(path: Path) -> dict[str, dict]:
    rows = {}
    for row in read_jsonl(path):
        recording_id = normalize_plain_text(str(row.get("recording_id") or ""))
        if recording_id:
            rows[recording_id] = row
    return rows


def load_metadata_rows(path: Path, speaker_id: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if normalize_plain_text(str(row.get("speaker_user_id") or "")) != speaker_id:
                continue
            rows.append(
                {
                    "word_id": normalize_plain_text(str(row.get("word_id") or "")),
                    "word_title": normalize_plain_text(str(row.get("word_title") or "")),
                    "word_translation": normalize_plain_text(str(row.get("word_translation") or "")),
                    "recording_id": normalize_plain_text(str(row.get("recording_id") or "")),
                    "recording_type": normalize_plain_text(str(row.get("recording_types") or "")) or "unknown",
                    "speaker_user_id": normalize_plain_text(str(row.get("speaker_user_id") or "")),
                    "speaker_name": normalize_plain_text(str(row.get("speaker_name") or "")),
                    "recording_text": normalize_plain_text(str(row.get("recording_text") or "")),
                    "recording_translation": normalize_plain_text(str(row.get("recording_translation") or "")),
                    "recording_ipa": normalize_ipa_text(str(row.get("recording_ipa") or "")),
                    "audio_relpath": normalize_plain_text(str(row.get("audio") or "")),
                }
            )
    rows.sort(key=lambda row: (row["word_title"], row["recording_type"], row["recording_id"]))
    return rows


def best_csv_canons(row: dict, csv_index: dict[str, list[dict[str, str]]]) -> set[str]:
    keys = {
        normalize_title_text(row["word_title"]),
        normalize_title_text(row["word_translation"]),
        normalize_title_text(row["recording_translation"]),
    }
    canons: set[str] = set()
    for key in keys:
        if not key:
            continue
        for entry in csv_index.get(key, []):
            canon = normalize_plain_text(str(entry.get("canonical") or ""))
            if canon:
                canons.add(canon)
    return canons


def main() -> None:
    args = parse_args()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    wordset_csv = Path(args.wordset_csv).expanduser().resolve()
    whisper_jsonl = Path(args.whisper_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review_root = (
        Path(args.review_root).expanduser().resolve()
        if args.review_root
        else metadata_csv.parent.parent
    )

    metadata_rows = load_metadata_rows(metadata_csv, normalize_plain_text(args.speaker_id))
    rows_by_recording = {row["recording_id"]: row for row in metadata_rows}
    rows_by_word: dict[str, list[dict]] = defaultdict(list)
    for row in metadata_rows:
        rows_by_word[row["word_id"] or row["word_title"]].append(row)

    review_ids = load_review_ids(review_root)
    whisper_by_recording = load_predictions(whisper_jsonl)
    mms_by_recording = load_predictions(Path(args.mms_jsonl).expanduser().resolve()) if args.mms_jsonl else {}
    csv_index = load_wordset_csv(wordset_csv)

    family_rows: list[dict] = []
    missing_rows: list[dict] = []
    unreviewed_rows: list[dict] = []
    target_rows: list[dict] = []
    high_conf_import: list[dict] = []
    high_conf_audit: list[dict] = []

    for row in metadata_rows:
        recording_id = row["recording_id"]
        family_reason = FAMILY_EXCLUDED_ROWS.get(recording_id, "")
        missing_ipa = not row["recording_ipa"]
        reviewed = recording_id in review_ids
        unreviewed_existing = bool(row["recording_ipa"]) and not reviewed

        if family_reason:
            family_rows.append({**row, "reason": family_reason})
        if missing_ipa:
            missing_rows.append({**row, "reason": "missing_recording_ipa"})
        if unreviewed_existing:
            unreviewed_rows.append({**row, "reason": "no_saved_manual_review"})

        if family_reason or missing_ipa or unreviewed_existing:
            whisper_row = whisper_by_recording.get(recording_id, {})
            mms_row = mms_by_recording.get(recording_id, {})
            whisper = summarize_ipa(
                str(whisper_row.get("recording_ipa") or ""),
                word_title=row["word_title"],
                word_translation=row["word_translation"],
            )
            current = summarize_ipa(
                row["recording_ipa"],
                word_title=row["word_title"],
                word_translation=row["word_translation"],
            )
            mms = summarize_ipa(
                str(mms_row.get("recording_ipa") or ""),
                word_title=row["word_title"],
                word_translation=row["word_translation"],
            )
            text_canonical = orthography_to_canonical(row["recording_text"])

            family_counter = Counter()
            family_same_type_counter = Counter()
            for other in rows_by_word[row["word_id"] or row["word_title"]]:
                if other["recording_id"] == recording_id:
                    continue
                if not other["recording_ipa"]:
                    continue
                other_summary = summarize_ipa(
                    other["recording_ipa"],
                    word_title=other["word_title"],
                    word_translation=other["word_translation"],
                )
                canon = str(other_summary["canonical"])
                if canon:
                    family_counter[canon] += 1
                    if other["recording_type"] == row["recording_type"]:
                        family_same_type_counter[canon] += 1

            csv_canons = best_csv_canons(row, csv_index)
            whisper_canonical = str(whisper["canonical"])
            current_canonical = str(current["canonical"])
            mms_canonical = str(mms["canonical"])

            whisper_matches_text = bool(whisper_canonical and whisper_canonical == text_canonical)
            current_matches_text = bool(current_canonical and current_canonical == text_canonical)
            mms_matches_text = bool(mms_canonical and mms_canonical == text_canonical)
            whisper_family_support = family_counter[whisper_canonical] > 0 if whisper_canonical else False
            whisper_same_type_support = family_same_type_counter[whisper_canonical] > 0 if whisper_canonical else False
            whisper_csv_support = whisper_canonical in csv_canons if whisper_canonical else False
            whisper_mms_agree = bool(whisper_canonical and whisper_canonical == mms_canonical)

            proposal = ""
            proposal_reason = ""
            if whisper["dialect_repaired"] and whisper["issue_count"] == 0:
                if missing_ipa:
                    if whisper_matches_text:
                        proposal = str(whisper["dialect_repaired"])
                        proposal_reason = "missing_ipa_whisper_matches_recording_text"
                    elif whisper_mms_agree and (whisper_family_support or whisper_csv_support):
                        proposal = str(whisper["dialect_repaired"])
                        proposal_reason = "missing_ipa_two_model_plus_anchor"
                    elif whisper_same_type_support and whisper_matches_text:
                        proposal = str(whisper["dialect_repaired"])
                        proposal_reason = "missing_ipa_same_type_family_plus_text"
                    elif whisper_csv_support and row["recording_type"] == "isolation" and whisper_matches_text:
                        proposal = str(whisper["dialect_repaired"])
                        proposal_reason = "missing_ipa_isolation_csv_plus_text"
                else:
                    if family_reason and current_canonical != whisper_canonical:
                        if whisper_matches_text and (whisper_family_support or whisper_mms_agree or whisper_csv_support):
                            proposal = str(whisper["dialect_repaired"])
                            proposal_reason = "family_excluded_whisper_matches_anchor"
                    elif unreviewed_existing and current_canonical != whisper_canonical:
                        if whisper_matches_text and not current_matches_text and (
                            whisper_mms_agree or whisper_family_support or whisper_csv_support
                        ):
                            proposal = str(whisper["dialect_repaired"])
                            proposal_reason = "unreviewed_existing_whisper_matches_anchor"

            target_row = {
                **row,
                "category_family_excluded": bool(family_reason),
                "category_missing_ipa": bool(missing_ipa),
                "category_unreviewed_existing_ipa": bool(unreviewed_existing),
                "reviewed_by_saved_artifacts": bool(reviewed),
                "family_reason": family_reason,
                "current_canonical": current_canonical,
                "recording_text_canonical": text_canonical,
                "whisper_recording_ipa": str(whisper["dialect_repaired"]),
                "whisper_canonical": whisper_canonical,
                "whisper_issue_count": int(whisper["issue_count"]),
                "mms_recording_ipa": str(mms["dialect_repaired"]),
                "mms_canonical": mms_canonical,
                "family_other_canons": dict(family_counter),
                "family_same_type_canons": dict(family_same_type_counter),
                "csv_canons": sorted(csv_canons),
                "whisper_matches_text": whisper_matches_text,
                "current_matches_text": current_matches_text,
                "mms_matches_text": mms_matches_text,
                "whisper_family_support": whisper_family_support,
                "whisper_same_type_support": whisper_same_type_support,
                "whisper_csv_support": whisper_csv_support,
                "whisper_mms_agree": whisper_mms_agree,
                "proposal_recording_ipa": proposal,
                "proposal_reason": proposal_reason,
            }
            target_rows.append(target_row)

            if proposal and proposal != row["recording_ipa"]:
                import_row = {
                    "word_id": int(row["word_id"]) if row["word_id"].isdigit() else row["word_id"],
                    "recording_id": int(recording_id) if recording_id.isdigit() else recording_id,
                    "recording_ipa": proposal,
                    "word_title": row["word_title"],
                    "word_translation": row["word_translation"],
                }
                high_conf_import.append(import_row)
                high_conf_audit.append(target_row)

    family_path = output_dir / "family_excluded_rows.tsv"
    missing_path = output_dir / "missing_ipa_rows.tsv"
    unreviewed_path = output_dir / "unreviewed_existing_ipa_rows.tsv"
    targets_path = output_dir / "second_opinion_target_rows.jsonl"
    targets_tsv_path = output_dir / "second_opinion_target_rows.tsv"
    import_path = output_dir / "high_conf_second_opinion_import.jsonl"
    audit_path = output_dir / "high_conf_second_opinion_import.audit.jsonl"
    summary_path = output_dir / "summary.json"

    def write_tsv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(family_path, family_rows)
    write_tsv(missing_path, missing_rows)
    write_tsv(unreviewed_path, unreviewed_rows)
    write_tsv(targets_tsv_path, target_rows)

    with targets_path.open("w", encoding="utf-8") as handle:
        for row in target_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with import_path.open("w", encoding="utf-8") as handle:
        for row in high_conf_import:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with audit_path.open("w", encoding="utf-8") as handle:
        for row in high_conf_audit:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "metadata_csv": str(metadata_csv),
        "wordset_csv": str(wordset_csv),
        "whisper_jsonl": str(whisper_jsonl),
        "mms_jsonl": str(Path(args.mms_jsonl).expanduser().resolve()) if args.mms_jsonl else "",
        "speaker_id": normalize_plain_text(args.speaker_id),
        "review_root": str(review_root),
        "total_speaker_rows": len(metadata_rows),
        "family_excluded_rows": len(family_rows),
        "missing_ipa_rows": len(missing_rows),
        "saved_review_ids": len(review_ids),
        "unreviewed_existing_ipa_rows": len(unreviewed_rows),
        "target_rows_total": len(target_rows),
        "high_conf_import_rows": len(high_conf_import),
        "proposal_reasons": dict(Counter(row["proposal_reason"] for row in high_conf_audit if row["proposal_reason"])),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
