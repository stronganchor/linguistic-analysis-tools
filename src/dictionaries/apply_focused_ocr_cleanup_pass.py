#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from _dictionary_pipeline_common import normalize_search_text


HARUN_SOURCE_DICTIONARY = "Palu - Bingöl Harun Turgut"

TOKEN_START_YEW_RE = re.compile(r"(?<!\w)(?:yêvv|yêw|yevv)")

# Apply after token-start yew normalization so the same map can fix both
# raw page-batch rows and a snapshot that was already yew-normalized.
CANONICAL_HEADWORD_FIXES = {
    "çnney a\\vk": "çımey awk",
    "çnney avk": "çımey awk",
    "facia, faci«a": "facia, faci'a",
    "gnu ser «ayış": "gun ser nayış",
    "gun tede ınnıcndış": "gun tede nımendiş",
    "hela u\\v weşibu": "hela uw weşibu",
    "hela uv weşibu": "hela uw weşibu",
    "hemc çi": "heme çi",
    "hemc kes": "heme kes",
    "hemclnayış": "hemelnayış",
    "hcınomun": "hemomun",
    "vevv ver yew pey vatış": "yew ver yew pey vatış",
    "ycu n yew": "yewra yew",
    "yew dirlıeın": "yew dirhem",
    "yew ena kemıbi": "yew ena kêmbi",
    "yew eni lıeta yew evvı heta:": "yew eni heta yew ewi heta:",
    "yew este yew pucste mcndış": "yew este yew pucste mendiş",
    "yew fırt": "yew firt",
    "yew lıcll": "yew hell",
    "yew lıell vêr": "yew hell vêr",
    "yew hcmlcdı": "yew hemledı",
    "yew uefesıdı": "yew nefesıdı",
    "yew de\\v xuıy yew veyvv ser «ışına": "yew dev xuy yew veyw ser nışına",
    "yew dev xuıy yew veyvv ser ışına": "yew dev xuy yew veyw ser nışına",
    "herae ye\\v": "heme yew",
    "herae yev": "heme yew",
    "hcmle": "hemle",
    "hcmle eştış": "hemle eştış",
    "kita pa mcndış": "kita pa mendış",
    "sare çe\\v bmo nıkerdış": "sare çew bıno nıkerdiş",
    "sare çev bmo nıkerdış": "sare çew bıno nıkerdiş",
    "zalıf anıeyış": "zahf ameyış",
    "zalıf bıyayış": "zahf bıyayış",
    "zalıf cıri anıeyış": "zahf cıri ameyış",
    "zalıf hetya": "zahf hêtya",
    "zalıf kerdış": "zahf kerdış",
    "zalıf nımcnd": "zahf nımend",
    "zunıy... yew vince dergu": "zunıy ... yew vince dergu",
    "zunıy&#8230; yew vince dergu": "zunıy ... yew vince dergu",
}

SUSPICIOUS_FRAGMENTS = (
    "\\",
    "«",
    "dirlı",
    "hcml",
    "uefes",
    "ycu n",
    "veyvv",
    "mcnd",
)


def clean_cell(value: object) -> str:
    return "" if value is None else str(value).strip()


def append_note(existing: str, note: str) -> str:
    existing = clean_cell(existing)
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing} | {note}"


def normalize_headword(text: str) -> str:
    text = clean_cell(text)
    if not text:
        return text
    updated = TOKEN_START_YEW_RE.sub("yew", text)
    updated = CANONICAL_HEADWORD_FIXES.get(updated, updated)
    return updated


def replace_headword_prefix(source_text_raw: str, old_headword: str, new_headword: str) -> str:
    source_text_raw = clean_cell(source_text_raw)
    old_headword = clean_cell(old_headword)
    new_headword = clean_cell(new_headword)
    if not source_text_raw or not old_headword or old_headword == new_headword:
        return source_text_raw
    if source_text_raw.startswith(old_headword):
        return new_headword + source_text_raw[len(old_headword) :]
    return source_text_raw.replace(old_headword, new_headword, 1)


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def update_page_batches(page_batches_dir: Path) -> list[dict[str, str]]:
    changed_rows: list[dict[str, str]] = []
    note = "ocr_focus_cleanup_20260420"

    for batch_path in sorted(page_batches_dir.glob("page_*.tsv")):
        fieldnames, rows = read_tsv(batch_path)
        batch_changed = False

        for row in rows:
            old_headword = clean_cell(row.get("entry_raw"))
            new_headword = normalize_headword(old_headword)
            if new_headword == old_headword:
                continue

            row["entry_raw"] = new_headword
            row["entry_display"] = new_headword
            row["entry_search"] = normalize_search_text(new_headword)
            row["source_text_raw"] = replace_headword_prefix(row.get("source_text_raw", ""), old_headword, new_headword)
            if "qa_notes" in row:
                row["qa_notes"] = append_note(row.get("qa_notes", ""), note)
            row["review_notes"] = append_note(row.get("review_notes", ""), note)

            changed_rows.append(
                {
                    "scope": "page_batch",
                    "page_batch_path": str(batch_path),
                    "entry_id": clean_cell(row.get("entry_id")),
                    "source_row_idx": clean_cell(row.get("original_row_index")),
                    "old_headword": old_headword,
                    "new_headword": new_headword,
                    "definition_tr": clean_cell(row.get("definition_tr_raw")),
                }
            )
            batch_changed = True

        if batch_changed:
            write_tsv(batch_path, fieldnames, rows)

    return changed_rows


def sync_master_from_page_batches(page_batches_dir: Path, master_path: Path) -> None:
    batch_paths = sorted(page_batches_dir.glob("page_*.tsv"))
    if not batch_paths:
        raise FileNotFoundError(f"No page-batch files found in {page_batches_dir}")

    master_fieldnames: list[str] = []
    synced_rows: list[dict[str, str]] = []
    for batch_path in batch_paths:
        fieldnames, rows = read_tsv(batch_path)
        if not master_fieldnames:
            master_fieldnames = fieldnames
        elif fieldnames != master_fieldnames:
            raise ValueError(f"Field mismatch in {batch_path}")
        synced_rows.extend(rows)

    write_tsv(master_path, master_fieldnames, synced_rows)


def update_snapshot(snapshot_input: Path, snapshot_output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    payload = json.loads(snapshot_input.read_text(encoding="utf-8"))
    changed_rows: list[dict[str, str]] = []
    remaining_suspicious: list[dict[str, str]] = []

    for entry in payload["entries"]:
        senses = entry.get("senses", [])
        harun_senses = [sense for sense in senses if clean_cell(sense.get("source_dictionary")) == HARUN_SOURCE_DICTIONARY]
        if not harun_senses:
            continue

        old_title = clean_cell(entry.get("title"))
        new_title = normalize_headword(old_title)

        if len(harun_senses) == len(senses) and new_title != old_title:
            entry["title"] = new_title

        sense_changed = False
        for sense in harun_senses:
            old_raw = clean_cell(sense.get("raw_headword"))
            new_raw = normalize_headword(old_raw)
            if new_raw != old_raw:
                sense["raw_headword"] = new_raw
                sense_changed = True

            old_parent = clean_cell(sense.get("parent"))
            new_parent = normalize_headword(old_parent)
            if new_parent != old_parent:
                sense["parent"] = new_parent
                sense_changed = True

        if (len(harun_senses) == len(senses) and new_title != old_title) or sense_changed:
            first_harun = harun_senses[0]
            changed_rows.append(
                {
                    "scope": "snapshot",
                    "entry_index": str(payload["entries"].index(entry)),
                    "source_row_idx": clean_cell(first_harun.get("source_row_idx")),
                    "old_headword": old_title,
                    "new_headword": clean_cell(entry.get("title")),
                    "definition_tr": clean_cell(first_harun.get("translations", {}).get("tr", first_harun.get("definition"))),
                }
            )

        title_for_scan = clean_cell(entry.get("title"))
        if any(fragment in title_for_scan for fragment in SUSPICIOUS_FRAGMENTS):
            first_harun = harun_senses[0]
            remaining_suspicious.append(
                {
                    "entry_index": str(payload["entries"].index(entry)),
                    "title": title_for_scan,
                    "source_row_idx": clean_cell(first_harun.get("source_row_idx")),
                    "definition_tr": clean_cell(first_harun.get("translations", {}).get("tr", first_harun.get("definition"))),
                }
            )

    snapshot_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return changed_rows, remaining_suspicious


def write_report_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a focused Harun OCR cleanup pass to reviewed page batches and a current snapshot.")
    parser.add_argument(
        "--page-batches-dir",
        default="$Z_WORKSPACE/Dictionaries/tsv_output/harun_review/page_batches",
    )
    parser.add_argument(
        "--harun-master-path",
        default="$Z_WORKSPACE/Dictionaries/tsv_output/harun_entries_review.tsv",
    )
    parser.add_argument(
        "--snapshot-input",
        required=True,
    )
    parser.add_argument(
        "--snapshot-output",
        required=True,
    )
    parser.add_argument(
        "--report-dir",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    page_batches_dir = Path(args.page_batches_dir)
    harun_master_path = Path(args.harun_master_path)
    snapshot_input = Path(args.snapshot_input)
    snapshot_output = Path(args.snapshot_output)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    page_batch_changes = update_page_batches(page_batches_dir)
    sync_master_from_page_batches(page_batches_dir, harun_master_path)
    snapshot_changes, remaining_suspicious = update_snapshot(snapshot_input, snapshot_output)

    changed_source_tsv = report_dir / "changed_page_batch_rows.tsv"
    changed_snapshot_tsv = report_dir / "changed_snapshot_entries.tsv"
    remaining_tsv = report_dir / "remaining_suspicious.tsv"

    write_report_tsv(
        changed_source_tsv,
        ["scope", "page_batch_path", "entry_id", "source_row_idx", "old_headword", "new_headword", "definition_tr"],
        page_batch_changes,
    )
    write_report_tsv(
        changed_snapshot_tsv,
        ["scope", "entry_index", "source_row_idx", "old_headword", "new_headword", "definition_tr"],
        snapshot_changes,
    )
    write_report_tsv(
        remaining_tsv,
        ["entry_index", "title", "source_row_idx", "definition_tr"],
        remaining_suspicious,
    )

    summary = {
        "page_batch_changes": len(page_batch_changes),
        "snapshot_changes": len(snapshot_changes),
        "remaining_suspicious": len(remaining_suspicious),
        "snapshot_input": str(snapshot_input),
        "snapshot_output": str(snapshot_output),
        "harun_master_path": str(harun_master_path),
        "changed_page_batch_rows_tsv": str(changed_source_tsv),
        "changed_snapshot_entries_tsv": str(changed_snapshot_tsv),
        "remaining_suspicious_tsv": str(remaining_tsv),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
