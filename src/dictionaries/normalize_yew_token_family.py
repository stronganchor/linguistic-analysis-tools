#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


TOKEN_START_YEVV_RE = re.compile(r"(?<!\w)(?:yêvv|yêw|yevv)")

# Exact garble observed in the Harun cluster: "yew ver yew pey ..."
SPECIAL_EXACT_REPLACEMENTS = {
    "vevv ver yevv pey vatış": "yew ver yew pey vatış",
}


def normalize_token_family(text: str) -> str:
    if not text:
        return text
    updated = SPECIAL_EXACT_REPLACEMENTS.get(text, text)
    updated = TOKEN_START_YEVV_RE.sub("yew", updated)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize likely OCR yevv/yêw token-family entries to yew."
    )
    parser.add_argument("--input", required=True, help="Input snapshot JSON path")
    parser.add_argument("--output", required=True, help="Output snapshot JSON path")
    parser.add_argument("--report-dir", required=True, help="Directory for summary/report files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entries = payload["entries"]

    changed_rows = []

    for entry_index, entry in enumerate(entries):
        senses = entry.get("senses", [])
        source_dictionaries = {sense.get("source_dictionary", "") for sense in senses}
        if source_dictionaries != {"Palu - Bingöl Harun Turgut"}:
            continue

        old_title = entry.get("title", "")
        new_title = normalize_token_family(old_title)

        sense_changes = []
        for sense_index, sense in enumerate(senses):
            old_raw_headword = sense.get("raw_headword", "")
            new_raw_headword = normalize_token_family(old_raw_headword)
            old_parent = sense.get("parent", "")
            new_parent = normalize_token_family(old_parent)
            if new_raw_headword != old_raw_headword:
                sense["raw_headword"] = new_raw_headword
            if new_parent != old_parent:
                sense["parent"] = new_parent
            if new_raw_headword != old_raw_headword or new_parent != old_parent:
                sense_changes.append(
                    {
                        "sense_index": sense_index,
                        "old_raw_headword": old_raw_headword,
                        "new_raw_headword": new_raw_headword,
                        "old_parent": old_parent,
                        "new_parent": new_parent,
                    }
                )

        if new_title != old_title:
            entry["title"] = new_title

        if new_title != old_title or sense_changes:
            first_sense = senses[0] if senses else {}
            changed_rows.append(
                {
                    "entry_index": entry_index,
                    "source_row_idx": first_sense.get("source_row_idx", ""),
                    "old_title": old_title,
                    "new_title": new_title,
                    "definition_tr": first_sense.get("translations", {}).get("tr", first_sense.get("definition", "")),
                    "sense_change_count": len(sense_changes),
                }
            )

    remaining_suspicious = []
    suspicious_needles = ("yevv", "yêw", "yêvv", "vevv")
    for entry_index, entry in enumerate(entries):
        title = entry.get("title", "")
        if any(needle in title for needle in suspicious_needles):
            first_sense = entry.get("senses", [{}])[0]
            remaining_suspicious.append(
                {
                    "entry_index": entry_index,
                    "title": title,
                    "source_dictionary": first_sense.get("source_dictionary", ""),
                    "source_row_idx": first_sense.get("source_row_idx", ""),
                    "definition_tr": first_sense.get("translations", {}).get("tr", first_sense.get("definition", "")),
                }
            )

    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    changed_tsv_path = report_dir / "changed_entries.tsv"
    with changed_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entry_index",
                "source_row_idx",
                "old_title",
                "new_title",
                "definition_tr",
                "sense_change_count",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(changed_rows)

    remaining_tsv_path = report_dir / "remaining_suspicious.tsv"
    with remaining_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "entry_index",
                "title",
                "source_dictionary",
                "source_row_idx",
                "definition_tr",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(remaining_suspicious)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "changed_entry_count": len(changed_rows),
        "remaining_suspicious_count": len(remaining_suspicious),
        "changed_entries_tsv": str(changed_tsv_path),
        "remaining_suspicious_tsv": str(remaining_tsv_path),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
