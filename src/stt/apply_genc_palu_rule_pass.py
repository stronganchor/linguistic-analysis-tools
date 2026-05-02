#!/usr/bin/env python3
import json
from pathlib import Path

from genc_palu_dialect_rules import summarize_genc_palu_dialect_repairs
from ipa_rule_checks import summarize_ipa_rule_issues


ROOT = Path(__file__).resolve().parent.parent
MASTER_PATH = (
    ROOT
    / "zazacaogren"
    / "master-live-site-update-april-2026"
    / "zazacaogren-master-recording_ipa-update-session-v3.jsonl"
)
UNRESOLVED_AUDIT_PATH = (
    ROOT
    / "zazacaogren"
    / "missing-ipa-non-cerez-ipa-review-april-2026"
    / "missing-non-cerez-resolution-audit.jsonl"
)
OUTPUT_DIR = ROOT / "zazacaogren" / "rule-phonology-review-april-2026"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_rows = load_jsonl(MASTER_PATH)
    master_fixes: list[dict] = []
    master_audit: list[dict] = []

    for row in master_rows:
        summary = summarize_genc_palu_dialect_repairs(
            row.get("recording_ipa", ""),
            word_title=str(row.get("word_title") or ""),
            word_translation=str(row.get("word_translation") or ""),
        )
        repaired = str(summary["repaired"] or "")
        if not summary["changed"]:
            continue
        if summarize_ipa_rule_issues(repaired)["count"]:
            continue
        master_fixes.append(
            {
                "word_id": row["word_id"],
                "recording_id": row["recording_id"],
                "recording_ipa": repaired,
                "word_title": row.get("word_title", ""),
                "word_translation": row.get("word_translation", ""),
                "selected_model": row.get("selected_model", ""),
                "resolution_type": "genc_palu_rule_fix",
                "evidence_sources": ["genc_palu_dialect_rules", *summary["types"]],
            }
        )
        master_audit.append(
            {
                "word_id": row["word_id"],
                "recording_id": row["recording_id"],
                "word_title": row.get("word_title", ""),
                "word_translation": row.get("word_translation", ""),
                "old_recording_ipa": row.get("recording_ipa", ""),
                "new_recording_ipa": repaired,
                "applied_rule_types": summary["types"],
            }
        )

    unresolved_rows = load_jsonl(UNRESOLVED_AUDIT_PATH)
    unresolved_updates: list[dict] = []
    unresolved_audit: list[dict] = []

    for row in unresolved_rows:
        if row.get("resolution_status") != "unresolved":
            continue
        word_title = str(row.get("word_title") or "")
        word_translation = str(row.get("word_translation") or "")
        left_summary = summarize_genc_palu_dialect_repairs(
            str(row.get("mms_recording_ipa") or ""),
            word_title=word_title,
            word_translation=word_translation,
        )
        right_summary = summarize_genc_palu_dialect_repairs(
            str(row.get("whisper_medium_recording_ipa") or ""),
            word_title=word_title,
            word_translation=word_translation,
        )
        left_repaired = str(left_summary["repaired"] or "")
        right_repaired = str(right_summary["repaired"] or "")
        if not left_repaired or left_repaired != right_repaired:
            continue
        if summarize_ipa_rule_issues(left_repaired)["count"]:
            continue
        unresolved_updates.append(
            {
                "word_id": row["word_id"],
                "recording_id": row["recording_id"],
                "recording_ipa": left_repaired,
                "word_title": word_title,
                "word_translation": word_translation,
                "selected_model": "dialect_rule_merge",
                "resolution_type": "genc_palu_rule_resolved_unresolved",
                "evidence_sources": [
                    "genc_palu_dialect_rules",
                    *left_summary["types"],
                    *[item for item in right_summary["types"] if item not in left_summary["types"]],
                ],
            }
        )
        unresolved_audit.append(
            {
                "word_id": row["word_id"],
                "recording_id": row["recording_id"],
                "word_title": word_title,
                "word_translation": word_translation,
                "left_original": row.get("mms_recording_ipa", ""),
                "right_original": row.get("whisper_medium_recording_ipa", ""),
                "resolved_recording_ipa": left_repaired,
                "left_rule_types": left_summary["types"],
                "right_rule_types": right_summary["types"],
            }
        )

    combined = [
        *master_fixes,
        *[
            row
            for row in unresolved_updates
            if (row["word_id"], row["recording_id"]) not in {(item["word_id"], item["recording_id"]) for item in master_fixes}
        ],
    ]

    master_fixes_path = OUTPUT_DIR / "master-recording_ipa-rule-fixes.jsonl"
    master_audit_path = OUTPUT_DIR / "master-recording_ipa-rule-fixes.audit.jsonl"
    unresolved_updates_path = OUTPUT_DIR / "unresolved-recording_ipa-rule-resolutions.jsonl"
    unresolved_audit_path = OUTPUT_DIR / "unresolved-recording_ipa-rule-resolutions.audit.jsonl"
    combined_path = OUTPUT_DIR / "recording_ipa-rule-pass-combined.jsonl"
    summary_path = OUTPUT_DIR / "rule-pass-summary.json"

    write_jsonl(master_fixes_path, master_fixes)
    write_jsonl(master_audit_path, master_audit)
    write_jsonl(unresolved_updates_path, unresolved_updates)
    write_jsonl(unresolved_audit_path, unresolved_audit)
    write_jsonl(combined_path, combined)

    summary = {
        "master_input": str(MASTER_PATH),
        "unresolved_audit_input": str(UNRESOLVED_AUDIT_PATH),
        "master_fix_rows": len(master_fixes),
        "unresolved_new_rows": len(unresolved_updates),
        "combined_rows": len(combined),
        "master_fixes_jsonl": str(master_fixes_path),
        "unresolved_updates_jsonl": str(unresolved_updates_path),
        "combined_jsonl": str(combined_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
