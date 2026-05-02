#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from genc_palu_dialect_rules import summarize_genc_palu_dialect_repairs
from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues
from prepare_lltools_ipa_dataset import normalize_ipa_text
from review_lltools_ipa_comparison import ipa_to_canonical, normalize_title_text


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


MODEL_ORDER = ["new_mms", "old_mms", "whisper"]
INTRO_STARTERS = {
    "ow",
    "owɨ",
    "owɪ",
    "eja",
    "ejo",
    "ɨna",
    "ɨn",
    "ina",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review the full new MMS pass against live-site lexical phonetic forms and any existing "
            "older-model outputs already on disk, then emit a conservative recording_ipa import JSONL."
        )
    )
    parser.add_argument("--new-path", required=True)
    parser.add_argument("--old-mms-path", action="append", dest="old_mms_paths", default=[])
    parser.add_argument("--whisper-path", action="append", dest="whisper_paths", default=[])
    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_jsonl_by_recording(paths: list[Path]) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                recording_id = str(row.get("recording_id") or "").strip()
                if recording_id:
                    rows[recording_id] = row
    return rows


def infer_form_type(ipa_text: str) -> str:
    text = normalize_ipa_text(ipa_text)
    if not text:
        return "unknown"
    if re.search(r"\b(ha|he|ho)\b .*t̪?͡?ʃad̪a$|t̪?͡?ʃad̪a$", text):
        return "question"
    tokens = text.split()
    if len(tokens) >= 2 and tokens[0] in INTRO_STARTERS:
        return "introduction"
    return "isolation"


def load_csv_index(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phon = normalize_ipa_text(row.get("PhoneticForm") or "")
            if not phon:
                continue
            entry = {
                "phonetic": phon,
                "canonical": ipa_to_canonical(phon),
                "type": infer_form_type(phon),
            }
            for field in ("Lexeme", "Gloss_tr"):
                key = normalize_title_text(row.get(field) or "")
                if key:
                    index[key].append(entry)
    return index


def summarize_text(ipa_text: str, word_title: str, word_translation: str) -> dict[str, object]:
    text = normalize_ipa_text(ipa_text)
    rule_summary = summarize_ipa_rule_issues(text)
    repaired = basic_repair_ipa_text(text)
    dialect = summarize_genc_palu_dialect_repairs(
        repaired,
        word_title=word_title,
        word_translation=word_translation,
    )
    final = normalize_ipa_text(str(dialect.get("repaired") or repaired))
    final_rule_summary = summarize_ipa_rule_issues(final)
    return {
        "raw": text,
        "repaired": repaired,
        "dialect_repaired": final,
        "canonical": ipa_to_canonical(final),
        "final_rule_count": int(final_rule_summary["count"]),
        "rule_types": list(rule_summary["types"]),
    }


def preferred_surface(infos: list[dict[str, object]], csv_canons: set[str]) -> str:
    def key(info: dict[str, object]) -> tuple:
        return (
            0 if str(info["canonical"]) in csv_canons else 1,
            int(info["final_rule_count"]),
            MODEL_ORDER.index(str(info["model"])),
        )

    return str(sorted(infos, key=key)[0]["dialect_repaired"])


def main() -> None:
    args = parse_args()
    new_rows = load_jsonl_by_recording([Path(args.new_path).expanduser().resolve()])
    old_rows = load_jsonl_by_recording([Path(item).expanduser().resolve() for item in args.old_mms_paths])
    whisper_rows = load_jsonl_by_recording([Path(item).expanduser().resolve() for item in args.whisper_paths])
    csv_index = load_csv_index(Path(args.csv_path).expanduser().resolve())

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    import_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    stats = Counter()

    for recording_id in sorted(new_rows, key=lambda value: int(value)):
        row = new_rows[recording_id]
        word_title = str(row.get("word_title") or "").strip()
        word_translation = str(row.get("word_translation") or "").strip()
        recording_type = str(row.get("recording_type") or "").strip()
        current_ipa = normalize_ipa_text(str(row.get("existing_ipa") or ""))

        csv_entries = []
        for key in {normalize_title_text(word_title), normalize_title_text(word_translation)}:
            if key:
                csv_entries.extend(csv_index.get(key, []))

        typed_entries = [entry for entry in csv_entries if entry["type"] in {recording_type, "unknown"}]
        csv_canons_typed = {entry["canonical"] for entry in (typed_entries or csv_entries) if entry["canonical"]}
        csv_canons_any = {entry["canonical"] for entry in csv_entries if entry["canonical"]}

        infos = []
        for model_name, lookup in (
            ("new_mms", new_rows),
            ("old_mms", old_rows),
            ("whisper", whisper_rows),
        ):
            source_row = lookup.get(recording_id)
            if not source_row:
                continue
            summary = summarize_text(
                str(source_row.get("recording_ipa") or ""),
                word_title=word_title,
                word_translation=word_translation,
            )
            summary["model"] = model_name
            infos.append(summary)

        current_info = summarize_text(current_ipa, word_title=word_title, word_translation=word_translation)
        current_canonical = str(current_info["canonical"])

        buckets: dict[str, dict[str, object]] = {}
        for info in infos:
            canonical = str(info["canonical"])
            if not canonical:
                continue
            bucket = buckets.setdefault(
                canonical,
                {"canonical": canonical, "models": set(), "infos": []},
            )
            bucket["models"].add(str(info["model"]))
            bucket["infos"].append(info)

        for bucket in buckets.values():
            bucket["support_count"] = len(bucket["models"])
            bucket["csv_support_typed"] = bucket["canonical"] in csv_canons_typed
            bucket["csv_support_any"] = bucket["canonical"] in csv_canons_any
            bucket["best_rule_count"] = min(int(info["final_rule_count"]) for info in bucket["infos"])
            bucket["chosen_ipa"] = preferred_surface(bucket["infos"], csv_canons_typed or csv_canons_any)

        ordered_buckets = sorted(
            buckets.values(),
            key=lambda bucket: (
                -int(bucket["support_count"]),
                0 if bucket["csv_support_typed"] else 1,
                0 if bucket["csv_support_any"] else 1,
                int(bucket["best_rule_count"]),
                0 if "new_mms" in bucket["models"] else 1,
            ),
        )

        chosen = None
        reason = ""
        for bucket in ordered_buckets:
            if bucket["canonical"] == current_canonical:
                continue
            if int(bucket["best_rule_count"]) > 0:
                continue
            if current_ipa:
                if int(bucket["support_count"]) >= 2 and bucket["csv_support_typed"]:
                    chosen = bucket
                    reason = "current_override_two_model_plus_csv"
                    break
                if int(bucket["support_count"]) >= 3 and bucket["csv_support_any"]:
                    chosen = bucket
                    reason = "current_override_three_model_support"
                    break
                if (
                    int(bucket["support_count"]) >= 1
                    and bucket["csv_support_typed"]
                    and int(current_info["final_rule_count"]) > 0
                ):
                    chosen = bucket
                    reason = "current_override_new_model_plus_csv_vs_rule_issues"
                    break
            else:
                if int(bucket["support_count"]) >= 2 and bucket["csv_support_typed"]:
                    chosen = bucket
                    reason = "new_resolution_two_model_plus_csv"
                    break
                if int(bucket["support_count"]) >= 3 and (bucket["csv_support_any"] or not csv_entries):
                    chosen = bucket
                    reason = "new_resolution_three_model_support"
                    break
                if (
                    int(bucket["support_count"]) >= 1
                    and "new_mms" in bucket["models"]
                    and bucket["csv_support_typed"]
                ):
                    chosen = bucket
                    reason = "new_resolution_new_model_plus_csv"
                    break

        accepted = bool(chosen)
        if accepted:
            import_rows.append(
                {
                    "word_id": row.get("word_id"),
                    "recording_id": row.get("recording_id"),
                    "recording_ipa": str(chosen["chosen_ipa"]),
                    "word_title": word_title,
                    "word_translation": word_translation,
                }
            )
            stats["accepted_rows"] += 1
            if current_ipa:
                stats["accepted_current_overrides"] += 1
            else:
                stats["accepted_new_resolutions"] += 1
        else:
            stats["rejected_rows"] += 1

        audit_rows.append(
            {
                "word_id": row.get("word_id"),
                "recording_id": row.get("recording_id"),
                "word_title": word_title,
                "word_translation": word_translation,
                "recording_type": recording_type,
                "current_recording_ipa": current_ipa,
                "current_canonical": current_canonical,
                "current_rule_count": int(current_info["final_rule_count"]),
                "csv_candidates": sorted({entry["phonetic"] for entry in csv_entries}),
                "new_mms_recording_ipa": str(new_rows[recording_id].get("recording_ipa") or ""),
                "old_mms_recording_ipa": str(old_rows.get(recording_id, {}).get("recording_ipa") or ""),
                "whisper_recording_ipa": str(whisper_rows.get(recording_id, {}).get("recording_ipa") or ""),
                "candidate_buckets": [
                    {
                        "canonical": bucket["canonical"],
                        "chosen_ipa": bucket["chosen_ipa"],
                        "models": sorted(bucket["models"]),
                        "support_count": int(bucket["support_count"]),
                        "csv_support_typed": bool(bucket["csv_support_typed"]),
                        "csv_support_any": bool(bucket["csv_support_any"]),
                        "best_rule_count": int(bucket["best_rule_count"]),
                    }
                    for bucket in ordered_buckets
                ],
                "accepted": accepted,
                "decision_reason": reason,
                "chosen_recording_ipa": str(chosen["chosen_ipa"]) if chosen else "",
            }
        )

    import_path = output_dir / "new-mms-live-csv-high-confidence-import.jsonl"
    with import_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in import_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    audit_path = output_dir / "new-mms-live-csv-high-confidence-audit.jsonl"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in audit_rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "new_rows": len(new_rows),
        "old_mms_overlap": sum(1 for rid in new_rows if rid in old_rows),
        "whisper_overlap": sum(1 for rid in new_rows if rid in whisper_rows),
        "accepted_rows": int(stats["accepted_rows"]),
        "accepted_current_overrides": int(stats["accepted_current_overrides"]),
        "accepted_new_resolutions": int(stats["accepted_new_resolutions"]),
        "rejected_rows": int(stats["rejected_rows"]),
        "decision_reason_counts": dict(
            Counter(item["decision_reason"] for item in audit_rows if item.get("decision_reason"))
        ),
    }
    summary_path = output_dir / "new-mms-live-csv-high-confidence-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote import JSONL: {import_path}")
    print(f"Wrote audit JSONL:  {audit_path}")
    print(f"Wrote summary:      {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
