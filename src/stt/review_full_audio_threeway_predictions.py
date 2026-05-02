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
    "ɨno",
    "ina",
}
QUESTION_RE = re.compile(r"\b(ha|he|ho)\b .*t[̪]?[͡]?ʃad̪a$|t[̪]?[͡]?ʃad̪a$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review three full-audio transcription JSONLs against current imported IPA state "
            "and live-site lexical phonetic forms, then emit a conservative combined update JSONL."
        )
    )
    parser.add_argument("--new-path", required=True)
    parser.add_argument("--old-mms-path", required=True)
    parser.add_argument("--whisper-path", required=True)
    parser.add_argument("--csv-path", required=True)
    parser.add_argument(
        "--overlay-update",
        action="append",
        dest="overlay_updates",
        default=[],
        help="Imported recording_ipa update JSONL(s) to overlay over the base bundle existing_ipa.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_jsonl_by_recording(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
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


def load_overlay_updates(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    current_map: dict[str, str] = {}
    source_map: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                recording_id = str(row.get("recording_id") or "").strip()
                recording_ipa = normalize_ipa_text(str(row.get("recording_ipa") or ""))
                if recording_id and recording_ipa:
                    current_map[recording_id] = recording_ipa
                    source_map[recording_id] = str(path)
    return current_map, source_map


def infer_form_type(ipa_text: str) -> str:
    text = normalize_ipa_text(ipa_text)
    if not text:
        return "unknown"
    if QUESTION_RE.search(text):
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
        "rule_count": int(rule_summary["count"]),
        "final_rule_count": int(final_rule_summary["count"]),
        "rule_types": list(rule_summary["types"]),
        "dialect_changed": bool(dialect.get("changed")),
        "dialect_types": list(dialect.get("types") or []),
    }


def choose_preferred_surface(infos: list[dict[str, object]], csv_canons: set[str], preferred_order: list[str]) -> str:
    def sort_key(info: dict[str, object]) -> tuple:
        return (
            0 if str(info["canonical"]) in csv_canons else 1,
            int(info["final_rule_count"]),
            preferred_order.index(str(info["model"])),
        )

    best = sorted(infos, key=sort_key)[0]
    return str(best["dialect_repaired"] or best["raw"])


def sort_candidate(bucket: dict) -> tuple:
    return (
        -int(bucket["support_count"]),
        0 if bucket["csv_support_typed"] else 1,
        0 if bucket["csv_support_any"] else 1,
        int(bucket["best_final_rule_count"]),
        0 if bucket["supports_new_mms"] else 1,
    )


def main() -> None:
    args = parse_args()
    new_path = Path(args.new_path).expanduser().resolve()
    old_mms_path = Path(args.old_mms_path).expanduser().resolve()
    whisper_path = Path(args.whisper_path).expanduser().resolve()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    new_rows = load_jsonl_by_recording(new_path)
    old_mms_rows = load_jsonl_by_recording(old_mms_path)
    whisper_rows = load_jsonl_by_recording(whisper_path)
    common_ids = sorted(set(new_rows) & set(old_mms_rows) & set(whisper_rows), key=lambda value: int(value))

    overlay_paths = [Path(item).expanduser().resolve() for item in args.overlay_updates]
    overlay_current_map, overlay_source_map = load_overlay_updates(overlay_paths)
    csv_index = load_csv_index(csv_path)

    import_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    stats = Counter()

    for recording_id in common_ids:
        base_row = new_rows[recording_id]
        word_title = str(base_row.get("word_title") or "").strip()
        word_translation = str(base_row.get("word_translation") or "").strip()
        recording_type = str(base_row.get("recording_type") or "").strip()
        base_existing_ipa = normalize_ipa_text(str(base_row.get("existing_ipa") or ""))

        current_ipa = overlay_current_map.get(recording_id, base_existing_ipa)
        current_source = overlay_source_map.get(recording_id, "base_existing_ipa" if base_existing_ipa else "missing")

        title_key = normalize_title_text(word_title)
        translation_key = normalize_title_text(word_translation)
        csv_entries = []
        if title_key:
            csv_entries.extend(csv_index.get(title_key, []))
        if translation_key:
            csv_entries.extend(csv_index.get(translation_key, []))

        csv_canons_any = {entry["canonical"] for entry in csv_entries if entry["canonical"]}
        typed_entries = [entry for entry in csv_entries if entry["type"] in {recording_type, "unknown"}]
        csv_canons_typed = {
            entry["canonical"]
            for entry in (typed_entries or csv_entries)
            if entry["canonical"]
        }

        model_infos: list[dict[str, object]] = []
        for model_name, rows in (
            ("new_mms", new_rows),
            ("old_mms", old_mms_rows),
            ("whisper", whisper_rows),
        ):
            summary = summarize_text(
                str(rows[recording_id].get("recording_ipa") or ""),
                word_title=word_title,
                word_translation=word_translation,
            )
            summary["model"] = model_name
            model_infos.append(summary)

        current_info = summarize_text(current_ipa, word_title=word_title, word_translation=word_translation)
        current_canonical = str(current_info["canonical"])
        current_csv_supported_typed = current_canonical in csv_canons_typed if current_canonical else False
        current_csv_supported_any = current_canonical in csv_canons_any if current_canonical else False

        buckets: dict[str, dict] = {}
        for info in model_infos:
            canonical = str(info["canonical"])
            if canonical == "":
                continue
            bucket = buckets.setdefault(
                canonical,
                {
                    "canonical": canonical,
                    "models": set(),
                    "infos": [],
                },
            )
            bucket["models"].add(str(info["model"]))
            bucket["infos"].append(info)

        for bucket in buckets.values():
            bucket["support_count"] = len(bucket["models"])
            bucket["supports_new_mms"] = "new_mms" in bucket["models"]
            bucket["csv_support_typed"] = bucket["canonical"] in csv_canons_typed
            bucket["csv_support_any"] = bucket["canonical"] in csv_canons_any
            bucket["best_final_rule_count"] = min(int(info["final_rule_count"]) for info in bucket["infos"])
            bucket["chosen_ipa"] = choose_preferred_surface(bucket["infos"], csv_canons_typed or csv_canons_any, MODEL_ORDER)

        sorted_buckets = sorted(buckets.values(), key=sort_candidate)
        chosen_bucket = None
        decision_reason = ""

        if current_ipa:
            stats["rows_with_current_ipa"] += 1
            if current_canonical and current_canonical in buckets:
                stats["current_supported_by_models"] += 1
            else:
                for bucket in sorted_buckets:
                    if bucket["canonical"] == current_canonical:
                        continue
                    if bucket["best_final_rule_count"] > 0:
                        continue
                    if bucket["support_count"] >= 3 and (
                        bucket["csv_support_typed"]
                        or (bucket["csv_support_any"] and not current_csv_supported_any)
                    ):
                        chosen_bucket = bucket
                        decision_reason = "override_current_triple_model_support"
                        break
                    if (
                        bucket["support_count"] >= 2
                        and bucket["csv_support_typed"]
                        and (
                            int(current_info["final_rule_count"]) > 0
                            or not current_csv_supported_typed
                        )
                    ):
                        chosen_bucket = bucket
                        decision_reason = "override_current_two_model_support_with_csv"
                        break
        else:
            stats["rows_without_current_ipa"] += 1
            for bucket in sorted_buckets:
                if bucket["best_final_rule_count"] > 0:
                    continue
                if bucket["support_count"] >= 3 and (
                    bucket["csv_support_typed"] or bucket["csv_support_any"] or not csv_canons_any
                ):
                    chosen_bucket = bucket
                    decision_reason = "new_resolution_triple_model_support"
                    break
                if bucket["support_count"] >= 2 and bucket["csv_support_typed"]:
                    chosen_bucket = bucket
                    decision_reason = "new_resolution_two_model_support_with_csv"
                    break
                if (
                    recording_type == "isolation"
                    and bucket["support_count"] >= 2
                    and bucket["csv_support_any"]
                ):
                    chosen_bucket = bucket
                    decision_reason = "new_isolation_resolution_two_model_support_with_csv"
                    break

        chosen_ipa = str(chosen_bucket["chosen_ipa"]) if chosen_bucket else ""
        accepted = bool(chosen_bucket and normalize_ipa_text(chosen_ipa) and chosen_bucket["canonical"] != current_canonical)
        if accepted:
            import_rows.append(
                {
                    "word_id": base_row.get("word_id"),
                    "recording_id": base_row.get("recording_id"),
                    "recording_ipa": chosen_ipa,
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
                "word_id": base_row.get("word_id"),
                "recording_id": base_row.get("recording_id"),
                "word_title": word_title,
                "word_translation": word_translation,
                "recording_type": recording_type,
                "current_recording_ipa": current_ipa,
                "current_source": current_source,
                "current_canonical": current_canonical,
                "current_rule_count": int(current_info["final_rule_count"]),
                "current_csv_supported_typed": current_csv_supported_typed,
                "current_csv_supported_any": current_csv_supported_any,
                "csv_candidate_count": len(csv_entries),
                "csv_candidates": sorted({entry["phonetic"] for entry in csv_entries}),
                "new_mms_recording_ipa": model_infos[0]["raw"],
                "old_mms_recording_ipa": model_infos[1]["raw"],
                "whisper_recording_ipa": model_infos[2]["raw"],
                "new_mms_dialect_repaired": model_infos[0]["dialect_repaired"],
                "old_mms_dialect_repaired": model_infos[1]["dialect_repaired"],
                "whisper_dialect_repaired": model_infos[2]["dialect_repaired"],
                "candidate_buckets": [
                    {
                        "canonical": bucket["canonical"],
                        "chosen_ipa": bucket["chosen_ipa"],
                        "models": sorted(bucket["models"]),
                        "support_count": int(bucket["support_count"]),
                        "csv_support_typed": bool(bucket["csv_support_typed"]),
                        "csv_support_any": bool(bucket["csv_support_any"]),
                        "best_final_rule_count": int(bucket["best_final_rule_count"]),
                    }
                    for bucket in sorted_buckets
                ],
                "accepted": accepted,
                "decision_reason": decision_reason,
                "chosen_recording_ipa": chosen_ipa,
            }
        )

    import_path = output_dir / "full-audio-threeway-high-confidence-import.jsonl"
    with import_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in import_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit_path = output_dir / "full-audio-threeway-high-confidence-audit.jsonl"
    with audit_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "new_path": str(new_path),
        "old_mms_path": str(old_mms_path),
        "whisper_path": str(whisper_path),
        "csv_path": str(csv_path),
        "overlay_updates": [str(path) for path in overlay_paths],
        "total_common_rows": len(common_ids),
        "accepted_rows": int(stats["accepted_rows"]),
        "accepted_current_overrides": int(stats["accepted_current_overrides"]),
        "accepted_new_resolutions": int(stats["accepted_new_resolutions"]),
        "rows_with_current_ipa": int(stats["rows_with_current_ipa"]),
        "rows_without_current_ipa": int(stats["rows_without_current_ipa"]),
        "current_supported_by_models": int(stats["current_supported_by_models"]),
        "rejected_rows": int(stats["rejected_rows"]),
        "decision_reason_counts": dict(
            Counter(
                str(row["decision_reason"])
                for row in audit_rows
                if row.get("decision_reason")
            )
        ),
    }
    summary_path = output_dir / "full-audio-threeway-high-confidence-summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote import JSONL: {import_path}")
    print(f"Wrote audit JSONL:  {audit_path}")
    print(f"Wrote summary:      {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
