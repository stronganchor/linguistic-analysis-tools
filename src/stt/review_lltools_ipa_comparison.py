#!/usr/bin/env python3
import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from genc_palu_dialect_rules import summarize_genc_palu_dialect_repairs
from ipa_rule_checks import basic_repair_ipa_text, summarize_ipa_rule_issues


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PRIMARY_SOURCES = {
    "word_translation",
    "same_word_existing",
    "same_word_exact",
    "manifest_title",
}
VOWELS = set("aeiou")


def default_dictionary_tsv() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "Dictionaries" / "tsv_output" / "Combined - Harun Turgut + DEZD.tsv",
        script_dir.parent / "Dictionaries" / "tsv_output" / "Palu - Bingöl Harun Turgut - corrected.tsv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Review an LL Tools IPA model-comparison JSONL, keep exact matches, "
            "and resolve a conservative subset of disagreements using bundle glosses, "
            "existing IPA, manifest overlap, and dictionary support."
        )
    )
    parser.add_argument("--comparison-path", required=True, help="Joined comparison JSONL path.")
    parser.add_argument(
        "--extract-dir",
        required=True,
        help="Root folder containing extracted LL Tools full-bundle data.json files.",
    )
    parser.add_argument(
        "--manifest-path",
        default=str(script_dir / "training_data" / "manifest_ipa.tsv"),
        help="Existing A.K. IPA manifest TSV.",
    )
    parser.add_argument(
        "--dictionary-tsv",
        default=str(default_dictionary_tsv()),
        help="Dictionary TSV used for exact-title supplemental evidence.",
    )
    parser.add_argument("--output-dir", required=True, help="Folder where outputs should be written.")
    parser.add_argument(
        "--left-field",
        default="mms_recording_ipa",
        help="Comparison JSONL field name for the first model output.",
    )
    parser.add_argument(
        "--right-field",
        default="whisper_medium_recording_ipa",
        help="Comparison JSONL field name for the second model output.",
    )
    parser.add_argument(
        "--left-label",
        default="mms",
        help="User-facing label for the first model.",
    )
    parser.add_argument(
        "--right-label",
        default="whisper_medium",
        help="User-facing label for the second model.",
    )
    return parser.parse_args()


def normalize_title_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").lower().strip()
    replacements = {
        "ı": "i",
        "î": "i",
        "ï": "i",
        "ê": "e",
        "é": "e",
        "â": "a",
        "û": "u",
        "ü": "u",
        "ö": "o",
        "ş": "s",
        "ç": "c",
        "ğ": "g",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def orthography_to_canonical(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "").lower().strip()
    replacements = {
        "ç": "C",
        "c": "J",
        "ş": "S",
        "j": "J",
        "y": "Y",
        "ı": "i",
        "î": "i",
        "ï": "i",
        "ê": "e",
        "é": "e",
        "â": "a",
        "û": "u",
        "ü": "u",
        "ö": "o",
        "ğ": "g",
        "w": "W",
        "q": "X",
        "x": "X",
        "’": "",
        "'": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ipa_to_canonical(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    replacements = [
        ("t̪͡ʃ", "C"),
        ("t͡ʃ", "C"),
        ("c͡ç", "C"),
        ("tʃ", "C"),
        ("d̪͡ʒ", "J"),
        ("d͡ʒ", "J"),
        ("ʒ", "J"),
        ("ʃ", "S"),
        ("j", "Y"),
        ("qʰʷ", "XW"),
        ("kʰʷ", "XW"),
        ("qʷ", "XW"),
        ("kʷ", "XW"),
        ("gʷ", "XW"),
        ("χʷ", "XW"),
        ("xʷ", "XW"),
        ("ħʷ", "XW"),
        ("qʰ", "X"),
        ("kʰ", "X"),
        ("q", "X"),
        ("χ", "X"),
        ("ħ", "X"),
        ("x", "X"),
        ("w", "W"),
        ("ɨ", "i"),
        ("ɪ", "i"),
        ("ɛ", "e"),
        ("æ", "a"),
        ("ɾ", "r"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(r"[ˈˌːʰʲʷ̪̥̞̯̆̃ⁿ]", "", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Za-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_variants(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|,/;()]", value or "") if part.strip()]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for index_a, char_a in enumerate(a, 1):
        current = [index_a]
        for index_b, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    previous[index_b] + 1,
                    current[index_b - 1] + 1,
                    previous[index_b - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def consonant_skeleton(value: str) -> str:
    return "".join(char for char in value if char.lower() not in VOWELS and char != " ")


def best_window_score(candidate: str, reference: str) -> tuple[float, str]:
    candidate_tokens = candidate.split()
    reference_tokens = reference.split()
    target = "".join(reference_tokens)
    if not candidate_tokens or not target:
        return 1.0, ""

    best_score = 9.0
    best_piece = ""
    min_length = max(2, len(target) - 2)
    max_window_tokens = min(len(candidate_tokens), max(1, len(reference_tokens) + 3))

    def consider(piece: str) -> None:
        nonlocal best_score, best_piece
        distance = levenshtein(piece, target) / max(len(piece), len(target), 1)
        piece_skeleton = consonant_skeleton(piece)
        target_skeleton = consonant_skeleton(target)
        skeleton_distance = (
            levenshtein(piece_skeleton, target_skeleton)
            / max(len(piece_skeleton), len(target_skeleton), 1)
            if (piece_skeleton or target_skeleton)
            else distance
        )
        score = 0.7 * distance + 0.3 * skeleton_distance
        if score < best_score:
            best_score = score
            best_piece = piece

    for start in range(len(candidate_tokens)):
        for end in range(start + 1, min(len(candidate_tokens), start + max_window_tokens) + 1):
            piece = "".join(candidate_tokens[start:end])
            if len(piece) < min_length:
                continue
            consider(piece)

    if best_score == 9.0:
        for start in range(len(candidate_tokens)):
            for end in range(start + 1, len(candidate_tokens) + 1):
                consider("".join(candidate_tokens[start:end]))

    return best_score, best_piece


def choose_source_pick(
    left_score: float,
    right_score: float,
    *,
    strong_margin: float,
    weak_margin: float,
    strong_abs: float,
    weak_abs: float,
) -> tuple[str, str]:
    if left_score <= strong_abs and left_score + strong_margin <= right_score:
        return "left", "strong"
    if right_score <= strong_abs and right_score + strong_margin <= left_score:
        return "right", "strong"
    if left_score <= weak_abs and left_score + weak_margin <= right_score:
        return "left", "weak"
    if right_score <= weak_abs and right_score + weak_margin <= left_score:
        return "right", "weak"
    return "", ""


def choose_cleaner_side(
    left_issue_count: int,
    right_issue_count: int,
) -> tuple[str, str]:
    if left_issue_count == right_issue_count:
        return "", ""
    if left_issue_count == 0 and right_issue_count > 0:
        return "left", "strong"
    if right_issue_count == 0 and left_issue_count > 0:
        return "right", "strong"
    if left_issue_count + 2 <= right_issue_count:
        return "left", "weak"
    if right_issue_count + 2 <= left_issue_count:
        return "right", "weak"
    return "", ""


def load_word_metadata(extract_dir: Path) -> dict[int, dict[str, object]]:
    metadata: dict[int, dict[str, object]] = {}
    for data_json_path in sorted(extract_dir.rglob("data.json")):
        with data_json_path.open("r", encoding="utf-8") as handle:
            bundle_data = json.load(handle)
        for word in bundle_data.get("words") or []:
            origin_id = word.get("origin_id")
            if origin_id in {None, ""}:
                continue
            word_id = int(origin_id)
            meta = word.get("meta") or {}
            existing_ipa: list[tuple[int, str]] = []
            for entry in word.get("audio_entries") or []:
                entry_id = entry.get("origin_id")
                if entry_id in {None, ""}:
                    continue
                entry_meta = entry.get("meta") or {}
                values = entry_meta.get("recording_ipa") or []
                ipa_value = str(values[0] or "").strip() if values else ""
                if ipa_value:
                    existing_ipa.append((int(entry_id), ipa_value))
            metadata[word_id] = {
                "word_title": str(word.get("title") or "").strip(),
                "word_translation": str((meta.get("word_translation") or [""])[0] or "").strip(),
                "existing_ipa": existing_ipa,
            }
    return metadata


def load_manifest_title_refs(path: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            title = normalize_title_text(row.get("word_title") or "")
            ipa_value = str(row.get("target_ipa") or "").strip()
            if title and ipa_value:
                refs[title].append(ipa_value)
    return refs


def load_dictionary_exact_title_refs(path: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            entry = str(row.get("entry") or "").strip()
            if not entry:
                continue
            title_values = []
            raw_title_keys = str(row.get("title_keys") or "").strip()
            if raw_title_keys:
                title_values.extend(part.strip() for part in raw_title_keys.split("|"))
            else:
                title_values.append(str(row.get("definition") or ""))
            for title_value in title_values:
                title = normalize_title_text(title_value)
                if title and entry:
                    refs[title].append(entry)
    return refs


def load_comparison_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda row: int(str(row.get("recording_id") or "0") or "0"))
    return rows


def build_summary(
    *,
    comparison_path: Path,
    update_path: Path,
    audit_path: Path,
    left_label: str,
    right_label: str,
    total_rows: int,
    exact_rows: int,
    resolved_disagreements: int,
    unresolved_disagreements: int,
    update_rows: list[dict],
    audit_rows: list[dict],
) -> dict[str, object]:
    resolved_rows = [row for row in audit_rows if row.get("resolution_status") == "resolved"]
    unresolved_rows = [row for row in audit_rows if row.get("resolution_status") == "unresolved"]
    return {
        "comparison_path": str(comparison_path),
        "update_jsonl": str(update_path),
        "audit_jsonl": str(audit_path),
        "total_rows": total_rows,
        "exact_rows": exact_rows,
        "resolved_disagreements": resolved_disagreements,
        "unresolved_disagreements": unresolved_disagreements,
        "total_update_rows": len(update_rows),
        "selected_model_counts": dict(Counter(str(row.get("selected_model") or "") for row in update_rows)),
        "resolution_type_counts": dict(Counter(str(row.get("resolution_type") or "") for row in update_rows)),
        "resolved_by_category_slug": dict(
            Counter(str(row.get("category_slug") or "") for row in resolved_rows)
        ),
        "unresolved_by_category_slug": dict(
            Counter(str(row.get("category_slug") or "") for row in unresolved_rows)
        ),
        "labels": {
            "left": left_label,
            "right": right_label,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_notes(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# Missing IPA Review Pass",
        "",
        f"- Comparison source: `{summary['comparison_path']}`",
        f"- Update JSONL: `{summary['update_jsonl']}`",
        f"- Audit JSONL: `{summary['audit_jsonl']}`",
        f"- Total rows reviewed: `{summary['total_rows']}`",
        f"- Exact model agreement rows kept: `{summary['exact_rows']}`",
        f"- Resolved disagreements: `{summary['resolved_disagreements']}`",
        f"- Unresolved disagreements left out: `{summary['unresolved_disagreements']}`",
        f"- Total update rows: `{summary['total_update_rows']}`",
        "",
        "## Resolution policy",
        "",
        "- Exact MMS/Whisper agreement is kept when clean; exact-agreement rows with obvious IPA rule corruption are repaired or held out.",
        "- IPA rule checks now flag misplaced aspiration, orphan modifiers, repeated modifiers, and orphan tie bars.",
        "- When a chosen form has only rule-level corruption and a clean repair is obvious, the repaired form is written instead.",
        "- Non-exact rows were resolved only when bundle glosses, same-word IPA anchors, manifest overlap, or canonical-equivalent outputs gave conservative support.",
        "- Dictionary evidence was used only as supporting evidence, not as a decisive source by itself.",
        "",
        "## Selected model counts",
        "",
    ]
    for key, value in sorted((summary.get("selected_model_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Resolution type counts",
            "",
        ]
    )
    for key, value in sorted((summary.get("resolution_type_counts") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    comparison_path = Path(args.comparison_path).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    manifest_path = Path(args.manifest_path).expanduser().resolve()
    dictionary_tsv = Path(args.dictionary_tsv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    update_path = output_dir / "missing-non-cerez-recording_ipa-update.jsonl"
    audit_path = output_dir / "missing-non-cerez-resolution-audit.jsonl"
    summary_path = output_dir / "missing-non-cerez-resolution.summary.json"
    notes_path = output_dir / "missing-non-cerez-review-notes.md"

    comparison_rows = load_comparison_rows(comparison_path)
    word_metadata = load_word_metadata(extract_dir)
    manifest_title_refs = load_manifest_title_refs(manifest_path)
    dictionary_exact_title_refs = load_dictionary_exact_title_refs(dictionary_tsv)

    rows_by_word: dict[int, list[dict]] = defaultdict(list)
    for row in comparison_rows:
        word_id = row.get("word_id")
        if isinstance(word_id, int):
            rows_by_word[word_id].append(row)

    update_rows: list[dict] = []
    audit_rows: list[dict] = []
    exact_rows = 0
    resolved_disagreements = 0
    unresolved_disagreements = 0

    for row in comparison_rows:
        word_id = int(row.get("word_id") or 0)
        recording_id = int(row.get("recording_id") or 0)
        word_meta = word_metadata.get(
            word_id,
            {"word_title": "", "word_translation": "", "existing_ipa": []},
        )
        word_title = str(word_meta.get("word_title") or "")
        word_translation = str(word_meta.get("word_translation") or "")

        left_prediction = str(row.get(args.left_field) or "")
        right_prediction = str(row.get(args.right_field) or "")
        left_rule_summary = summarize_ipa_rule_issues(left_prediction)
        right_rule_summary = summarize_ipa_rule_issues(right_prediction)
        left_issue_count = int(left_rule_summary["count"])
        right_issue_count = int(right_rule_summary["count"])
        left_repaired = basic_repair_ipa_text(left_prediction)
        right_repaired = basic_repair_ipa_text(right_prediction)
        left_repaired_issue_count = int(summarize_ipa_rule_issues(left_repaired)["count"])
        right_repaired_issue_count = int(summarize_ipa_rule_issues(right_repaired)["count"])
        left_dialect_summary = summarize_genc_palu_dialect_repairs(
            left_repaired,
            word_title=word_title,
            word_translation=word_translation,
        )
        right_dialect_summary = summarize_genc_palu_dialect_repairs(
            right_repaired,
            word_title=word_title,
            word_translation=word_translation,
        )
        left_dialect_repaired = str(left_dialect_summary["repaired"] or "")
        right_dialect_repaired = str(right_dialect_summary["repaired"] or "")
        left_dialect_issue_count = int(summarize_ipa_rule_issues(left_dialect_repaired)["count"])
        right_dialect_issue_count = int(summarize_ipa_rule_issues(right_dialect_repaired)["count"])
        left_canonical = ipa_to_canonical(left_prediction)
        right_canonical = ipa_to_canonical(right_prediction)

        base_audit = {
            "word_id": word_id,
            "recording_id": recording_id,
            "word_title": word_title,
            "word_translation": word_translation,
            "word_slug": row.get("word_slug"),
            "recording_slug": row.get("recording_slug"),
            "recording_type": row.get("recording_type"),
            "speaker_user_id": row.get("speaker_user_id"),
            "source_bundle": row.get("source_bundle"),
            "category_slug": row.get("category_slug"),
            "category_name": row.get("category_name"),
            args.left_field: left_prediction,
            args.right_field: right_prediction,
            f"{args.left_label}_rule_issue_count": left_issue_count,
            f"{args.left_label}_rule_issue_types": left_rule_summary["types"],
            f"{args.left_label}_rule_repaired": left_repaired,
            f"{args.left_label}_dialect_rule_types": left_dialect_summary["types"],
            f"{args.left_label}_dialect_rule_repaired": left_dialect_repaired,
            f"{args.right_label}_rule_issue_count": right_issue_count,
            f"{args.right_label}_rule_issue_types": right_rule_summary["types"],
            f"{args.right_label}_rule_repaired": right_repaired,
            f"{args.right_label}_dialect_rule_types": right_dialect_summary["types"],
            f"{args.right_label}_dialect_rule_repaired": right_dialect_repaired,
        }

        if row.get("exact_match"):
            exact_rows += 1
            if left_issue_count == 0:
                chosen_text = left_prediction
                evidence_sources = ["exact_model_agreement"]
                resolution_type = "exact_model_agreement"
                if (
                    left_dialect_repaired
                    and left_dialect_repaired != left_prediction
                    and left_dialect_issue_count == 0
                ):
                    chosen_text = left_dialect_repaired
                    evidence_sources.append("genc_palu_dialect_rules")
                    resolution_type = "exact_model_agreement_dialect_rule_repaired"
                update_rows.append(
                    {
                        "word_id": word_id,
                        "recording_id": recording_id,
                        "recording_ipa": chosen_text,
                        "selected_model": "both",
                        "resolution_type": resolution_type,
                        "word_title": word_title,
                        "word_translation": word_translation,
                        "evidence_sources": evidence_sources,
                    }
                )
                audit_rows.append(
                    {
                        **base_audit,
                        "resolution_status": "resolved",
                        "selected_model": "both",
                        "resolution_type": resolution_type,
                        "chosen_recording_ipa": chosen_text,
                        "evidence": [{"source": source} for source in evidence_sources],
                    }
                )
            elif left_repaired == right_repaired and left_repaired and left_repaired_issue_count == 0:
                update_rows.append(
                    {
                        "word_id": word_id,
                        "recording_id": recording_id,
                        "recording_ipa": left_repaired,
                        "selected_model": "both",
                        "resolution_type": "exact_model_agreement_rule_repaired",
                        "word_title": word_title,
                        "word_translation": word_translation,
                        "evidence_sources": ["exact_model_agreement", "ipa_rule_repair"],
                    }
                )
                audit_rows.append(
                    {
                        **base_audit,
                        "resolution_status": "resolved",
                        "selected_model": "both",
                        "resolution_type": "exact_model_agreement_rule_repaired",
                        "chosen_recording_ipa": left_repaired,
                        "evidence": [
                            {"source": "exact_model_agreement"},
                            {"source": "ipa_rule_repair", "repaired_text": left_repaired},
                        ],
                    }
                )
            else:
                unresolved_disagreements += 1
                audit_rows.append(
                    {
                        **base_audit,
                        "resolution_status": "unresolved",
                        "selected_model": "",
                        "resolution_type": "exact_model_agreement_rule_issue",
                        "chosen_recording_ipa": "",
                        "evidence": [{"source": "exact_model_agreement"}],
                    }
                )
            continue

        if (
            left_dialect_repaired
            and left_dialect_repaired == right_dialect_repaired
            and left_dialect_issue_count == 0
            and right_dialect_issue_count == 0
        ):
            resolved_disagreements += 1
            update_rows.append(
                {
                    "word_id": word_id,
                    "recording_id": recording_id,
                    "recording_ipa": left_dialect_repaired,
                    "selected_model": "dialect_rule_merge",
                    "resolution_type": "high_conf_dialect_rule_equivalence",
                    "word_title": word_title,
                    "word_translation": word_translation,
                    "evidence_sources": ["genc_palu_dialect_rules"],
                }
            )
            audit_rows.append(
                {
                    **base_audit,
                    "resolution_status": "resolved",
                    "selected_model": "dialect_rule_merge",
                    "resolution_type": "high_conf_dialect_rule_equivalence",
                    "chosen_recording_ipa": left_dialect_repaired,
                    "evidence": [{"source": "genc_palu_dialect_rules"}],
                }
            )
            continue

        if left_canonical.replace(" ", "") == right_canonical.replace(" ", ""):
            chosen_side = ""
            if left_issue_count != right_issue_count:
                chosen_side = "left" if left_issue_count < right_issue_count else "right"
            if not chosen_side:
                chosen_side = (
                    "left"
                    if len(left_prediction.replace(" ", "")) <= len(right_prediction.replace(" ", ""))
                    else "right"
                )
            chosen_model = args.left_label if chosen_side == "left" else args.right_label
            chosen_text = left_prediction if chosen_side == "left" else right_prediction
            chosen_repaired = left_repaired if chosen_side == "left" else right_repaired
            chosen_repaired_issue_count = (
                left_repaired_issue_count if chosen_side == "left" else right_repaired_issue_count
            )
            evidence_sources = ["canonical_equivalence"]
            resolution_type = "high_conf_canonical_equivalence"
            if chosen_repaired != chosen_text and chosen_repaired and chosen_repaired_issue_count == 0:
                chosen_text = chosen_repaired
                evidence_sources.append("ipa_rule_repair")
                resolution_type = "high_conf_canonical_equivalence_rule_repaired"
            chosen_dialect = left_dialect_repaired if chosen_side == "left" else right_dialect_repaired
            chosen_dialect_issue_count = left_dialect_issue_count if chosen_side == "left" else right_dialect_issue_count
            if chosen_dialect and chosen_dialect != chosen_text and chosen_dialect_issue_count == 0:
                chosen_text = chosen_dialect
                evidence_sources.append("genc_palu_dialect_rules")
                resolution_type = "high_conf_canonical_equivalence_dialect_rule_repaired"
            resolved_disagreements += 1
            update_rows.append(
                {
                    "word_id": word_id,
                    "recording_id": recording_id,
                    "recording_ipa": chosen_text,
                    "selected_model": chosen_model,
                    "resolution_type": resolution_type,
                    "word_title": word_title,
                    "word_translation": word_translation,
                    "evidence_sources": evidence_sources,
                }
            )
            audit_rows.append(
                {
                    **base_audit,
                    "resolution_status": "resolved",
                    "selected_model": chosen_model,
                    "resolution_type": resolution_type,
                    "chosen_recording_ipa": chosen_text,
                    "evidence": [{"source": source} for source in evidence_sources],
                }
            )
            continue

        votes: list[dict[str, object]] = []

        if word_translation:
            refs = [
                orthography_to_canonical(item)
                for item in split_variants(word_translation)
                if orthography_to_canonical(item)
            ]
            if refs:
                left_score = min(best_window_score(left_canonical, ref)[0] for ref in refs)
                right_score = min(best_window_score(right_canonical, ref)[0] for ref in refs)
                side, strength = choose_source_pick(
                    left_score,
                    right_score,
                    strong_margin=0.17,
                    weak_margin=0.10,
                    strong_abs=0.30,
                    weak_abs=0.42,
                )
                if side:
                    votes.append(
                        {
                            "source": "word_translation",
                            "pick": args.left_label if side == "left" else args.right_label,
                            "strength": strength,
                            "left_score": left_score,
                            "right_score": right_score,
                            "reference": word_translation,
                        }
                    )

        existing_ipa = [
            ipa_value
            for entry_id, ipa_value in (word_meta.get("existing_ipa") or [])
            if entry_id != recording_id
        ]
        if existing_ipa:
            refs = [ipa_to_canonical(item) for item in existing_ipa if ipa_to_canonical(item)]
            if refs:
                left_score = min(best_window_score(left_canonical, ref)[0] for ref in refs)
                right_score = min(best_window_score(right_canonical, ref)[0] for ref in refs)
                side, strength = choose_source_pick(
                    left_score,
                    right_score,
                    strong_margin=0.15,
                    weak_margin=0.08,
                    strong_abs=0.28,
                    weak_abs=0.38,
                )
                if side:
                    votes.append(
                        {
                            "source": "same_word_existing",
                            "pick": args.left_label if side == "left" else args.right_label,
                            "strength": strength,
                            "left_score": left_score,
                            "right_score": right_score,
                            "reference": existing_ipa[0],
                        }
                    )

        exact_rows_for_word = [
            str(item.get(args.left_field) or "")
            for item in rows_by_word.get(word_id, [])
            if item.get("exact_match")
        ]
        if exact_rows_for_word:
            refs = [ipa_to_canonical(item) for item in exact_rows_for_word if ipa_to_canonical(item)]
            if refs:
                left_score = min(best_window_score(left_canonical, ref)[0] for ref in refs)
                right_score = min(best_window_score(right_canonical, ref)[0] for ref in refs)
                side, strength = choose_source_pick(
                    left_score,
                    right_score,
                    strong_margin=0.15,
                    weak_margin=0.08,
                    strong_abs=0.28,
                    weak_abs=0.38,
                )
                if side:
                    votes.append(
                        {
                            "source": "same_word_exact",
                            "pick": args.left_label if side == "left" else args.right_label,
                            "strength": strength,
                            "left_score": left_score,
                            "right_score": right_score,
                            "reference": exact_rows_for_word[0],
                        }
                    )

        title_key = normalize_title_text(word_title)
        manifest_refs = manifest_title_refs.get(title_key, [])
        if manifest_refs:
            refs = [ipa_to_canonical(item) for item in manifest_refs if ipa_to_canonical(item)]
            if refs:
                left_score = min(best_window_score(left_canonical, ref)[0] for ref in refs)
                right_score = min(best_window_score(right_canonical, ref)[0] for ref in refs)
                side, strength = choose_source_pick(
                    left_score,
                    right_score,
                    strong_margin=0.15,
                    weak_margin=0.08,
                    strong_abs=0.28,
                    weak_abs=0.38,
                )
                if side:
                    votes.append(
                        {
                            "source": "manifest_title",
                            "pick": args.left_label if side == "left" else args.right_label,
                            "strength": strength,
                            "left_score": left_score,
                            "right_score": right_score,
                            "reference": manifest_refs[0],
                        }
                    )

        dictionary_refs = dictionary_exact_title_refs.get(title_key, [])
        if dictionary_refs:
            refs = [
                orthography_to_canonical(item)
                for entry in dictionary_refs
                for item in split_variants(entry)
                if orthography_to_canonical(item)
            ]
            if refs:
                left_score = min(best_window_score(left_canonical, ref)[0] for ref in refs)
                right_score = min(best_window_score(right_canonical, ref)[0] for ref in refs)
                side, strength = choose_source_pick(
                    left_score,
                    right_score,
                    strong_margin=0.17,
                    weak_margin=0.10,
                    strong_abs=0.30,
                    weak_abs=0.42,
                )
                if side:
                    votes.append(
                        {
                            "source": "dictionary_exact_title",
                            "pick": args.left_label if side == "left" else args.right_label,
                            "strength": strength,
                            "left_score": left_score,
                            "right_score": right_score,
                            "reference": dictionary_refs[0],
                        }
                    )

        cleaner_side, cleaner_strength = choose_cleaner_side(left_issue_count, right_issue_count)
        if cleaner_side:
            votes.append(
                {
                    "source": "ipa_rule_checks",
                    "pick": args.left_label if cleaner_side == "left" else args.right_label,
                    "strength": cleaner_strength,
                    "left_issue_count": left_issue_count,
                    "right_issue_count": right_issue_count,
                    "left_repaired": left_repaired,
                    "right_repaired": right_repaired,
                }
            )

        unique_picks = {str(item.get("pick") or "") for item in votes}
        primary_votes = [item for item in votes if str(item.get("source")) in PRIMARY_SOURCES]
        strong_primary_votes = [item for item in primary_votes if item.get("strength") == "strong"]

        resolution_status = "unresolved"
        selected_model = ""
        resolution_type = ""
        chosen_text = ""

        if not votes:
            resolution_type = "unresolved_no_vote"
        elif len(unique_picks) > 1:
            resolution_type = "unresolved_conflicting_sources"
        else:
            selected_model = next(iter(unique_picks))
            if len(primary_votes) >= 2 and strong_primary_votes:
                resolution_status = "resolved"
                resolution_type = "high_conf_multi_source"
            elif (
                len(primary_votes) == 1
                and primary_votes[0].get("source") == "word_translation"
                and primary_votes[0].get("strength") == "strong"
            ):
                resolution_status = "resolved"
                resolution_type = "high_conf_word_translation"
            elif (
                len(primary_votes) == 1
                and primary_votes[0].get("source") in {"same_word_existing", "same_word_exact", "manifest_title"}
                and primary_votes[0].get("strength") == "strong"
            ):
                resolution_status = "resolved"
                resolution_type = "high_conf_anchor"
            elif primary_votes:
                resolution_type = "unresolved_weak_support"
            else:
                resolution_type = "unresolved_dictionary_only"

        if resolution_status == "resolved":
            chosen_text = left_prediction if selected_model == args.left_label else right_prediction
            chosen_repaired = left_repaired if selected_model == args.left_label else right_repaired
            chosen_repaired_issue_count = (
                left_repaired_issue_count if selected_model == args.left_label else right_repaired_issue_count
            )
            if chosen_repaired != chosen_text and chosen_repaired and chosen_repaired_issue_count == 0:
                chosen_text = chosen_repaired
                resolution_type = f"{resolution_type}_rule_repaired"
            chosen_dialect = left_dialect_repaired if selected_model == args.left_label else right_dialect_repaired
            chosen_dialect_issue_count = (
                left_dialect_issue_count if selected_model == args.left_label else right_dialect_issue_count
            )
            if chosen_dialect and chosen_dialect != chosen_text and chosen_dialect_issue_count == 0:
                chosen_text = chosen_dialect
                resolution_type = f"{resolution_type}_dialect_rule_repaired"
            resolved_disagreements += 1
            update_rows.append(
                {
                    "word_id": word_id,
                    "recording_id": recording_id,
                    "recording_ipa": chosen_text,
                    "selected_model": selected_model,
                    "resolution_type": resolution_type,
                    "word_title": word_title,
                    "word_translation": word_translation,
                    "evidence_sources": [
                        *[str(item.get("source") or "") for item in votes],
                        *(
                            ["ipa_rule_repair"]
                            if chosen_repaired != (left_prediction if selected_model == args.left_label else right_prediction)
                            and chosen_repaired
                            and chosen_repaired_issue_count == 0
                            else []
                        ),
                        *(
                            ["genc_palu_dialect_rules"]
                            if chosen_dialect
                            and chosen_dialect
                            != (
                                chosen_repaired
                                if chosen_repaired != (left_prediction if selected_model == args.left_label else right_prediction)
                                and chosen_repaired
                                and chosen_repaired_issue_count == 0
                                else (left_prediction if selected_model == args.left_label else right_prediction)
                            )
                            and chosen_dialect_issue_count == 0
                            else []
                        ),
                    ],
                }
            )
        else:
            unresolved_disagreements += 1

        audit_rows.append(
            {
                **base_audit,
                "resolution_status": resolution_status,
                "selected_model": selected_model,
                "resolution_type": resolution_type,
                "chosen_recording_ipa": chosen_text,
                "evidence": votes,
            }
        )

    write_jsonl(update_path, update_rows)
    write_jsonl(audit_path, audit_rows)
    summary = build_summary(
        comparison_path=comparison_path,
        update_path=update_path,
        audit_path=audit_path,
        left_label=args.left_label,
        right_label=args.right_label,
        total_rows=len(comparison_rows),
        exact_rows=exact_rows,
        resolved_disagreements=resolved_disagreements,
        unresolved_disagreements=unresolved_disagreements,
        update_rows=update_rows,
        audit_rows=audit_rows,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_notes(notes_path, summary)

    print(f"Wrote update JSONL: {update_path}")
    print(f"Wrote audit JSONL:  {audit_path}")
    print(f"Wrote summary:      {summary_path}")
    print(f"Wrote notes:        {notes_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
