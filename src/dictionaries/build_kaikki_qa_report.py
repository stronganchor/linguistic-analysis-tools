#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from _dictionary_pipeline_common import SCRIPT_DIR, clean_cell, ensure_parent_dir


QUEUE_PATH = SCRIPT_DIR / "tsv_output" / "open_source_review" / "open_source_import_review_queue.tsv"
OUTPUT_DIR = SCRIPT_DIR / "tsv_output" / "open_source_review"
FLAGS_PATH = OUTPUT_DIR / "kaikki_qa_flags.tsv"
REPORT_PATH = OUTPUT_DIR / "kaikki_qa_report.md"


FLAG_FIELDS = [
    "issue_id",
    "severity",
    "recommendation",
    "issue_type",
    "review_id",
    "source_locator",
    "headword",
    "entry_type",
    "source_gloss_en",
    "source_gloss_raw",
    "auto_flags",
    "issue_note",
]

ALT_OR_MISSPELLING_RE = re.compile(r"\b(alternative form of|misspelling of)\b", re.I)
ABBREV_OR_ACRONYM_RE = re.compile(r"\b(abbreviation of|acronym of|initialism of)\b", re.I)
INFLECTED_FORM_RE = re.compile(r"\b(plural of|passive of|verbal noun of|diminutive of)\b", re.I)
SENSITIVE_RE = re.compile(r"\b(vulgar|slang|fuck|whore|faggot|fag|cuckold|bitch|slut)\b", re.I)
NUMERAL_COMPOUND_RE = re.compile(r"^(des|vist|hirıs|çewres) u ")

MANUAL_GLOSS_ISSUES: dict[str, tuple[str, str, str, str]] = {
    "kaikki:0067:01": (
        "major",
        "review_before_use",
        "gloss_issue",
        'English gloss says "father (sort form)", which looks like a typo for "short form".',
    ),
    "kaikki:0100:01": (
        "minor",
        "review_before_use",
        "gloss_issue",
        'Gloss is formatted as "A lake." with odd capitalization and punctuation for a dictionary gloss.',
    ),
    "kaikki:0384:01": (
        "critical",
        "exclude_until_verified",
        "gloss_issue",
        "Proverb gloss looks corrupted and does not match the literal note in etymology_text.",
    ),
    "kaikki:0412:01": (
        "major",
        "review_before_use",
        "gloss_issue",
        'Gloss "A Japanese" is awkward English and may reflect bad translation or incomplete sense labeling.',
    ),
    "kaikki:0636:01": (
        "major",
        "exclude_until_fixed",
        "gloss_issue",
        "Gloss has unmatched parentheses: alternative form of mês (“fly (insect”).",
    ),
    "kaikki:0688:02": (
        "minor",
        "review_before_use",
        "gloss_issue",
        'Gloss is styled as "persona: ..." instead of a normal dictionary gloss.',
    ),
    "kaikki:0834:01": (
        "major",
        "exclude_until_fixed",
        "gloss_issue",
        'Gloss says "past tence form to tırıten.", which contains an obvious typo and awkward grammar.',
    ),
    "kaikki:0964:01": (
        "major",
        "exclude_until_fixed",
        "gloss_issue",
        'Gloss says "to make someone smmile", with an obvious typo.',
    ),
    "kaikki:1008:01": (
        "major",
        "exclude_until_fixed",
        "gloss_issue",
        'Gloss says "to be turn over" instead of idiomatic English such as "to be turned over".',
    ),
    "kaikki:1020:01": (
        "major",
        "exclude_until_fixed",
        "gloss_issue",
        'Gloss says "verbal noun of split off", which is malformed and likely lost the lemma target.',
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a QA report for Kaikki entries in the open-source review queue.")
    parser.add_argument("--queue-path", default=str(QUEUE_PATH))
    parser.add_argument("--flags-path", default=str(FLAGS_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    return parser.parse_args()


def load_queue_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def make_flag(
    row: dict[str, str],
    *,
    issue_id: str,
    severity: str,
    recommendation: str,
    issue_type: str,
    issue_note: str,
) -> dict[str, str]:
    return {
        "issue_id": issue_id,
        "severity": severity,
        "recommendation": recommendation,
        "issue_type": issue_type,
        "review_id": clean_cell(row.get("review_id")),
        "source_locator": clean_cell(row.get("source_locator")),
        "headword": clean_cell(row.get("headword")),
        "entry_type": clean_cell(row.get("entry_type")),
        "source_gloss_en": clean_cell(row.get("source_gloss_en")),
        "source_gloss_raw": clean_cell(row.get("source_gloss_raw")),
        "auto_flags": clean_cell(row.get("auto_flags")),
        "issue_note": issue_note,
    }


def build_flags(kaikki_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for row in kaikki_rows:
        review_id = clean_cell(row.get("review_id"))
        headword = clean_cell(row.get("headword"))
        gloss_en = clean_cell(row.get("source_gloss_en"))
        gloss_raw = clean_cell(row.get("source_gloss_raw"))
        entry_type = clean_cell(row.get("entry_type"))
        auto_flags = clean_cell(row.get("auto_flags"))

        if headword == "a/languages M to Z":
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:parser_artifact",
                    severity="critical",
                    recommendation="exclude",
                    issue_type="parser_artifact",
                    issue_note="Corrupted headword produced by Kaikki/Wiktionary parsing; not a real Zazaki entry.",
                )
            )

        if clean_cell(row.get("candidate_kind")) == "skip_candidate":
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:structural_nonlemma",
                    severity="high",
                    recommendation="exclude",
                    issue_type="structural_nonlemma",
                    issue_note="Character/suffix/bound-form style entry; generally unsuitable for a public learner dictionary headword list.",
                )
            )
        elif entry_type == "postp" and headword.startswith("-"):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:bound_postposition",
                    severity="high",
                    recommendation="exclude_or_notes_only",
                    issue_type="structural_nonlemma",
                    issue_note="Bound postposition entry begins with a hyphen; likely not suitable as a standalone public headword.",
                )
            )

        combined_gloss = " || ".join(part for part in [gloss_en, gloss_raw] if part)
        if ALT_OR_MISSPELLING_RE.search(combined_gloss):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:alias_or_misspelling",
                    severity="medium",
                    recommendation="alias_only_or_skip",
                    issue_type="redirect_like_entry",
                    issue_note="Alternative-form or misspelling entry. Useful for search aliases, but not as a full standalone dictionary sense.",
                )
            )
        if ABBREV_OR_ACRONYM_RE.search(combined_gloss):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:abbreviation_or_acronym",
                    severity="medium",
                    recommendation="alias_only_or_skip",
                    issue_type="redirect_like_entry",
                    issue_note="Abbreviation/acronym/initialism entry. Probably not worth importing as a normal dictionary entry.",
                )
            )
        if INFLECTED_FORM_RE.search(combined_gloss):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:inflected_or_derived_form",
                    severity="medium",
                    recommendation="skip_as_entry",
                    issue_type="redirect_like_entry",
                    issue_note="Inflected/passive/verbal-noun/diminutive style row. Better treated as morphology or cross-reference, not a full entry.",
                )
            )

        if NUMERAL_COMPOUND_RE.match(headword) and any(
            token in gloss_en
            for token in [
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
                "twenty-",
                "thirty-",
                "forty-",
            ]
        ):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:mechanical_numeral",
                    severity="medium",
                    recommendation="policy_review",
                    issue_type="mechanical_compound",
                    issue_note="Predictable spelled-out numeral compound; probably low value as a separate dictionary headword.",
                )
            )

        if entry_type == "name":
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:proper_noun_policy",
                    severity="low",
                    recommendation="policy_review",
                    issue_type="proper_noun_policy",
                    issue_note="Proper noun or name entry. Decide separately whether the core dictionary should include names, countries, and language names.",
                )
            )

        if SENSITIVE_RE.search(" || ".join([gloss_en, gloss_raw])):
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:sensitive_lexis",
                    severity="low",
                    recommendation="policy_review",
                    issue_type="sensitive_lexis",
                    issue_note="Sensitive/vulgar/slur content. Not necessarily an error, but may need product-policy review before publication.",
                )
            )

        manual = MANUAL_GLOSS_ISSUES.get(review_id)
        if manual is not None:
            severity, recommendation, issue_type, issue_note = manual
            flags.append(
                make_flag(
                    row,
                    issue_id=f"{review_id}:{issue_type}",
                    severity=severity,
                    recommendation=recommendation,
                    issue_type=issue_type,
                    issue_note=issue_note,
                )
            )

    return flags


def count_unique_rows(flags: list[dict[str, str]], issue_type: str) -> int:
    return len({flag["review_id"] for flag in flags if flag["issue_type"] == issue_type})


def sample_lines(flags: list[dict[str, str]], issue_type: str, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for flag in flags:
        if flag["issue_type"] != issue_type:
            continue
        review_id = flag["review_id"]
        if review_id in seen:
            continue
        seen.add(review_id)
        lines.append(
            f"- `{flag['review_id']}` at `{flag['source_locator']}`: `{flag['headword']}` ({flag['entry_type']}) - {flag['issue_note']}"
        )
        if len(lines) >= limit:
            break
    return lines


def build_report(kaikki_rows: list[dict[str, str]], flags: list[dict[str, str]]) -> str:
    issue_counts = Counter(flag["issue_type"] for flag in flags)
    severity_counts = Counter(flag["severity"] for flag in flags)
    unique_flagged_rows = len({flag["review_id"] for flag in flags})

    lines: list[str] = []
    lines.append("# Kaikki QA Report")
    lines.append("")
    lines.append("This report reviews the Kaikki-derived candidate rows in the open-source review queue, not the master dictionary.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Kaikki candidate rows reviewed: `{len(kaikki_rows)}`")
    lines.append(f"- Unique rows flagged by QA heuristics/manual review: `{unique_flagged_rows}`")
    lines.append(f"- Critical findings: `{severity_counts.get('critical', 0)}`")
    lines.append(f"- High-severity findings: `{severity_counts.get('high', 0)}`")
    lines.append(f"- Major findings: `{severity_counts.get('major', 0)}`")
    lines.append(f"- Medium-severity findings: `{severity_counts.get('medium', 0)}`")
    lines.append(f"- Low-severity findings: `{severity_counts.get('low', 0)}`")
    lines.append(f"- Minor findings: `{severity_counts.get('minor', 0)}`")
    lines.append("")
    lines.append("## High-Confidence Exclusions")
    lines.append("")
    lines.append(f"- `parser_artifact`: `{count_unique_rows(flags, 'parser_artifact')}` unique rows")
    lines.extend(sample_lines(flags, "parser_artifact", limit=4))
    lines.append(f"- `structural_nonlemma`: `{count_unique_rows(flags, 'structural_nonlemma')}` unique rows")
    lines.extend(sample_lines(flags, "structural_nonlemma", limit=8))
    lines.append("")
    lines.append("## Redirect / Morphology Rows")
    lines.append("")
    lines.append(f"- `redirect_like_entry`: `{count_unique_rows(flags, 'redirect_like_entry')}` unique rows")
    lines.append("- These are mostly alternative forms, misspellings, abbreviations, acronyms, passive forms, plurals, and verbal nouns.")
    lines.extend(sample_lines(flags, "redirect_like_entry", limit=10))
    lines.append("")
    lines.append("## Policy-Review Buckets")
    lines.append("")
    lines.append(f"- `mechanical_compound`: `{count_unique_rows(flags, 'mechanical_compound')}` unique rows")
    lines.append("- Most are predictable numeral compounds like `des u yew`, `vist u yew`, `hirıs u yew`, and `çewres u yew`.")
    lines.extend(sample_lines(flags, "mechanical_compound", limit=6))
    lines.append(f"- `proper_noun_policy`: `{count_unique_rows(flags, 'proper_noun_policy')}` unique rows")
    lines.append("- These include countries, language names, personal names, and abbreviations of geopolitical entities.")
    lines.extend(sample_lines(flags, "proper_noun_policy", limit=8))
    lines.append(f"- `sensitive_lexis`: `{count_unique_rows(flags, 'sensitive_lexis')}` unique rows")
    lines.append("- These appear lexically valid but may need a product-policy decision before public display.")
    lines.extend(sample_lines(flags, "sensitive_lexis", limit=5))
    lines.append("")
    lines.append("## Gloss / Data Quality Problems")
    lines.append("")
    lines.append(f"- `gloss_issue`: `{count_unique_rows(flags, 'gloss_issue')}` unique rows")
    lines.extend(sample_lines(flags, "gloss_issue", limit=20))
    lines.append("")
    lines.append("## Recommended Actions")
    lines.append("")
    lines.append("1. Exclude the parser-artifact and structural non-lemma rows from import.")
    lines.append("2. Treat alternative-form / inflection rows as alias-only or skip them entirely as standalone entries.")
    lines.append("3. Review whether predictable multiword numerals and proper nouns belong in the core dictionary at all.")
    lines.append("4. Fix or reject the gloss-quality problem rows before any merge.")
    lines.append("5. Manually verify `her çı zano once buro` before use; it is the strongest sign of corrupted or mismatched source data.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- Flagged-row TSV: `tsv_output/open_source_review/kaikki_qa_flags.tsv`")
    lines.append("- This report: `tsv_output/open_source_review/kaikki_qa_report.md`")
    lines.append("")
    lines.append("## Issue Counts")
    lines.append("")
    for issue_type, count in sorted(issue_counts.items()):
        lines.append(f"- `{issue_type}`: `{count}` flag rows")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FLAG_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field)) for field in FLAG_FIELDS})


def main() -> None:
    args = parse_args()
    rows = load_queue_rows(Path(args.queue_path))
    kaikki_rows = [row for row in rows if clean_cell(row.get("source_family")) == "kaikki"]
    flags = build_flags(kaikki_rows)
    flags.sort(key=lambda row: (row["severity"], row["issue_type"], row["source_locator"], row["review_id"]))

    flags_path = Path(args.flags_path)
    report_path = Path(args.report_path)
    write_tsv(flags_path, flags)
    ensure_parent_dir(report_path)
    report_path.write_text(build_report(kaikki_rows, flags), encoding="utf-8")

    print(
        {
            "kaikki_rows": len(kaikki_rows),
            "flag_rows": len(flags),
            "unique_flagged_rows": len({flag["review_id"] for flag in flags}),
            "flags_path": str(flags_path),
            "report_path": str(report_path),
        }
    )


if __name__ == "__main__":
    main()
