#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from _dictionary_pipeline_common import write_json, write_tsv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_PATH = SCRIPT_DIR / "Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK.pdf"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "tsv_output" / "Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK.tsv"
DEFAULT_SUMMARY_PATH = SCRIPT_DIR / "tsv_output" / "Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK.summary.json"

SOURCE_DICTIONARY = "Genç Abdurahman Yilmaz ZAZACA TURKCE SOZLUK"
PDF_START_PAGE = 18
PDF_END_PAGE = 186

LETTER_HEADERS = {
    "A",
    "B",
    "C",
    "Ç",
    "D",
    "E",
    "Ê",
    "F",
    "G",
    "Ğ",
    "H",
    "I",
    "İ",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "Ş",
    "T",
    "U",
    "U/Û",
    "Û",
    "V",
    "W",
    "X",
    "Y",
    "Z",
}

SOURCE_ABBREVIATIONS = {
    "Alm.",
    "Ar.",
    "Bul.",
    "Çi.",
    "Erm.",
    "Fa.",
    "Fr.",
    "İng.",
    "İsp.",
    "İt.",
    "Ku.",
    "Lat.",
    "Moğ.",
    "Rus.",
    "Sı.",
    "Tr.",
    "Yun.",
}

HEAD_WITH_COLON_RE = re.compile(r"^(?P<head>.+?)\s*:\s*(?P<rest>.*)$")
HEAD_WITH_PERIOD_RE = re.compile(r"^(?P<head>.+?)\s*\.\s*(?P<rest>.*)$")
HEAD_WITH_INLINE_POS_RE = re.compile(r"^(?P<head>.+?)\s+\((?P<pos>[^)]{1,16})\)\s*(?P<rest>.*)$")
POS_RE = re.compile(r"\(([^)]{1,16})\)")
ABBREV_RE = re.compile(r"^([A-ZÇĞİIÖŞÜÊÛ][A-Za-zÇĞİIÖŞÜÊÛçğıöşüâîû]{0,8}\.)")
MULTISPACE_RE = re.compile(r"\s+")

TR_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})

FIELDNAMES = [
    "entry",
    "definition",
    "gender_number",
    "entry_type",
    "parent",
    "needs_review",
    "page_number",
    "source_dictionary",
    "source_row_idx",
    "letter_section",
    "raw_headword",
    "raw_pos",
    "normalized_pos",
    "source_abbreviations",
    "source_pdf_page",
]

ENTRY_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    ("BIRCELI", "Favori (faul)"): {"raw_pos": "i"},
    ("BUMBA BİYAYİŞ", "Bombalanmak"): {"raw_pos": "f"},
    ("CEM'A KERDIŞ", "Halvet olmak"): {"raw_pos": "f"},
    ("CİKERDI", "Ayırt edilmiş, bölünmüş"): {"raw_pos": "sf"},
    ("ÇÊWRA", "Kimseden, kimseye"): {"raw_pos": "zm"},
    ("HEŞEK", "Şamata, gırgır"): {"raw_pos": "i"},
    ("HOR SİYA", "Kara bulut"): {"raw_pos": "i"},
    ("İHMALKARİ KERDIŞ", "İhmalkârlık yapmak-etmek"): {"raw_pos": "f"},
    ("KAHIN", "Kâhin"): {"raw_pos": "i"},
    ("KEW VETIŞ", "Köpük çıkarmak"): {"raw_pos": "f"},
    ("KÊ/Kİ", "Giller, oğulları (kê Eli- Ali giller, Ali oğulları)"): {"raw_pos": "ek"},
    ("MEHKUM BİYAYİŞ", "Mahkûm olmak, hüküm giymek, kaçak olmak"): {"raw_pos": "f"},
    ("MEHKUM KERDIŞ", "Mahkûm etmek, hükümlü etmek, kaçak yapmak-etmek"): {"raw_pos": "f"},
    ("ME'MELI", "Davranış, uygulama, muamele, iş"): {"raw_pos": "i"},
    ("ME'NI BİYAYİŞ", "Muayene olmak"): {"raw_pos": "f"},
    ("NEQUIR KERDI", "Gagalanmış, zedelenmiş, tırtıklanmış"): {"raw_pos": "sf"},
    ("NIMAJCA", "Namazgâh"): {"raw_pos": "i"},
    ("NIMYAYİŞ", "Gizlenmek, saklanmak"): {"raw_pos": "f"},
    ("NİWÊŞİYÊ MERG", "Ölümcül hastalık"): {"raw_pos": "i"},
    ("NİWÊŞİYÊ XIRAW", "Kötü hastalık, kanser"): {"raw_pos": "i"},
    ("NUIR", ""): {"raw_pos": "i", "definition": "Nur", "source_abbreviations": "Ar."},
    ("QUMBİZARİ", "Merdümgirizlik"): {"raw_pos": "i"},
    ("REHN", "Rehin, borç karılığı verilen güvence"): {"raw_pos": "i"},
    ("REHİNI", "Rehine"): {"raw_pos": "i"},
    ("RENCUN KERDI", "Yormuş"): {"raw_pos": "sf"},
    ("RENCUN KERDIŞ", "Yormak"): {"raw_pos": "f"},
    ("SEBAT KERDIŞ", "Sebat etmek"): {"raw_pos": "f"},
    ("SERDEW", "Muhtar"): {"raw_pos": "i"},
    ("SERGUIRI", "İş Çavuşu-başı, kalfa"): {"raw_pos": "i"},
    ("SİNAYİ", "Çok sevilen"): {"raw_pos": "sf"},
    ("TEDI MENDIŞ", "İçinde kalmak, kalmak"): {"raw_pos": "f"},
    ("TERA MENDIŞ", "Arta kalmak, fazla kalmak, arta kalmak"): {"raw_pos": "f"},
    ("TEXERNAYİŞ", "Ertelemek, ötelemek, oyalamak"): {"raw_pos": "f"},
    ("WUIQIF", "Vakıf, bilen, uzman"): {"raw_pos": "i.sf"},
    ("XERCKERDUEĞ", "Harcayan, tüketen, tüketici"): {"raw_pos": "i.sf"},
    ("XIŞNYER", "İrinin irisi, daha iri"): {"raw_pos": "sf"},
    ("XUWA VER", "İlerisine, ötesine"): {"raw_pos": "zf"},
    ("ZERÊ ZERIYİN", "İçerinin içerisi"): {"raw_pos": "i"},
}


@dataclass
class EntryStart:
    head: str
    payload: str
    separator: str
    review_tags: set[str] = field(default_factory=set)


@dataclass
class EntryBlock:
    head: str
    page_number: str
    pdf_page_number: int
    line_number: int
    letter_section: str
    separator: str
    payload_fragments: list[str] = field(default_factory=list)
    review_tags: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Genç Abdurahman Yilmaz dictionary PDF to a structured TSV.")
    parser.add_argument("--pdf-path", default=str(DEFAULT_PDF_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--summary-path", default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--start-page", type=int, default=PDF_START_PAGE)
    parser.add_argument("--end-page", type=int, default=PDF_END_PAGE)
    return parser.parse_args()


def normalize_spaces(value: str) -> str:
    return MULTISPACE_RE.sub(" ", (value or "").strip())


def clean_text(value: str) -> str:
    return normalize_spaces(unicodedata.normalize("NFC", value or "").replace("’", "'").replace("`", "'"))


def tr_lower(value: str) -> str:
    return clean_text(value).translate(TR_LOWER_MAP).lower()


def merge_with_dehyphen(previous: str, current: str) -> str:
    prev = previous.rstrip()
    curr = current.lstrip()
    if not prev:
        return curr
    if not curr:
        return prev
    if prev.endswith("-"):
        return prev + curr
    return f"{prev} {curr}"


def page_display_number(page_text: str, pdf_page_number: int) -> str:
    for line in page_text.splitlines()[:8]:
        cleaned = clean_text(line)
        if cleaned.isdigit():
            return cleaned
    return str(pdf_page_number)


def is_letter_header(line: str) -> bool:
    return clean_text(line) in LETTER_HEADERS


def looks_like_head(candidate: str) -> bool:
    text = clean_text(candidate).strip(":.")
    if not text or len(text) > 140:
        return False
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    upper_like = sum(1 for char in letters if char == char.upper())
    return (upper_like / len(letters)) >= 0.9


def parse_entry_start(line: str) -> EntryStart | None:
    cleaned = clean_text(line)
    if not cleaned:
        return None

    match = HEAD_WITH_COLON_RE.match(cleaned)
    if match and looks_like_head(match.group("head")):
        return EntryStart(
            head=clean_text(match.group("head")).strip(":."),
            payload=clean_text(match.group("rest")),
            separator=":",
            review_tags=set(),
        )

    match = HEAD_WITH_PERIOD_RE.match(cleaned)
    if match and looks_like_head(match.group("head")):
        return EntryStart(
            head=clean_text(match.group("head")).strip(":."),
            payload=clean_text(match.group("rest")),
            separator=".",
            review_tags=set(),
        )

    match = HEAD_WITH_INLINE_POS_RE.match(cleaned)
    if match and looks_like_head(match.group("head")):
        return EntryStart(
            head=clean_text(match.group("head")),
            payload=clean_text(f"({match.group('pos')}) {match.group('rest')}"),
            separator="",
            review_tags=set(),
        )

    return None


def merge_fragments(fragments: list[str]) -> str:
    merged = ""
    for fragment in fragments:
        cleaned = clean_text(fragment)
        if not cleaned:
            continue
        merged = merge_with_dehyphen(merged, cleaned)
    return merged


def pop_leading_source_abbreviations(text: str) -> tuple[list[str], str]:
    sources: list[str] = []
    working = clean_text(text)
    while working:
        match = ABBREV_RE.match(working)
        if not match:
            break
        token = match.group(1)
        if token not in SOURCE_ABBREVIATIONS:
            break
        sources.append(token)
        working = working[match.end() :].lstrip()
        if working.startswith("/"):
            working = working[1:].lstrip()
            continue
    return sources, working


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_raw_pos(raw_pos: str) -> str:
    token = tr_lower(raw_pos)
    token = re.sub(r"\s+", "", token)
    token = token.replace("ç.i.", "ç.i")
    token = token.rstrip(".")
    raw_to_normalized = {
        "i": "i",
        "ı": "i",
        "sf": "sf",
        "s": "sf",
        "f": "f",
        "zf": "zf",
        "z": "zf",
        "zm": "zm",
        "ü": "ü",
        "b": "b",
        "e": "e",
        "d": "d",
        "ç.i": "ç.i",
        "ek": "ek",
        "i.s": "i.sf",
        "i.sf": "i.sf",
        "s.zf": "sf.zf",
        "zf.s": "sf.zf",
        "f.e": "f.e",
        "b.e": "b.e",
        "s.e": "sf.e",
    }
    return raw_to_normalized.get(token, token)


def entry_type_for_pos(normalized_pos: str) -> str:
    if normalized_pos == "i":
        return "noun"
    if normalized_pos == "sf":
        return "adjective"
    if normalized_pos == "f":
        return "verb"
    if normalized_pos == "zf":
        return "adverb"
    if normalized_pos == "zm":
        return "pronoun"
    if normalized_pos == "ü":
        return "interjection"
    if normalized_pos == "b":
        return "conjunction"
    if normalized_pos == "e":
        return "adposition"
    if normalized_pos == "d":
        return "idiom"
    if normalized_pos == "ç.i":
        return "plural"
    if normalized_pos == "ek":
        return "suffix"
    if normalized_pos:
        return "multi_pos"
    return "word"


def parse_payload(payload: str) -> tuple[str, str, list[str], str, set[str]]:
    text = clean_text(payload)
    review_tags: set[str] = set()
    raw_pos = ""
    normalized_pos = ""
    sources: list[str] = []
    definition = text

    match = None
    before_sources: list[str] = []
    for candidate in POS_RE.finditer(text[:48]):
        candidate_before = clean_text(text[: candidate.start()])
        candidate_sources, candidate_remaining = pop_leading_source_abbreviations(candidate_before)
        if candidate_remaining:
            continue
        match = candidate
        before_sources = candidate_sources
        break

    if match:
        raw_pos = clean_text(match.group(1))
        normalized_pos = normalize_raw_pos(raw_pos)
        if normalized_pos not in {
            "i",
            "sf",
            "f",
            "zf",
            "zm",
            "ü",
            "b",
            "e",
            "d",
            "ç.i",
            "ek",
            "i.sf",
            "sf.zf",
            "f.e",
            "b.e",
            "sf.e",
        }:
            review_tags.add("unknown_pos")

        after = clean_text(text[match.end() :])
        after_sources, definition = pop_leading_source_abbreviations(after)
        sources = dedupe_preserve_order(before_sources + after_sources)
    else:
        sources, definition = pop_leading_source_abbreviations(text)
        review_tags.add("missing_pos")

    if not definition:
        review_tags.add("missing_definition")

    return raw_pos, normalized_pos, sources, definition, review_tags


def apply_entry_override(entry: str, definition: str, raw_pos: str, normalized_pos: str, sources: list[str]) -> tuple[str, str, list[str], str, bool]:
    override = ENTRY_OVERRIDES.get((entry, definition))
    if not override:
        return definition, raw_pos, sources, normalized_pos, False

    raw_pos = override.get("raw_pos", raw_pos)
    normalized_pos = normalize_raw_pos(raw_pos) if raw_pos else normalized_pos
    if "definition" in override:
        definition = override["definition"]
    if "source_abbreviations" in override:
        sources = [item.strip() for item in override["source_abbreviations"].split("/") if item.strip()]
    return definition, raw_pos, sources, normalized_pos, True


def blocks_from_pdf(pdf_path: Path, start_page: int, end_page: int) -> tuple[list[EntryBlock | dict[str, object]], int]:
    reader = PdfReader(str(pdf_path))
    blocks: list[EntryBlock | dict[str, object]] = []
    current: EntryBlock | None = None
    current_letter = ""
    skipped_orphan_lines = 0

    def flush_current() -> None:
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    for pdf_page_number in range(start_page, end_page + 1):
        page_text = reader.pages[pdf_page_number - 1].extract_text() or ""
        source_page_number = page_display_number(page_text, pdf_page_number)
        for line_number, raw_line in enumerate(page_text.splitlines(), start=1):
            line = clean_text(raw_line)
            if not line or line == source_page_number:
                continue
            if is_letter_header(line):
                flush_current()
                current_letter = line
                blocks.append(
                    {
                        "kind": "header",
                        "entry": line,
                        "page_number": source_page_number,
                        "source_pdf_page": str(pdf_page_number),
                        "letter_section": line,
                    }
                )
                continue

            entry_start = parse_entry_start(line)
            if entry_start:
                flush_current()
                current = EntryBlock(
                    head=entry_start.head,
                    page_number=source_page_number,
                    pdf_page_number=pdf_page_number,
                    line_number=line_number,
                    letter_section=current_letter,
                    separator=entry_start.separator,
                    payload_fragments=[entry_start.payload] if entry_start.payload else [],
                    review_tags=set(entry_start.review_tags),
                )
                continue

            if current is not None:
                current.payload_fragments.append(line)
            else:
                skipped_orphan_lines += 1

    flush_current()
    return blocks, skipped_orphan_lines


def row_from_block(block: EntryBlock | dict[str, object], source_row_idx: int) -> dict[str, str]:
    if isinstance(block, dict) and block.get("kind") == "header":
        return {
            "entry": str(block["entry"]),
            "definition": "",
            "gender_number": "",
            "entry_type": "header",
            "parent": "",
            "needs_review": "",
            "page_number": str(block["page_number"]),
            "source_dictionary": SOURCE_DICTIONARY,
            "source_row_idx": str(source_row_idx),
            "letter_section": str(block["letter_section"]),
            "raw_headword": str(block["entry"]),
            "raw_pos": "",
            "normalized_pos": "",
            "source_abbreviations": "",
            "source_pdf_page": str(block["source_pdf_page"]),
        }

    assert isinstance(block, EntryBlock)
    payload = merge_fragments(block.payload_fragments)
    raw_pos, normalized_pos, sources, definition, parse_review_tags = parse_payload(payload)
    definition, raw_pos, sources, normalized_pos, override_applied = apply_entry_override(
        clean_text(block.head), definition, raw_pos, normalized_pos, sources
    )
    review_tags = set(block.review_tags) | parse_review_tags
    if not block.letter_section:
        review_tags.add("missing_letter_section")
    if override_applied:
        review_tags.discard("missing_pos")
        review_tags.discard("missing_definition")

    return {
        "entry": clean_text(block.head),
        "definition": clean_text(definition),
        "gender_number": "",
        "entry_type": entry_type_for_pos(normalized_pos),
        "parent": "",
        "needs_review": " ".join(sorted(review_tags)),
        "page_number": str(block.page_number),
        "source_dictionary": SOURCE_DICTIONARY,
        "source_row_idx": str(source_row_idx),
        "letter_section": clean_text(block.letter_section),
        "raw_headword": clean_text(block.head),
        "raw_pos": raw_pos,
        "normalized_pos": normalized_pos,
        "source_abbreviations": " / ".join(sources),
        "source_pdf_page": str(block.pdf_page_number),
    }


def dedupe_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    removed = 0
    for row in rows:
        key = tuple(row.get(field, "") for field in FIELDNAMES if field != "source_row_idx")
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, removed


def build_summary(rows: list[dict[str, str]], skipped_orphan_lines: int, duplicates_removed: int, output_path: Path) -> dict[str, object]:
    entry_rows = [row for row in rows if row["entry_type"] != "header"]
    review_counter: Counter[str] = Counter()
    for row in entry_rows:
        for tag in row["needs_review"].split():
            review_counter[tag] += 1

    return {
        "output_path": str(output_path),
        "rows_written": len(rows),
        "headers_written": len(rows) - len(entry_rows),
        "entries_written": len(entry_rows),
        "unique_entry_strings": len({row["entry"].casefold() for row in entry_rows}),
        "entry_type_counts": dict(sorted(Counter(row["entry_type"] for row in rows).items())),
        "normalized_pos_counts": dict(sorted(Counter(row["normalized_pos"] for row in entry_rows if row["normalized_pos"]).items())),
        "review_counts": dict(sorted(review_counter.items())),
        "source_page_range": {
            "start": rows[0]["page_number"] if rows else "",
            "end": rows[-1]["page_number"] if rows else "",
        },
        "pdf_page_range": {
            "start": rows[0]["source_pdf_page"] if rows else "",
            "end": rows[-1]["source_pdf_page"] if rows else "",
        },
        "skipped_orphan_lines": skipped_orphan_lines,
        "duplicates_removed": duplicates_removed,
    }


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = Path(args.summary_path).expanduser().resolve()

    blocks, skipped_orphan_lines = blocks_from_pdf(pdf_path, args.start_page, args.end_page)
    rows = [row_from_block(block, idx) for idx, block in enumerate(blocks)]
    rows, duplicates_removed = dedupe_rows(rows)

    write_tsv(output_path, FIELDNAMES, rows)
    summary = build_summary(rows, skipped_orphan_lines, duplicates_removed, output_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
