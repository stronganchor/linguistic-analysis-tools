#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
PDF → TXT → TSV pipeline for the Harun Turgut Zazaca→Turkish dictionary.

Usage:
    py _pdf_to_tsv.py [--engine pypdf|pdftotext_raw|pdftotext_layout] [--long-def N]

Output:
    <PDF folder>\tsv_output\<PDF-stem>.tsv

Columns (TSV):
    entry, definition, gender_number, entry_type, parent, needs_review, page_number
"""

from __future__ import annotations
from pathlib import Path
import sys, argparse, re, csv, shutil, subprocess, tempfile

# ----------------------- Config -----------------------
DEFAULT_LONG_DEF = 400          # marks long_def
HUGE_ENTRY_CHARS = 80           # marks huge_entry in needs_review
HUGE_DEF_CHARS   = 800          # marks huge_def in needs_review

# ------------------ Engine detection -----------------
try:
    from pypdf import PdfReader
except Exception as e:
    sys.exit(f"pypdf is required: {e}\nInstall with: py -m pip install pypdf")

def find_pdftotext() -> str | None:
    exe = shutil.which("pdftotext")
    if exe:
        return exe
    candidates = [
        r"C:\ProgramData\chocolatey\lib\poppler\tools\poppler-25.10.0\bin\pdftotext.exe",
        r"C:\ProgramData\chocolatey\lib\poppler\tools\pdftotext.exe",
        r"C:\tools\poppler\Library\bin\pdftotext.exe",
        r"C:\tools\poppler\poppler-25.07.0\Library\bin\pdftotext.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

# -------------------- Text extraction --------------------
def open_with_pypdf(src: Path):
    try:
        reader = PdfReader(str(src))
    except Exception as e:
        sys.exit(f"Error opening PDF with pypdf: {e}")
    if getattr(reader, "is_encrypted", False):
        try:
            if reader.decrypt("") == 0:
                sys.exit("Error: PDF is encrypted (pypdf) and needs a password.")
        except Exception:
            sys.exit("Error: PDF is encrypted (pypdf) and could not be opened.")
    return reader

def extract_text_pypdf(reader, page_index: int) -> str:
    try:
        page = reader.pages[page_index]
        text = page.extract_text() or ""
    except Exception as e:
        text = f"[Extraction error on page {page_index+1} with pypdf: {e}]\n"
    if not text.endswith("\n"):
        text += "\n"
    return text

def extract_by_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    reader = open_with_pypdf(pdf_path)
    out: list[tuple[int, str]] = []
    for i in range(len(reader.pages)):
        text = extract_text_pypdf(reader, i)
        out.append((i + 1, text))
    return out

def run_pdftotext(pdftotext_exe: str, src: Path, first_page: int, last_page: int, layout: bool) -> str:
    with tempfile.TemporaryDirectory() as td:
        out_txt = Path(td) / "chunk.txt"
        cmd = [
            pdftotext_exe, "-enc", "UTF-8", "-nopgbrk",
            "-f", str(first_page), "-l", str(last_page)
        ]
        if layout:
            cmd.append("-layout")
        cmd += [str(src), str(out_txt)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return out_txt.read_text(encoding="utf-8", errors="replace")
        except subprocess.CalledProcessError as e:
            return f"[pdftotext error: {e.stderr.decode(errors='replace')}]"
        except Exception as e:
            return f"[pdftotext error: {e}]"

def extract_by_pdftotext(pdf_path: Path, layout: bool) -> list[tuple[int, str]]:
    exe = find_pdftotext()
    if not exe:
        print("pdftotext not found; falling back to pypdf.")
        return extract_by_pypdf(pdf_path)
    reader = open_with_pypdf(pdf_path)
    total = len(reader.pages)
    out: list[tuple[int, str]] = []
    for p in range(1, total + 1):
        out.append((p, run_pdftotext(exe, pdf_path, p, p, layout=layout)))
    return out

# ---------------------- Parsing helpers ----------------------
LetterHeader = re.compile(r'^\s*([A-ZÇĞİÖŞÜ])\s*$')
HeadLine     = re.compile(r'^\s*([^\s].*?):\s*(.*)$')
ExampleLine  = re.compile(r'^\s*([^\s].{0,60}?):\s+(.*)$')

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def sanitize_field(s: str) -> str:
    """Make a field TSV-safe: remove tabs/newlines; collapse pipes/spaces; trim."""
    if s is None:
        return ''
    s = s.replace('\t', ' ')              # kill real cell splitters
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = s.replace('\n', ' | ')
    s = re.sub(r'\s*\|\s*', ' | ', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def merge_with_dehyphen(prev: str, nxt: str) -> str:
    prev_strip = prev.rstrip()
    nxt_strip = nxt.lstrip()
    if prev_strip.endswith('-') and nxt_strip:
        return prev_strip[:-1] + nxt_strip
    return prev_strip + ' ' + nxt_strip

def collect_letter_spans_with_pages(lines_with_pages: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    headers = []
    for i, (_, ln) in enumerate(lines_with_pages):
        m = LetterHeader.match(ln)
        if m:
            headers.append((i, m.group(1)))
    spans = []
    for idx, (i, letter) in enumerate(headers):
        j = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines_with_pages)
        spans.append((i, j, letter))
    return spans

def collect_prefix_bases(section_lines: list[str]) -> list[str]:
    bases = set()
    for ln in section_lines:
        m = HeadLine.match(ln)
        if not m:
            continue
        head = normalize_spaces(m.group(1))
        if head.endswith('-'):
            bases.add(head)
    return sorted(bases, key=len, reverse=True)

def is_true_suffix_head(head: str) -> bool:
    return head.startswith('-') and (' ' not in head)

# Turkish/Zazaki order (for suffix folding only)
_order = ['a','b','c','ç','d','e','f','g','ğ','h','ı','i','j','k','l','m','n','o','ö','p','r','s','ş','t','u','ü','v','y','z']
idxmap = {ch:i for i,ch in enumerate(_order)}
fallback_map = {'q':'k~','w':'v~','x':'ks'}

def turkic_key(s: str) -> tuple:
    s = s.lower().strip()
    s = re.sub(r'\((?:m|n|z)\)', '', s)
    s = re.sub(r'\s+', ' ', s)
    key = []
    for ch in s:
        if ch in idxmap:
            key.append((0, idxmap[ch]))
        elif ch in (' ', '-', '/'):
            key.append((1, ord(ch)))
        elif ch in fallback_map:
            key.append((2, fallback_map[ch]))
        elif 'a' <= ch <= 'z':
            key.append((3, ord(ch)))
        else:
            continue
    return tuple(key) if key else ((9, 0),)

def first_real_letter_for_section(head: str) -> str:
    s = head.strip().lower()
    if s.startswith('-'):
        s = s[1:].lstrip()
    for ch in s:
        if ch.isalpha():
            return ch
    return ''

# Section normalization (diacritic-insensitive for section membership)
_SECTION_BASE = str.maketrans({
    'ç': 'c', 'ğ': 'g', 'ö': 'o', 'ş': 's', 'ü': 'u', 'ı': 'i', 'ï': 'i', 'î': 'i',
    'â': 'a', 'á': 'a', 'à': 'a', 'ã': 'a', 'å': 'a', 'é': 'e', 'è': 'e', 'ê': 'e',
    'í': 'i', 'ì': 'i', 'ó': 'o', 'ò': 'o', 'ô': 'o', 'ú': 'u', 'ù': 'u',
})
def normalize_for_section(ch: str) -> str:
    return ch.lower().translate(_SECTION_BASE)

def section_match(head_letter: str, section_letter: str) -> bool:
    if not head_letter or not section_letter:
        return True
    return normalize_for_section(head_letter) == normalize_for_section(section_letter)

# OCR corrections for H & M sections
_OCR_H_START = re.compile(r'^((?:[a-zçğıöşü]+-)*)(l[ıiI1])', re.IGNORECASE)
_OCR_M_START_111 = re.compile(r'^((?:[a-zçğıöşü]+-)*)(1{2,})', re.IGNORECASE)
_OCR_M_START_NI  = re.compile(r'^((?:[a-zçğıöşü]+-)*)(?:n[ıiI1]|[ıiI1]n)', re.IGNORECASE)

def fix_ocr_head_for_section(head: str, section_letter: str, is_affix: bool) -> str:
    if is_affix:
        return head
    sec = section_letter.lower()
    if sec == 'h':
        return _OCR_H_START.sub(lambda m: f"{m.group(1)}h", head, count=1)
    if sec == 'm':
        new_head = _OCR_M_START_111.sub(lambda m: f"{m.group(1)}m", head, count=1)
        if new_head != head:
            return new_head
        new_head = _OCR_M_START_NI.sub(lambda m: f"{m.group(1)}m", head, count=1)
        return new_head
    return head

def base_for_order(head: str) -> str:
    return re.split(r'[\s-]', head.lower(), maxsplit=1)[0]

def guess_entry_type(head: str, gender: str) -> str:
    if is_true_suffix_head(head):
        return 'suffix'
    if head.endswith('-'):
        return 'prefix'
    if gender:
        return 'noun'
    if re.search(r'(ayış|eyiş|ıyayış|iyayış|nayış|tiş|tış|diş|dış|kerdış|kotış|berdış|gêrayiş|guzıyayış|guznayış)$', head.lower()):
        return 'verb'
    return 'word'

# ---------------------- Core parsing ----------------------
def parse_to_rows(lines_with_pages: list[tuple[int, str]],
                  include_parent: bool = True,
                  variant_delim: str = " / ",
                  long_def_threshold: int = DEFAULT_LONG_DEF) -> list[tuple[str,str,str,str,str,str,int|str]]:
    spans = collect_letter_spans_with_pages(lines_with_pages)
    results: list[tuple[str,str,str,str,str,str,int|str]] = []

    for start_idx, end_idx, letter in spans:
        section_letter = letter.lower()
        results.append((letter, '', '', 'header', '', '', ''))  # header row

        section_lines = [lines_with_pages[k][1] for k in range(start_idx+1, end_idx)]
        prefix_bases = collect_prefix_bases(section_lines)

        i = start_idx + 1
        last_order_key = None  # suffix folding only

        while i < end_idx:
            page_no, line = lines_with_pages[i]
            m = HeadLine.match(line)
            if not m:
                i += 1
                continue

            head_raw, defn_raw = m.group(1), m.group(2)
            head = normalize_spaces(head_raw)
            defn = normalize_spaces(defn_raw)

            # gender and variants
            gender = ''
            g = re.search(r'\((m|n|z)\)', head, re.IGNORECASE)
            if g:
                gender = g.group(1).lower()
                head = normalize_spaces(re.sub(r'\s*\((m|n|z)\)\s*', ' ', head, flags=re.IGNORECASE))
            head = re.sub(r'\s*/\s*', variant_delim, head)

            # prelim type and OCR fix (section-aware)
            prelim_type = guess_entry_type(head, gender)
            prelim_is_affix = (prelim_type in ('suffix', 'prefix'))
            head = fix_ocr_head_for_section(head, section_letter, prelim_is_affix)

            # finalize type
            entry_type = guess_entry_type(head, gender)
            is_suffix = (entry_type == 'suffix')
            is_prefix = (entry_type == 'prefix')

            # suffix folding heuristic (only for suffixes out of relative order)
            order_head_for_compare = head[1:] if head.startswith('-') else head
            curr_order_key = turkic_key(order_head_for_compare)

            if is_suffix and last_order_key is not None and curr_order_key < last_order_key:
                # swallow into previous entry
                j = i + 1
                cont: list[str] = []
                while j < end_idx:
                    _, ln = lines_with_pages[j]
                    if LetterHeader.match(ln): break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        if is_true_suffix_head(h2) or h2.endswith('-'):
                            break
                        cont.append(m2.group(0).strip()); j += 1; continue
                    cont.append(ln); j += 1

                merged = defn
                for ln in cont:
                    exm = ExampleLine.match(ln)
                    merged = (merged + ' | ' + (exm.group(1).strip() + ': ' + exm.group(2).strip())) if exm else merge_with_dehyphen(merged, ln.strip())
                merged = sanitize_field(normalize_spaces(merged))

                # append to previous non-header row
                k = len(results) - 1
                while k >= 0 and results[k][3] == 'header':
                    k -= 1
                if k >= 0:
                    prev = list(results[k])
                    prev[1] = sanitize_field((prev[1] + ' | ' + f'{head}: {merged}').strip())
                    results[k] = tuple(prev)
                i = j
                continue

            # normal continuation gather
            j = i + 1
            cont: list[str] = []
            if is_suffix:
                while j < end_idx:
                    _, ln = lines_with_pages[j]
                    if LetterHeader.match(ln): break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        if is_true_suffix_head(h2) or h2.endswith('-'):
                            break
                        cont.append(m2.group(0).strip()); j += 1; continue
                    cont.append(ln); j += 1
            else:
                while j < end_idx:
                    _, ln = lines_with_pages[j]
                    if LetterHeader.match(ln): break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        # sub-entry/example like "-dı bıyayış:" stays with current head
                        if h2.startswith('-') and (' ' in h2):
                            cont.append(m2.group(0).strip()); j += 1; continue
                        break
                    cont.append(ln); j += 1

            merged = defn
            for ln in cont:
                exm = ExampleLine.match(ln)
                merged = (merged + ' | ' + (exm.group(1).strip() + ': ' + exm.group(2).strip())) if exm else merge_with_dehyphen(merged, ln.strip())
            merged = sanitize_field(normalize_spaces(merged))
            merged = re.sub(r'\s*\d\)\s*', ' | ', merged)
            merged = sanitize_field(normalize_spaces(re.sub(r'\s*\|\s*', ' | ', merged)))

            # parent via prefix bases
            parent = ''
            if include_parent and not is_prefix:
                for base in prefix_bases:
                    if head.startswith(base):
                        parent = base; break

            # needs_review tags
            review_tags: list[str] = []
            first_letter = first_real_letter_for_section(order_head_for_compare)
            if first_letter and not section_match(first_letter, section_letter):
                review_tags.append('out_of_order')
            if re.search(r'\d', head):
                review_tags.append('num_in_entry')
            if re.search(r'\d', merged):
                review_tags.append('num_in_def')
            if len(merged) > DEFAULT_LONG_DEF:
                review_tags.append('long_def')
            if len(head) > HUGE_ENTRY_CHARS:
                review_tags.append('huge_entry')
            if len(merged) > HUGE_DEF_CHARS:
                review_tags.append('huge_def')
            needs_review = ' '.join(sorted(set(review_tags)))

            # sanitize all fields to prevent column slide in TSV
            row = (
                sanitize_field(head),
                sanitize_field(merged),
                sanitize_field(gender),
                sanitize_field(entry_type),
                sanitize_field(parent),
                sanitize_field(needs_review),
                page_no  # keep as int; writer will stringify
            )
            results.append(row)

            last_order_key = curr_order_key
            i = j

    return results

# ---------------------- I/O & CLI ----------------------
def ask(prompt: str) -> str:
    return input(prompt).strip()

def prompt_for_pdf() -> Path:
    while True:
        p_str = ask("Path to PDF file: ").strip().strip('"')
        if not p_str:
            print("Please paste a path to a .pdf.")
            continue
        src = Path(p_str).expanduser().resolve()
        if src.exists() and src.is_file() and src.suffix.lower() == ".pdf":
            return src
        print(f"Not found or not a .pdf: {src}")

def write_tsv(rows, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_NONE, escapechar='\\')
        # header
        w.writerow(['entry', 'definition', 'gender_number', 'entry_type', 'parent', 'needs_review', 'page_number'])
        for r in rows:
            # harden row shape and sanitize (defense in depth)
            entry, definition, gender, entry_type, parent, needs_review, page_number = r
            entry = sanitize_field(entry)
            definition = sanitize_field(definition)
            gender = sanitize_field(gender)
            entry_type = sanitize_field(entry_type)
            parent = sanitize_field(parent)
            needs_review = sanitize_field(needs_review)
            w.writerow([entry, definition, gender, entry_type, parent, needs_review, page_number])

def main():
    ap = argparse.ArgumentParser(description="PDF → TXT → TSV pipeline")
    ap.add_argument(
        "--engine",
        choices=["pypdf", "pdftotext_raw", "pdftotext_layout"],
        default="pypdf",
        help="Extraction engine (default: pypdf)",
    )
    ap.add_argument(
        "--long-def",
        type=int,
        default=DEFAULT_LONG_DEF,
        help="Threshold for long_def review tag",
    )
    args = ap.parse_args()

    pdf_path = prompt_for_pdf()
    out_dir = pdf_path.parent / "tsv_output"
    out_file = out_dir / f"{pdf_path.stem}.tsv"

    # 1) Extract text page-by-page
    if args.engine == "pypdf":
        pages = extract_by_pypdf(pdf_path)
        engines_used = ["pypdf"]
    elif args.engine == "pdftotext_raw":
        pages = extract_by_pdftotext(pdf_path, layout=False)
        engines_used = ["pdftotext_raw"]
    else:
        pages = extract_by_pdftotext(pdf_path, layout=True)
        engines_used = ["pdftotext_layout"]

    print(f"Extracted text from {len(pages)} page(s) using {', '.join(engines_used)}.")

    # 2) Flatten to (page, line)
    lines_with_pages: list[tuple[int, str]] = []
    for page_no, page_text in pages:
        for ln in (page_text or "").splitlines():
            lines_with_pages.append((page_no, ln))

    # 3) Parse → rows
    rows = parse_to_rows(
        lines_with_pages,
        include_parent=True,
        variant_delim=" / ",
        long_def_threshold=args.long_def,
    )

    # 4) Write TSV
    write_tsv(rows, out_file)

    flagged = sum(1 for r in rows if r[3] != "header" and r[5])
    print(f"Wrote {len(rows)} rows to: {out_file}")
    print(f"Flagged for review: {flagged} entries (see 'needs_review').")

if __name__ == "__main__":
    main()
