#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Harun Turgut Zazaca→Turkish dictionary (.txt) → TSV converter (interactive)

Enhancements:
- True suffix heads (start with '-' and have NO spaces) become their own entries
  ONLY if they are in alphabetical order relative to the previous entry in the
  same letter section. Otherwise they are appended as sub-entries/examples to
  the previous head.
- Lines that start with '-' AND contain a space (e.g., "-dı bıyayış: ...") are
  always treated as sub-entries/examples (kept inside the current head).
- Adds a "needs_review" column with tags: out_of_order, num_in_entry, num_in_def, long_def.

Columns:
    entry, definition, gender_number, entry_type, parent, needs_review
"""

from __future__ import annotations
import re
import csv
import sys
from pathlib import Path
from typing import List, Tuple

# ---- Tuning ----
LONG_DEF_CHARS = 400  # definitions longer than this get "long_def"

LetterHeader = re.compile(r'^\s*([A-ZÇĞİÖŞÜ])\s*$')
HeadLine     = re.compile(r'^\s*([^\s].*?):\s*(.*)$')
ExampleLine  = re.compile(r'^\s*([^\s].{0,60}?):\s+(.*)$')

def normalize_spaces(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()

def merge_with_dehyphen(prev: str, nxt: str) -> str:
    prev_strip = prev.rstrip()
    nxt_strip = nxt.lstrip()
    if prev_strip.endswith('-') and nxt_strip:
        return prev_strip[:-1] + nxt_strip
    return prev_strip + ' ' + nxt_strip

def collect_letter_spans(lines: List[str]) -> List[Tuple[int, int, str]]:
    headers = [(i, m.group(1)) for i, ln in enumerate(lines) if (m := LetterHeader.match(ln))]
    spans = []
    for idx, (i, letter) in enumerate(headers):
        j = headers[idx + 1][0] if idx + 1 < len(headers) else len(lines)
        spans.append((i, j, letter))
    return spans

def collect_prefix_bases(section_lines: List[str]) -> List[str]:
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
    """Real suffix head: starts with '-' and contains NO spaces."""
    return head.startswith('-') and (' ' not in head)

# --- Turkish-like alphabetical key for order checks ---
# Order: A B C Ç D E F G Ğ H I İ J K L M N O Ö P R S Ş T U Ü V Y Z
_order = ['a','b','c','ç','d','e','f','g','ğ','h','ı','i','j','k','l','m','n','o','ö','p','r','s','ş','t','u','ü','v','y','z']
idx = {ch:i for i,ch in enumerate(_order)}
# Map some ASCII fallbacks to keep comparisons stable if OCR introduced them
fallback = {'q':'k~','w':'v~','x':'ks'}  # crude but deterministic

def turkic_key(s: str) -> tuple:
    """
    Build a sortable key approximating Turkish/Zazaki dictionary order.
    - Lowercases
    - Strips non-letters except internal spaces/hyphens which affect order mildly
    """
    s = s.lower().strip()
    # remove gender markers etc. if any slipped in
    s = re.sub(r'\((?:m|n|z)\)', '', s)
    # collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    key = []
    for ch in s:
        if ch in idx:
            key.append((0, idx[ch]))
        elif ch in (' ', '-', '/'):
            key.append((1, ord(ch)))  # secondary weight
        elif ch in fallback:
            key.append((2, fallback[ch]))
        elif 'a' <= ch <= 'z':
            key.append((3, ord(ch)))  # unknown latin -> coarse order
        else:
            # ignore other punctuation/diacritics for ordering purposes
            continue
    return tuple(key) if key else ((9, 0),)

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

def parse(text: str,
          include_parent: bool = True,
          variant_delim: str = " / ") -> List[Tuple[str,str,str,str,str,str]]:

    lines = text.splitlines()
    spans = collect_letter_spans(lines)
    results: List[Tuple[str,str,str,str,str,str]] = []

    for start_idx, end_idx, letter in spans:
        # header row for the letter
        results.append((letter, '', '', 'header', '', ''))

        section_lines = lines[start_idx+1:end_idx]
        prefix_bases = collect_prefix_bases(section_lines)

        i = start_idx + 1
        last_order_key = None  # track order within the section for out-of-order and suffix decision

        while i < end_idx:
            line = lines[i]
            m = HeadLine.match(line)
            if not m:
                i += 1
                continue

            head_raw, defn_raw = m.group(1), m.group(2)
            head = normalize_spaces(head_raw)
            defn = normalize_spaces(defn_raw)

            # Extract and remove gender markers like "(n)"
            gender = ''
            g = re.search(r'\((m|n|z)\)', head, re.IGNORECASE)
            if g:
                gender = g.group(1).lower()
                head = normalize_spaces(re.sub(r'\s*\((m|n|z)\)\s*', ' ', head, flags=re.IGNORECASE))

            # normalize variant slashes in the head
            head = re.sub(r'\s*/\s*', variant_delim, head)

            # Determine intended type (preliminary)
            entry_type = guess_entry_type(head, gender)
            is_suffix = (entry_type == 'suffix')
            is_prefix = (entry_type == 'prefix')

            # Compute an "order key" for comparisons (drop leading '-' for suffix heads)
            order_head_for_compare = head[1:] if head.startswith('-') else head
            curr_order_key = turkic_key(order_head_for_compare)

            # Decide if a true suffix should be folded into previous head (out-of-order)
            if is_suffix and last_order_key is not None and curr_order_key < last_order_key:
                # Treat this whole thing as a sub-entry/example of the previous REAL entry.
                # Consume its continuation like a suffix would, then append to previous row's definition.
                j = i + 1
                cont: List[str] = []
                while j < end_idx:
                    ln = lines[j]
                    if LetterHeader.match(ln):
                        break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        # stop on next affix head (true suffix or prefix)
                        if is_true_suffix_head(h2) or h2.endswith('-'):
                            break
                        cont.append(m2.group(0).strip())
                        j += 1
                        continue
                    cont.append(ln)
                    j += 1

                merged = defn
                for ln in cont:
                    exm = ExampleLine.match(ln)
                    if exm:
                        merged += ' | ' + (exm.group(1).strip() + ': ' + exm.group(2).strip())
                    else:
                        merged = merge_with_dehyphen(merged, ln.strip())
                merged = normalize_spaces(merged)

                # Append to previous entry's definition
                # Find last non-header row
                k = len(results) - 1
                while k >= 0 and results[k][3] == 'header':
                    k -= 1
                if k >= 0:
                    prev_entry, prev_def, prev_gender, prev_type, prev_parent, prev_review = results[k]
                    prev_def = (prev_def + ' | ' + f'{head}: {merged}').strip()
                    results[k] = (prev_entry, prev_def, prev_gender, prev_type, prev_parent, prev_review)
                i = j
                # do NOT update last_order_key (we didn't add a new entry)
                continue

            # Otherwise, proceed to collect continuation for a normal entry/prefix/suffix
            j = i + 1
            cont: List[str] = []
            if is_suffix:
                while j < end_idx:
                    ln = lines[j]
                    if LetterHeader.match(ln):
                        break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        if is_true_suffix_head(h2) or h2.endswith('-'):
                            break
                        cont.append(m2.group(0).strip())
                        j += 1
                        continue
                    cont.append(ln)
                    j += 1
            else:
                while j < end_idx:
                    ln = lines[j]
                    if LetterHeader.match(ln):
                        break
                    m2 = HeadLine.match(ln)
                    if m2:
                        h2 = normalize_spaces(m2.group(1))
                        # sub-entry/example like "-dı bıyayış:" → keep with current head
                        if h2.startswith('-') and (' ' in h2):
                            cont.append(m2.group(0).strip())
                            j += 1
                            continue
                        # otherwise it's the next head
                        break
                    cont.append(ln)
                    j += 1

            # Build definition string
            merged = defn
            for ln in cont:
                exm = ExampleLine.match(ln)
                if exm:
                    merged += ' | ' + (exm.group(1).strip() + ': ' + exm.group(2).strip())
                else:
                    merged = merge_with_dehyphen(merged, ln.strip())
            merged = normalize_spaces(merged)
            merged = re.sub(r'\s*\d\)\s*', ' | ', merged)
            merged = merged.replace(';', ' | ')
            merged = re.sub(r'\s*\|\s*', ' | ', merged)
            merged = normalize_spaces(merged)

            # parent detection (prefix bases); do not set parent for the base itself
            parent = ''
            if include_parent and not is_prefix:
                for base in prefix_bases:
                    if head.startswith(base):
                        parent = base
                        break

            # --- needs_review tagging ---
            review_tags: List[str] = []
            if last_order_key is not None and curr_order_key < last_order_key:
                review_tags.append('out_of_order')
            if re.search(r'\d', head):
                review_tags.append('num_in_entry')
            if re.search(r'\d', merged):
                review_tags.append('num_in_def')
            if len(merged) > LONG_DEF_CHARS:
                review_tags.append('long_def')

            results.append((head, merged, gender, entry_type, parent, ' '.join(sorted(set(review_tags)))))
            # update last_order_key only after adding a real entry (not headers/examples)
            last_order_key = curr_order_key
            i = j

    return results

def write_tsv(rows: List[Tuple[str,str,str,str,str,str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['entry', 'definition', 'gender_number', 'entry_type', 'parent', 'needs_review'])
        for row in rows:
            w.writerow(row)

def main():
    # Ask for file path like your other utilities
    txt_path_str = input("Path to dictionary TXT file: ").strip().strip('"')
    if not txt_path_str:
        sys.exit("No input path provided.")
    txt_path = Path(txt_path_str).expanduser().resolve()
    if not txt_path.exists() or txt_path.suffix.lower() != '.txt':
        sys.exit(f"Error: file not found or not a .txt file: {txt_path}")

    text = txt_path.read_text(encoding='utf-8', errors='replace')
    rows = parse(text, include_parent=True, variant_delim=" / ")

    out_dir  = txt_path.parent / "tsv_output"
    out_file = out_dir / f"{txt_path.stem}.tsv"
    write_tsv(rows, out_file)

    # Simple summary
    flagged = sum(1 for r in rows if r[3] != 'header' and r[5])
    print(f"Wrote {len(rows)} rows to: {out_file}")
    print(f"Flagged for review: {flagged} entries (see 'needs_review' column).")

if __name__ == '__main__':
    main()
