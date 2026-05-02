#!/usr/bin/env python3
# Extract PDF text using:
#   1) pypdf (always)
#   2) Poppler pdftotext (raw and layout modes) if available
#
# Output: <PDF folder>\<PDF stem>_txt\<stem>_p###-###__<engine>.txt
# Requires: py -m pip install pypdf
#
# Optionally uses pdftotext.exe from:
#   - PATH
#   - C:\ProgramData\chocolatey\lib\poppler\tools\pdftotext.exe
#
from pathlib import Path
import sys, shutil, subprocess, tempfile

CHUNK_SIZE = 500  # pages per .txt file

try:
    from pypdf import PdfReader
except Exception as e:
    sys.exit(f"pypdf is required: {e}\nInstall with: py -m pip install pypdf")


def ask(prompt: str) -> str:
    return input(prompt).strip()


def clean_path(p: str) -> Path:
    p = p.strip().strip('"').strip("'")
    return Path(p).expanduser().resolve()


def prompt_for_pdf() -> Path:
    while True:
        p_str = ask("Path to PDF file: ")
        if not p_str:
            print("Please paste a path to a .pdf.")
            continue
        src = clean_path(p_str)
        if src.exists() and src.is_file() and src.suffix.lower() == ".pdf":
            return src
        print(f"Not found or not a .pdf: {src}")


# ---------- pypdf ----------
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


# ---------- Poppler detection ----------
def find_pdftotext() -> str | None:
    exe = shutil.which("pdftotext")
    if exe:
        return exe

    default_path = Path(r"C:\tools\poppler\poppler-25.07.0\Library\bin\pdftotext.exe")
    if default_path.exists():
        return str(default_path)

    return None


def run_pdftotext(pdftotext_exe: str, src: Path, first_page: int, last_page: int, layout: bool) -> str:
    with tempfile.TemporaryDirectory() as td:
        out_txt = Path(td) / "chunk.txt"
        cmd = [
            pdftotext_exe,
            "-enc", "UTF-8",
            "-nopgbrk",
            "-f", str(first_page),
            "-l", str(last_page),
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


def main():
    src = prompt_for_pdf()
    reader = open_with_pypdf(src)
    total = len(reader.pages)
    if total == 0:
        sys.exit("Error: PDF has 0 pages.")

    pdftotext_exe = find_pdftotext()
    out_dir = src.parent / f"{src.stem}_txt"
    out_dir.mkdir(exist_ok=True)
    pad = max(3, len(str(total)))

    engines = ["pypdf"]
    if pdftotext_exe:
        engines += ["pdftotext_raw", "pdftotext_layout"]

    print(
        f"Transcribing {total} page(s) in {CHUNK_SIZE}-page chunks → {out_dir}\n"
        f"Engines: {', '.join(engines)}"
    )

    for start_idx in range(0, total, CHUNK_SIZE):
        end_idx = min(start_idx + CHUNK_SIZE, total)
        label_start = str(start_idx + 1).zfill(pad)
        label_end = str(end_idx).zfill(pad)

        # pypdf
        out_pypdf = out_dir / f"{src.stem}_p{label_start}-{label_end}__pypdf.txt"
        with out_pypdf.open("w", encoding="utf-8", newline="\n") as f:
            for i in range(start_idx, end_idx):
                f.write(extract_text_pypdf(reader, i))
        print(f"✓ Wrote {out_pypdf.name}")

        if pdftotext_exe:
            # RAW
            out_raw = out_dir / f"{src.stem}_p{label_start}-{label_end}__pdftotext_raw.txt"
            txt_raw = run_pdftotext(pdftotext_exe, src, start_idx + 1, end_idx, layout=False)
            out_raw.write_text(txt_raw, encoding="utf-8", newline="\n")
            print(f"✓ Wrote {out_raw.name}")

            # LAYOUT
            out_lyt = out_dir / f"{src.stem}_p{label_start}-{label_end}__pdftotext_layout.txt"
            txt_lyt = run_pdftotext(pdftotext_exe, src, start_idx + 1, end_idx, layout=True)
            out_lyt.write_text(txt_lyt, encoding="utf-8", newline="\n")
            print(f"✓ Wrote {out_lyt.name}")
        else:
            print("• Skipped pdftotext (not found on PATH or in standard location)")

    print("Done.")


if __name__ == "__main__":
    main()
