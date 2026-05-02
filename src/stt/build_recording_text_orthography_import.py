import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


STRESS_MARK_PATTERN = re.compile(r"[ˈˌ]+")
COMBINING_REMOVE_PATTERN = re.compile(r"[\u02B0\u02D0\u0303\u031C\u031D\u031E\u031F\u0320\u0324\u0325\u0329\u032A\u032C\u032F\u0330\u0331\u035C\u0361\u0306\u0308]")
PUNCT_NORMALIZE_PATTERN = re.compile(r"\s+")
SHORT_VOWEL_PATTERN = re.compile(r"[aæeɛɨiɪouʊəɔ]\u0306")
EPENTHETIC_VOWEL_PATTERN = re.compile(r"[ᵊ]")
SKIP_LOOKAHEAD_CHARS = set("ˈˌʰʷʲː")
PREVELAR_FOLLOWERS = {"k", "g", "q", "ɢ", "c", "ɟ", "x", "χ"}
ALLOWED_ORTHOGRAPHY_OUTPUT_PATTERN = re.compile(r"^[A-Za-zÂÇÊÛâçêûüĞğİıŞş'?.!,:;() -]*$")
FINAL_I_NEAR_DOTLESS_I_WHITELIST = {
    "mı",
    "owı",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_metadata = (
        script_dir
        / "extracted_bundles_20260408_134225"
        / "ll-tools-stt-training-genc-palu-recording_ipa-20260408-134225"
        / "metadata.csv"
    )
    default_output_dir = (
        script_dir.parent / "zazacaogren" / "orthography-text-fill-april-2026"
    )
    parser = argparse.ArgumentParser(
        description=(
            "Build an LL Tools metadata import that fills or corrects recording_text "
            "from recording_ipa using the current Central Z orthography proposal."
        )
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(default_metadata),
        help="Source LL Tools metadata.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Folder for import, audit, and summary output.",
    )
    parser.add_argument(
        "--output-prefix",
        default="genc-palu-recording-text-orthography-import",
        help="Prefix for generated files.",
    )
    return parser.parse_args()


def normalize_space(value: str) -> str:
    return PUNCT_NORMALIZE_PATTERN.sub(" ", (value or "").strip())


def sentence_case_z(value: str) -> str:
    chars = list(value)
    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.upper()
            return "".join(chars)
    return value


def strip_terminal_punctuation(value: str) -> str:
    return re.sub(r"[.?!]+$", "", value or "").strip()


def is_trusted_word_text(value: str) -> bool:
    text = normalize_space(value)
    for ch in text:
        if ch.isalpha():
            return ch.isupper()
    return False


def next_base_segment(token: str, start_index: int) -> str:
    for ch in token[start_index:]:
        if unicodedata.combining(ch):
            continue
        if ch in SKIP_LOOKAHEAD_CHARS:
            continue
        return ch
    return ""


def collect_review_flags(ipa: str) -> list[str]:
    flags: list[str] = []
    normalized = normalize_space(unicodedata.normalize("NFC", ipa or ""))
    tokens = [token for token in normalized.split(" ") if token]

    if any(token.startswith("ʔ") for token in tokens):
        flags.append("initial_ʔ_manual_review")
    if "ʊ" in normalized:
        flags.append("contains_ʊ_manual_review")
    if SHORT_VOWEL_PATTERN.search(unicodedata.normalize("NFD", normalized)):
        flags.append("contains_short_vowel_manual_review")
    if EPENTHETIC_VOWEL_PATTERN.search(normalized):
        flags.append("contains_epenthetic_vowel_manual_review")

    for token in tokens:
        for idx, ch in enumerate(token):
            if ch != "ŋ":
                continue
            follower = next_base_segment(token, idx + 1)
            if follower not in PREVELAR_FOLLOWERS:
                flags.append("non_prevelar_ŋ_manual_review")
                break
        if "non_prevelar_ŋ_manual_review" in flags:
            break
    return flags


def collect_output_review_flags(text: str) -> list[str]:
    flags: list[str] = []
    normalized = unicodedata.normalize("NFC", text or "")
    if not ALLOWED_ORTHOGRAPHY_OUTPUT_PATTERN.match(normalized):
        flags.append("contains_residual_ipa_output_manual_review")
    return flags


def clean_ipa_for_orthography(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = STRESS_MARK_PATTERN.sub("", text)
    text = text.replace("’", "'").replace("‘", "'")
    # Preserve orthographic dorsal+labialized contrasts explicitly.
    text = text.replace("qʷʰ", "QW_SEG")
    text = text.replace("qʷ", "QW_SEG")
    text = text.replace("ɢʷ", "GW_SEG")
    text = text.replace("ɟʷ", "GW_SEG")
    text = text.replace("gʷ", "GW_SEG")
    text = text.replace("c͡çʷ", "KW_SEG")
    text = text.replace("cʷʰ", "KW_SEG")
    text = text.replace("cʷ", "KW_SEG")
    text = text.replace("kʷʰ", "KW_SEG")
    text = text.replace("kʷ", "KW_SEG")
    text = text.replace("t̪͡ʃ", "T_AFFR")
    text = text.replace("d̪͡ʒ", "D_AFFR")
    text = text.replace("t͡ʃ", "T_AFFR")
    text = text.replace("d͡ʒ", "D_AFFR")
    text = text.replace("c͡ç", "K_AFFR")
    text = text.replace("t̪͡ʙ̥̆", "TP_SEG")
    text = text.replace("t̪͡ʙ̥̆", "TP_SEG")
    text = text.replace("t̪͡ʙ̥", "TP_SEG")
    text = text.replace("t̪͡ʙ̥", "TP_SEG")
    text = text.replace("t̪͡p", "TP_SEG")
    text = text.replace("t̪͡p", "TP_SEG")
    text = text.replace("d̪͡ʙ̆", "DB_SEG")
    text = text.replace("d̪͡ʙ", "DB_SEG")
    text = text.replace("d̪͡b", "DB_SEG")
    text = text.replace("r̥", "RR_SEG")
    text = text.replace("ɾ", "R_SEG")
    text = text.replace("ɹ", "R_SEG")
    text = text.replace("r", "RR_SEG")
    text = text.replace("c", "K_STOP_SEG")
    text = text.replace("ɨ̯j", "Y_GLIDE")
    text = text.replace("ɨj", "Y_GLIDE")
    text = text.replace("ij", "Y_GLIDE")
    text = text.replace("i̯j", "Y_GLIDE")
    text = text.replace("j", "J_GLIDE")
    return text


def orthography_word_from_ipa(word: str) -> str:
    text = clean_ipa_for_orthography(word)
    text = unicodedata.normalize("NFD", text)
    # Nasalized vowels are not preserved in practical orthography.
    text = text.replace(f"a\u0303", "A_NAS_SEG")
    text = text.replace(f"æ\u0303", "AE_NAS_SEG")
    text = text.replace(f"e\u0303", "E_NAS_SEG")
    text = text.replace(f"ɛ\u0303", "EH_NAS_SEG")
    text = text.replace(f"i\u0303", "I_NAS_SEG")
    text = text.replace(f"ɨ\u0303", "II_NAS_SEG")
    text = text.replace(f"o\u0303", "O_NAS_SEG")
    text = text.replace(f"u\u0303", "U_NAS_SEG")
    text = text.replace(f"ʊ\u0303", "UU_NAS_SEG")
    text = COMBINING_REMOVE_PATTERN.sub("", text)
    if text.startswith("ʔ"):
        text = text[1:]
    text = text.replace("QW_SEG", "qw")
    text = text.replace("KW_SEG", "kw")
    text = text.replace("GW_SEG", "gw")
    text = text.replace("RR_SEG", "rr")
    text = text.replace("R_SEG", "r")
    text = text.replace("K_STOP_SEG", "k")
    text = text.replace("TP_SEG", "tp")
    text = text.replace("DB_SEG", "db")
    text = text.replace("T_AFFR", "Ç_SEG")
    text = text.replace("D_AFFR", "C_SEG")
    text = text.replace("K_AFFR", "K_SEG")
    replacements = [
        ("ʕ", "'"),
        ("ʔ", "'"),
        ("ʡ", "'"),
        ("ɢ", "'g"),
        ("ɟ", "g"),
        ("ħ", "'h"),
        ("ʜ", "'h"),
        ("н", "'h"),
        ("ɭ", "'l"),
        ("ŋ", "ng"),
        ("ɲ", "ny"),
        ("nʲ", "ny"),
        ("nj", "ny"),
        ("ʃ", "ş"),
        ("ʒ", "j"),
        ("χ", "x"),
        ("x", "x"),
        ("ʁ", "ğ"),
        ("q", "q"),
        ("k", "k"),
        ("g", "g"),
        ("d", "d"),
        ("t", "t"),
        ("p", "p"),
        ("b", "b"),
        ("f", "f"),
        ("v", "v"),
        ("s", "s"),
        ("z", "z"),
        ("h", "h"),
        ("m", "m"),
        ("n", "n"),
        ("l", "l"),
        ("ɫ", "l"),
        ("ʎ", "l"),
        ("w", "w"),
        ("ʷ", "w"),
        ("ʲ", "y"),
        ("Y_GLIDE", "y"),
        ("J_GLIDE", "y"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    # Vowels after consonant replacement.
    text = text.replace("a", "a")
    text = text.replace("æ", "â")
    text = text.replace("e", "ê")
    text = text.replace("ɛ", "e")
    text = text.replace("ɨ", "ı")
    text = text.replace("i", "i")
    text = text.replace("o", "o")
    text = text.replace("ʊ", "u")
    text = text.replace("u", "û")
    text = text.replace("ə", "e")
    text = text.replace("ɔ", "o")
    text = text.replace("A", "a")
    text = text.replace("A_NAS_SEG", "a")
    text = text.replace("AE_NAS_SEG", "a")
    text = text.replace("E_NAS_SEG", "e")
    text = text.replace("EH_NAS_SEG", "e")
    text = text.replace("I_NAS_SEG", "i")
    text = text.replace("II_NAS_SEG", "ı")
    text = text.replace("O_NAS_SEG", "o")
    text = text.replace("U_NAS_SEG", "u")
    text = text.replace("UU_NAS_SEG", "u")

    # Final ɪ usually maps to orthographic e, but keep a small whitelist
    # for lexical items that are better written with final ı.
    text = text.replace("ɪ", "I_NEAR")
    if text.endswith("I_NEAR"):
        final_e = text[:-6] + "e"
        final_dotless_i = text[:-6] + "ı"
        text = final_dotless_i if final_dotless_i in FINAL_I_NEAR_DOTLESS_I_WHITELIST else final_e
    text = text.replace("I_NEAR", "i")

    # Bare trill marker to rr except word-initial.
    if text.startswith("rr"):
        text = "r" + text[2:]

    # Normalize apostrophe spacing and duplicates.
    text = text.replace("Ç_SEG", "ç")
    text = text.replace("C_SEG", "c")
    text = text.replace("K_SEG", "k")
    text = text.replace("nggw", "ngw")
    text = text.replace("ngg", "ng")
    text = text.replace("gg", "g")
    text = text.replace("''", "'")
    text = text.replace("rrr", "rr")
    text = text.replace("yyy", "yy")
    return unicodedata.normalize("NFC", text)


def orthography_from_ipa(ipa: str, recording_type: str) -> str:
    words = [orthography_word_from_ipa(part) for part in normalize_space(ipa).split(" ") if part]
    text = " ".join(words)
    text = sentence_case_z(text)
    if recording_type == "introduction":
        text = strip_terminal_punctuation(text) + "."
    elif recording_type == "question":
        text = strip_terminal_punctuation(text) + "?"
    return text


def choose_isolation_text(rows: list[dict[str, str]]) -> str:
    isolation_rows = [row for row in rows if row["recording_type"] == "isolation"]
    if isolation_rows:
        isolation_rows.sort(key=lambda row: int(row["recording_id"]))
        return isolation_rows[0]["generated_recording_text"]
    rows = sorted(rows, key=lambda row: int(row["recording_id"]))
    return strip_terminal_punctuation(rows[0]["generated_recording_text"])


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    by_word_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            recording_type = normalize_space(entry.get("recording_types", "")).split("|")[0] or "unknown"
            recording_ipa = normalize_space(entry.get("recording_ipa", ""))
            generated_text = orthography_from_ipa(recording_ipa, recording_type)
            review_flags = collect_review_flags(recording_ipa) + collect_output_review_flags(generated_text)
            review_flags = list(dict.fromkeys(review_flags))
            row = {
                "word_id": normalize_space(entry.get("word_id", "")),
                "recording_id": normalize_space(entry.get("recording_id", "")),
                "word_title": normalize_space(entry.get("word_title", "")),
                "word_translation": normalize_space(entry.get("word_translation", "")),
                "recording_text": normalize_space(entry.get("recording_text", "")),
                "recording_ipa": recording_ipa,
                "recording_type": recording_type,
                "generated_recording_text": generated_text,
                "review_flags": review_flags,
            }
            rows.append(row)
            by_word_id[row["word_id"]].append(row)

    import_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    flagged_rows: list[dict[str, object]] = []
    summary = Counter()

    for row in rows:
        rid = row["recording_id"]
        wid = row["word_id"]
        if row["review_flags"]:
            flagged_rows.append(
                {
                    "word_id": int(wid),
                    "recording_id": int(rid),
                    "word_title": row["word_title"],
                    "recording_type": row["recording_type"],
                    "recording_ipa": row["recording_ipa"],
                    "review_flags": row["review_flags"],
                    "current_recording_text": row["recording_text"],
                    "generated_recording_text": row["generated_recording_text"],
                    "current_word_translation": row["word_translation"],
                }
            )
            summary["flagged_rows_excluded"] += 1
            for flag in row["review_flags"]:
                summary[f"flag_{flag}"] += 1
            continue
        generated_text = row["generated_recording_text"]
        current_text = row["recording_text"]
        update_row: dict[str, object] = {
            "word_id": int(wid),
            "recording_id": int(rid),
        }
        changed = False
        if generated_text != current_text:
            update_row["recording_text"] = generated_text
            summary["recording_text_updates"] += 1
            if current_text:
                summary["recording_text_corrected"] += 1
            else:
                summary["recording_text_filled"] += 1
            changed = True

        if changed:
            import_rows.append(update_row)
            audit_rows.append(
                {
                    "word_id": int(wid),
                    "recording_id": int(rid),
                    "word_title": row["word_title"],
                    "recording_type": row["recording_type"],
                    "recording_ipa": row["recording_ipa"],
                    "old_recording_text": current_text,
                    "new_recording_text": update_row.get("recording_text", ""),
                }
            )

    summary_data = {
        "source_metadata_csv": str(metadata_path),
        "total_metadata_rows": len(rows),
        "import_rows": len(import_rows),
        "recording_text_updates": summary["recording_text_updates"],
        "recording_text_filled": summary["recording_text_filled"],
        "recording_text_corrected": summary["recording_text_corrected"],
        "flagged_rows_excluded": summary["flagged_rows_excluded"],
        "flag_initial_ʔ_manual_review": summary["flag_initial_ʔ_manual_review"],
        "flag_contains_ʊ_manual_review": summary["flag_contains_ʊ_manual_review"],
        "flag_contains_short_vowel_manual_review": summary["flag_contains_short_vowel_manual_review"],
        "flag_contains_epenthetic_vowel_manual_review": summary["flag_contains_epenthetic_vowel_manual_review"],
        "flag_non_prevelar_ŋ_manual_review": summary["flag_non_prevelar_ŋ_manual_review"],
        "flag_contains_residual_ipa_output_manual_review": summary["flag_contains_residual_ipa_output_manual_review"],
    }

    prefix = args.output_prefix
    import_path = output_dir / f"{prefix}.jsonl"
    audit_path = output_dir / f"{prefix}.audit.jsonl"
    flagged_path = output_dir / f"{prefix}.review_flags.jsonl"
    summary_path = output_dir / f"{prefix}.summary.json"

    with import_path.open("w", encoding="utf-8") as handle:
        for row in import_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with flagged_path.open("w", encoding="utf-8") as handle:
        for row in flagged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, ensure_ascii=False, indent=2)

    print(f"Wrote import:  {import_path}")
    print(f"Wrote audit:   {audit_path}")
    print(f"Wrote flags:   {flagged_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
