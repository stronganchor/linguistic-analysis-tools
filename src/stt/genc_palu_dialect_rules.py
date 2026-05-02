#!/usr/bin/env python3
import re
import unicodedata


FRONT_VOWELS = "ieɛɜɪ"
VOWELS = "aeiouɨɪɛæœəɜʉʊɔɑɒ"

PRESERVE_SEGMENTS = {
    "t̪͡ʃ": "§TDSH§",
    "d̪͡ʒ": "§DDJ§",
    "t͡ʃ": "§TSH§",
    "d͡ʒ": "§DJ§",
    "c͡ç": "§CCH§",
    "t̪": "§TD§",
    "d̪": "§DD§",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def _normalize_meta(word_title: str, word_translation: str) -> str:
    meta = f"{word_title} {word_translation}".lower()
    meta = meta.replace("ı", "i")
    meta = meta.replace("ê", "e")
    meta = meta.replace("â", "a")
    meta = meta.replace("û", "u")
    meta = meta.replace("'", "")
    meta = re.sub(r"\s+", " ", meta)
    return meta.strip()


def _tokenize_spaces(text: str) -> list[str]:
    return re.split(r"(\s+)", text)


def repair_genc_palu_ipa_text(
    value: str,
    *,
    word_title: str = "",
    word_translation: str = "",
) -> str:
    repaired, _ = repair_genc_palu_ipa_with_details(
        value,
        word_title=word_title,
        word_translation=word_translation,
    )
    return repaired


def summarize_genc_palu_dialect_repairs(
    value: str,
    *,
    word_title: str = "",
    word_translation: str = "",
) -> dict[str, object]:
    repaired, applied_rules = repair_genc_palu_ipa_with_details(
        value,
        word_title=word_title,
        word_translation=word_translation,
    )
    return {
        "changed": repaired != _normalize(value),
        "repaired": repaired,
        "types": applied_rules,
        "count": len(applied_rules),
    }


def repair_genc_palu_ipa_with_details(
    value: str,
    *,
    word_title: str = "",
    word_translation: str = "",
) -> tuple[str, list[str]]:
    text = _normalize(value)
    if not text:
        return "", []

    applied: list[str] = []
    original = text

    for source, placeholder in PRESERVE_SEGMENTS.items():
        text = text.replace(source, placeholder)

    dentalized = text.replace("t", "t̪").replace("d", "d̪")
    if dentalized != text:
        applied.append("dentalize_t_d")
        text = dentalized

    front_shift = re.sub(rf"k(?=[{FRONT_VOWELS}])", "c͡ç", text)
    front_shift = re.sub(rf"c(?=[{FRONT_VOWELS}])", "c͡ç", front_shift)
    if front_shift != text:
        applied.append("front_k_to_c_cedilla")
        text = front_shift

    aspirated_parts: list[str] = []
    aspiration_changed = False
    for piece in _tokenize_spaces(text):
        if not piece or piece.isspace():
            aspirated_parts.append(piece)
            continue
        token = piece
        if token.startswith("p") and not token.startswith("pʰ") and len(token) > 1 and token[1] in VOWELS:
            token = "pʰ" + token[1:]
        if token.startswith("t̪") and not token.startswith("t̪ʰ") and len(token) > 2 and token[2] in VOWELS:
            token = "t̪ʰ" + token[2:]
        if token.startswith("k") and not token.startswith("kʰ") and len(token) > 1 and token[1] in VOWELS:
            token = "kʰ" + token[1:]
        if token.startswith("q") and not token.startswith("qʰ") and len(token) > 1 and token[1] in VOWELS:
            token = "qʰ" + token[1:]
        if (
            token.startswith("c")
            and not token.startswith("cʰ")
            and not token.startswith("c͡ç")
            and len(token) > 1
            and token[1] in VOWELS
        ):
            token = "cʰ" + token[1:]
        if token != piece:
            aspiration_changed = True
        aspirated_parts.append(token)
    if aspiration_changed:
        applied.append("aspirate_voiceless_onsets")
    text = "".join(aspirated_parts)

    for source, placeholder in PRESERVE_SEGMENTS.items():
        text = text.replace(placeholder, source)

    affricate_dentalized = text.replace("t͡ʃ", "t̪͡ʃ").replace("d͡ʒ", "d̪͡ʒ")
    if affricate_dentalized != text:
        applied.append("dentalize_affricates")
        text = affricate_dentalized

    meta = _normalize_meta(word_title, word_translation)
    special_voiceless = any(
        pattern in meta
        for pattern in (
            "tp",
            " tw",
            "tw ",
            "twer",
            "tpir",
            "tper",
            "twern",
            "twape",
            "tpez",
            "twir",
            "twel",
            "astwar",
            "satwir",
            "zeytwin",
            "beytun",
        )
    )
    special_vistori = any(pattern in meta for pattern in ("vistori", "visturi"))
    special_baldiz = "baldiz" in meta
    special_dwi = "dwi" in meta

    if special_voiceless:
        # Default automation now emits the reduced bilabial-release notation.
        voiceless_text = re.sub(r"jt̪(?:ʷʰ|ʷ|ʰ)(?=[eɨouiɛɪ])", "jt̪͡ʙ̥̆", text)
        voiceless_text = re.sub(r"t̪(?:ʰw|w)(?=[aeiouɨɪɛæœəɜʉʊɔɑɒ])", "t̪͡ʙ̥̆", voiceless_text)
        voiceless_text = re.sub(r"t̪(?:ʷʰ|ʷ|pʰ|p|ʰ)(?=[eɨouiɛɪ])", "t̪͡ʙ̥̆", voiceless_text)
        voiceless_text = re.sub(r"pʷʰ?(?=[eɨouiɛɪ])", "t̪͡ʙ̥̆", voiceless_text)
        voiceless_text = re.sub(r"\bp(?=[eɨouiɛɪ])", "t̪͡ʙ̥̆", voiceless_text)
        if voiceless_text != text:
            applied.append("special_tw_tp_segment")
            text = voiceless_text

    if special_vistori:
        vistori_text = re.sub(r"v[ɪi]st̪(?:ʷʰ|ʷ|ʰ)(?=[ouɨ])", "vɪst̪͡ʙ̥̆", text)
        vistori_text = re.sub(r"w[ɪi]st̪(?:ʷʰ|ʷ|ʰ)(?=[ouɨ])", "vɪst̪͡ʙ̥̆", vistori_text)
        vistori_text = re.sub(r"v[ɪi]st̪͡ʙ̥̆ɔ", "vɪst̪͡ʙ̥̆o", vistori_text)
        vistori_text = re.sub(r"v[ɪi]st̪ʰɔ", "vɪst̪͡ʙ̥̆o", vistori_text)
        if vistori_text != text:
            applied.append("special_vistori_segment")
            text = vistori_text

    if special_baldiz:
        baldiz_text = re.sub(r"ba[lzw]?[zw]?b[ɨiı]za", "bald̪͡bɨza", text)
        baldiz_text = re.sub(r"ba[lzw]?[zw]?b[ɨiı]z", "bald̪͡bɨz", baldiz_text)
        if baldiz_text != text:
            applied.append("special_baldiz_segment")
            text = baldiz_text

    if special_dwi:
        dwi_text = re.sub(r"\bd̪o\b", "d̪ʷɪ", text)
        dwi_text = re.sub(r"\bdo\b", "d̪ʷɪ", dwi_text)
        dwi_text = re.sub(r"\bbɨ\b", "d̪ʷɪ", dwi_text)
        dwi_text = re.sub(r"\bbe\b", "d̪ʷɪ", dwi_text)
        if dwi_text != text:
            applied.append("special_dwi_segment")
            text = dwi_text

    if text == original:
        return text, []
    return text, applied
