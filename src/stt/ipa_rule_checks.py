#!/usr/bin/env python3
import re
import unicodedata


MODIFIERS = {"ʰ", "ʷ", "ʲ"}
TIE_BAR = "͡"
ALLOWED_TIEBAR_CLUSTERS = tuple(
    dict.fromkeys(
        unicodedata.normalize("NFC", cluster)
        for cluster in (
            "t̪͡ʃ",
            "d̪͡ʒ",
            "t͡ʃ",
            "d͡ʒ",
            "c͡ç",
            "t̪͡s",
            "d̪͡z",
            "t͡s",
            "d͡z",
            "t̪͡ʙ̥",
            "t̪͡ʙ̥̆",
            "d̪͡ʙ",
            "d̪͡ʙ̆",
            "t̪͡p",
            "d̪͡b",
        )
    )
)
ALLOWED_ASPIRATION_BASES = (
    "t̪͡ʃ",
    "t͡ʃ",
    "c͡ç",
    "t͡s",
    "t̪͡s",
    "p",
    "t",
    "t̪",
    "k",
    "q",
    "c",
)


def normalize_ipa_for_rules(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _has_allowed_aspiration_base(prefix: str) -> bool:
    candidate = prefix
    while candidate.endswith(("ʷ", "ʲ")):
        candidate = candidate[:-1]
    return any(candidate.endswith(base) for base in ALLOWED_ASPIRATION_BASES)


def _has_base_before(text: str, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0 and not text[cursor].isspace():
        if text[cursor] not in MODIFIERS and text[cursor] != TIE_BAR:
            return True
        cursor -= 1
    return False


def _tie_bar_is_allowed_cluster(text: str, index: int) -> bool:
    for cluster in ALLOWED_TIEBAR_CLUSTERS:
        tie_index = cluster.find(TIE_BAR)
        if tie_index < 0:
            continue
        start = index - tie_index
        end = start + len(cluster)
        if start < 0 or end > len(text):
            continue
        if text[start:end] == cluster:
            return True
    return False


def find_ipa_rule_issues(value: str) -> list[dict[str, object]]:
    text = normalize_ipa_for_rules(value)
    issues: list[dict[str, object]] = []
    for index, char in enumerate(text):
        prev_char = text[index - 1] if index > 0 else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if char == TIE_BAR:
            if (
                not prev_char
                or not next_char
                or prev_char.isspace()
                or next_char.isspace()
                or prev_char in MODIFIERS
                or next_char in MODIFIERS
                or prev_char == TIE_BAR
                or next_char == TIE_BAR
            ):
                issues.append(
                    {
                        "type": "tie_bar_without_pair",
                        "index": index,
                        "context": text[max(0, index - 2) : index + 3],
                        "message": "A tie bar appears without two joined sounds.",
                    }
                )
            elif not _tie_bar_is_allowed_cluster(text, index):
                issues.append(
                    {
                        "type": "unsupported_tie_bar_cluster",
                        "index": index,
                        "context": text[max(0, index - 3) : index + 4],
                        "message": "This tie bar is not in the current approved set of complex segments.",
                    }
                )
            continue

        if char in MODIFIERS:
            if not _has_base_before(text, index):
                issues.append(
                    {
                        "type": "modifier_without_base_sound",
                        "index": index,
                        "context": text[max(0, index - 2) : index + 3],
                        "message": "A modifier appears without a base sound.",
                    }
                )
                continue

            if prev_char == char:
                issues.append(
                    {
                        "type": "repeated_ipa_modifier",
                        "index": index,
                        "context": text[max(0, index - 2) : index + 3],
                        "message": "The same IPA modifier is repeated on one sound.",
                    }
                )
                continue

            if char == "ʰ" and not _has_allowed_aspiration_base(text[:index]):
                issues.append(
                    {
                        "type": "aspiration_placement",
                        "index": index,
                        "context": text[max(0, index - 3) : index + 2],
                        "message": "Aspiration should usually follow a voiceless stop or affricate.",
                    }
                )
    return issues


def summarize_ipa_rule_issues(value: str) -> dict[str, object]:
    issues = find_ipa_rule_issues(value)
    return {
        "count": len(issues),
        "types": sorted({str(item.get("type") or "") for item in issues if item.get("type")}),
        "details": issues,
    }


def basic_repair_ipa_text(value: str) -> str:
    text = normalize_ipa_for_rules(value)
    if not text:
        return ""

    changed = True
    while changed:
        changed = False
        repaired: list[str] = []
        for index, char in enumerate(text):
            prev_char = repaired[-1] if repaired else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""

            if char == TIE_BAR:
                if (
                    not prev_char
                    or not next_char
                    or prev_char.isspace()
                    or next_char.isspace()
                    or prev_char in MODIFIERS
                    or next_char in MODIFIERS
                    or prev_char == TIE_BAR
                    or next_char == TIE_BAR
                ):
                    changed = True
                    continue
                if not _tie_bar_is_allowed_cluster(text, index):
                    changed = True
                    continue

            if char in MODIFIERS:
                if not _has_base_before(text, index):
                    changed = True
                    continue
                if prev_char == char:
                    changed = True
                    continue
                if char == "ʰ" and not _has_allowed_aspiration_base("".join(repaired)):
                    changed = True
                    continue

            repaired.append(char)

        text = normalize_ipa_for_rules("".join(repaired))

    return text
