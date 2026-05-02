#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent

TEXT_MANIFEST_HEADERS = [
    "split",
    "audio",
    "audio_windows",
    "audio_posix",
    "target_text",
    "target_text_full",
    "recording_text",
    "recording_translation",
    "recording_type",
    "word_title",
    "word_slug",
    "speaker_user_id",
    "source_bundle",
    "audio_relpath",
    "source_id",
    "notes",
]

WORD_RE = re.compile(r"[0-9A-Za-zÇĞİIÖŞÜçğıöşüÂâÊêÎîÔôÛû'’]+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
ALIGNMENT_CHAR_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
        "û": "u",
        "Û": "u",
        "ı": "i",
        "I": "i",
        "İ": "i",
        "ê": "e",
        "Ê": "e",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "ô": "o",
        "Ô": "o",
        "é": "e",
        "É": "e",
        "ğ": "g",
        "Ğ": "g",
        "ş": "s",
        "Ş": "s",
        "ç": "c",
        "Ç": "c",
        "ö": "o",
        "Ö": "o",
        "ü": "u",
        "Ü": "u",
    }
)


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def simplify_orthography(value: str) -> str:
    text = value
    replacements = {
        "Â": "E",
        "â": "e",
        "“": '"',
        "”": '"',
        "’": "'",
        "‘": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.lower()
    text = re.sub(r"[.,!?;:()\\[\\]\"]+", " ", text)
    return normalize_spaces(text)


def to_windows_path(path: Path) -> str:
    posix = path.resolve().as_posix()
    parts = posix.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper() + ":"
        tail = "\\".join(parts[3:])
        return drive + ("\\" + tail if tail else "")
    return str(path.resolve())


def win_to_posix(path_str: str) -> Path:
    windows_path = PureWindowsPath(path_str)
    drive = windows_path.drive.rstrip(":").lower()
    path = Path("/mnt") / drive
    for part in windows_path.parts[1:]:
        if part not in {"\\", "/"}:
            path /= part
    return path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def format_vtt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def write_vtt(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")
        for item in items:
            handle.write(f"{item['id']}\n")
            handle.write(f"{format_vtt_timestamp(float(item['start_sec']))} --> {format_vtt_timestamp(float(item['end_sec']))}\n")
            handle.write(f"{item['text_full']}\n\n")


def parse_numbered_clip_transcript(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*(\d+)\.\s*(.*)$", raw_line)
        if not match:
            continue
        rows[int(match.group(1))] = match.group(2).strip()
    if not rows:
        raise RuntimeError(f"No numbered clip rows found in {path}")
    return rows


def parse_paragraphs(text: str) -> list[str]:
    paragraphs = [normalize_spaces(block.replace("\n", " ")) for block in re.split(r"\n\s*\n", text) if normalize_spaces(block)]
    if not paragraphs:
        raise RuntimeError("Transcript did not contain any paragraphs")
    return paragraphs


def split_sentences(paragraph_text: str) -> list[str]:
    rough = [normalize_spaces(part) for part in SENTENCE_SPLIT_RE.split(paragraph_text) if normalize_spaces(part)]
    return rough or [paragraph_text]


def normalize_alignment_token(token: str) -> str:
    text = token.translate(ALIGNMENT_CHAR_MAP).lower()
    text = re.sub(r"[^a-z0-9\\[\\]']+", "", text)
    return text


def extract_surface_tokens(text: str) -> list[str]:
    return [match.group(0).replace("’", "'") for match in WORD_RE.finditer(text)]


def build_corrected_tokens(paragraphs: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paragraph_rows: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    sentence_counter = 0
    token_counter = 0
    for paragraph_index, paragraph_text in enumerate(paragraphs, start=1):
        paragraph_rows.append(
            {
                "id": paragraph_index,
                "text_full": paragraph_text,
            }
        )
        for sentence_text in split_sentences(paragraph_text):
            sentence_counter += 1
            sentence_rows.append(
                {
                    "id": sentence_counter,
                    "paragraph_id": paragraph_index,
                    "text_full": sentence_text,
                }
            )
            for surface in extract_surface_tokens(sentence_text):
                normalized = normalize_alignment_token(surface)
                if not normalized:
                    continue
                token_counter += 1
                tokens.append(
                    {
                        "id": token_counter,
                        "surface": surface,
                        "normalized": normalized,
                        "paragraph_id": paragraph_index,
                        "sentence_id": sentence_counter,
                    }
                )
    return paragraph_rows, sentence_rows, tokens


def build_reference_tokens(clip_text_rows: dict[int, str]) -> tuple[list[str], list[int]]:
    ref_tokens: list[str] = []
    ref_clip_ids: list[int] = []
    for clip_id in sorted(clip_text_rows):
        for surface in extract_surface_tokens(clip_text_rows[clip_id]):
            normalized = normalize_alignment_token(surface)
            if not normalized:
                continue
            ref_tokens.append(normalized)
            ref_clip_ids.append(clip_id)
    if not ref_tokens:
        raise RuntimeError("Reference clip transcript produced zero tokens")
    return ref_tokens, ref_clip_ids


def align_corrected_tokens_to_clips(reference_tokens: list[str], reference_clip_ids: list[int], corrected_tokens: list[dict[str, Any]]) -> None:
    corrected_norm = [token["normalized"] for token in corrected_tokens]
    matcher = difflib.SequenceMatcher(a=reference_tokens, b=corrected_norm, autojunk=False)
    corrected_to_reference: dict[int, int] = {}
    for tag, ref_start, ref_end, corr_start, corr_end in matcher.get_opcodes():
        if tag == "equal":
            for ref_index, corr_index in zip(range(ref_start, ref_end), range(corr_start, corr_end)):
                corrected_to_reference[corr_index] = ref_index
        elif tag == "replace":
            ref_span = list(range(ref_start, ref_end))
            corr_span = list(range(corr_start, corr_end))
            if ref_span and corr_span:
                for position, corr_index in enumerate(corr_span):
                    if len(corr_span) == 1:
                        ref_index = ref_span[len(ref_span) // 2]
                    else:
                        ratio = position / float(len(corr_span) - 1)
                        ref_index = ref_span[min(len(ref_span) - 1, round(ratio * (len(ref_span) - 1)))]
                    corrected_to_reference[corr_index] = ref_index

    mapped_indices = sorted(corrected_to_reference)
    for corr_index in range(len(corrected_tokens)):
        if corr_index in corrected_to_reference:
            continue
        left = max((index for index in mapped_indices if index < corr_index), default=None)
        right = min((index for index in mapped_indices if index > corr_index), default=None)
        if left is not None and right is not None:
            corrected_to_reference[corr_index] = (
                corrected_to_reference[left] if (corr_index - left) <= (right - corr_index) else corrected_to_reference[right]
            )
        elif left is not None:
            corrected_to_reference[corr_index] = corrected_to_reference[left]
        elif right is not None:
            corrected_to_reference[corr_index] = corrected_to_reference[right]
        else:
            raise RuntimeError("Unable to map corrected tokens to reference tokens")

    for corr_index, token in enumerate(corrected_tokens):
        ref_index = corrected_to_reference[corr_index]
        token["reference_token_index"] = ref_index
        token["clip_id"] = reference_clip_ids[ref_index]


def assign_token_times(corrected_tokens: list[dict[str, Any]], clip_rows: dict[int, dict[str, Any]]) -> None:
    tokens_by_clip: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for token in corrected_tokens:
        tokens_by_clip[int(token["clip_id"])].append(token)

    for clip_id, tokens in tokens_by_clip.items():
        clip_row = clip_rows[clip_id]
        start_sec = float(clip_row["start_sec"])
        end_sec = float(clip_row["end_sec"])
        duration_sec = max(0.0, end_sec - start_sec)
        if not tokens:
            continue
        step = duration_sec / float(len(tokens))
        for index, token in enumerate(tokens):
            token["start_sec"] = round(start_sec + (index * step), 3)
            token["end_sec"] = round(start_sec + ((index + 1) * step), 3)
            token["audio_windows"] = clip_row["source_audio_windows"]
            token["audio_posix"] = clip_row["source_audio_posix"]
            token["clip_audio_windows"] = clip_row["audio_windows"]
            token["clip_audio_posix"] = clip_row["audio_posix_resolved"]


def join_surface_tokens(tokens: list[dict[str, Any]]) -> str:
    return " ".join(token["surface"] for token in tokens).strip()


def build_group_rows(
    source_rows: list[dict[str, Any]],
    corrected_tokens: list[dict[str, Any]],
    clip_rows: dict[int, dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    tokens_by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for token in corrected_tokens:
        tokens_by_group[int(token[group_key])].append(token)

    output_rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        group_id = int(source_row["id"])
        tokens = tokens_by_group.get(group_id, [])
        if not tokens:
            continue
        start_sec = min(float(token["start_sec"]) for token in tokens)
        end_sec = max(float(token["end_sec"]) for token in tokens)
        clip_ids = sorted({int(token["clip_id"]) for token in tokens})
        output_rows.append(
            {
                "id": group_id,
                "text_full": source_row["text_full"],
                "text_projected": join_surface_tokens(tokens),
                "token_count": len(tokens),
                "clip_start": clip_ids[0],
                "clip_end": clip_ids[-1],
                "clip_count": len(clip_ids),
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "duration_sec": round(end_sec - start_sec, 3),
                "audio_windows": clip_rows[clip_ids[0]]["source_audio_windows"],
                "audio_posix": clip_rows[clip_ids[0]]["source_audio_posix"],
            }
        )
    return output_rows


def build_clip_projection_rows(corrected_tokens: list[dict[str, Any]], clip_rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    tokens_by_clip: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for token in corrected_tokens:
        tokens_by_clip[int(token["clip_id"])].append(token)

    rows: list[dict[str, Any]] = []
    for clip_id in sorted(clip_rows):
        clip_row = clip_rows[clip_id]
        tokens = tokens_by_clip.get(clip_id, [])
        text_full = join_surface_tokens(tokens)
        sentence_ids = sorted({int(token["sentence_id"]) for token in tokens})
        paragraph_ids = sorted({int(token["paragraph_id"]) for token in tokens})
        rows.append(
            {
                "clip_id": clip_id,
                "audio_filename": clip_row["audio_filename"],
                "audio_windows": clip_row["audio_windows"],
                "audio_posix": clip_row["audio_posix_resolved"],
                "audio_relpath": clip_row["audio_relpath"],
                "start_sec": round(float(clip_row["start_sec"]), 3),
                "end_sec": round(float(clip_row["end_sec"]), 3),
                "duration_sec": round(float(clip_row["duration_sec"]), 3),
                "projected_text_full": text_full,
                "projected_text": simplify_orthography(text_full),
                "token_count": len(tokens),
                "paragraph_ids": ",".join(str(item) for item in paragraph_ids),
                "sentence_ids": ",".join(str(item) for item in sentence_ids),
            }
        )
    return rows


def build_training_manifest_rows(clip_projection_rows: list[dict[str, Any]], story_slug: str, source_bundle: str) -> list[dict[str, Any]]:
    manifest_rows: list[dict[str, Any]] = []
    for row in clip_projection_rows:
        text_full = str(row["projected_text_full"]).strip()
        if not text_full:
            continue
        target_text = simplify_orthography(text_full)
        if not target_text:
            continue
        clip_id = int(row["clip_id"])
        manifest_rows.append(
            {
                "split": "train",
                "audio": row["audio_windows"],
                "audio_windows": row["audio_windows"],
                "audio_posix": row["audio_posix"],
                "target_text": target_text,
                "target_text_full": text_full,
                "recording_text": text_full,
                "recording_translation": "",
                "recording_type": "narrative_segment",
                "word_title": f"{story_slug} clip {clip_id:04d}",
                "word_slug": story_slug,
                "speaker_user_id": "0",
                "source_bundle": source_bundle,
                "audio_relpath": row["audio_relpath"],
                "source_id": f"{story_slug}_{clip_id:04d}",
                "notes": "projected from corrected story transcript onto existing phase-split clip timing",
            }
        )
    return manifest_rows


def build_highlight_payload(
    corrected_tokens: list[dict[str, Any]],
    paragraph_rows: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
    clip_projection_rows: list[dict[str, Any]],
    clip_rows: dict[int, dict[str, Any]],
    story_id: str,
) -> dict[str, Any]:
    first_clip = clip_rows[min(clip_rows)]
    return {
        "story_id": story_id,
        "source_audio_windows": first_clip["source_audio_windows"],
        "source_audio_posix": first_clip["source_audio_posix"],
        "approximate_word_timing": True,
        "timing_basis": "Corrected words were projected onto the existing phase-split clip sequence, then evenly distributed inside each clip.",
        "lines": paragraph_rows,
        "paragraphs": paragraph_rows,
        "sentences": sentence_rows,
        "clips": clip_projection_rows,
        "tokens": [
            {
                "id": int(token["id"]),
                "text": token["surface"],
                "clip_id": int(token["clip_id"]),
                "paragraph_id": int(token["paragraph_id"]),
                "sentence_id": int(token["sentence_id"]),
                "start_sec": round(float(token["start_sec"]), 3),
                "end_sec": round(float(token["end_sec"]), 3),
                "audio_windows": token["audio_windows"],
                "audio_posix": token["audio_posix"],
                "clip_audio_windows": token["clip_audio_windows"],
                "clip_audio_posix": token["clip_audio_posix"],
            }
            for token in corrected_tokens
        ],
    }


def build_content_lesson_payload(
    *,
    story_id: str,
    title: str,
    corrected_text: str,
    paragraph_rows: list[dict[str, Any]],
    sentence_rows: list[dict[str, Any]],
    clip_projection_rows: list[dict[str, Any]],
    highlight_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "content_lesson_import.v1",
        "lesson_id": story_id,
        "title": title,
        "audio_windows": highlight_payload["source_audio_windows"],
        "audio_posix": highlight_payload["source_audio_posix"],
        "transcript_full": corrected_text,
        "transcript_oneline": normalize_spaces(corrected_text.replace("\n", " ")),
        "display_mode": "line",
        "approximate_word_timing": True,
        "timing_basis": highlight_payload["timing_basis"],
        "lines": paragraph_rows,
        "sentences": sentence_rows,
        "clips": clip_projection_rows,
        "tokens": highlight_payload["tokens"],
    }


def read_corrected_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not normalize_spaces(text):
        raise RuntimeError(f"Transcript file is empty: {path}")
    return text


def build_clip_rows(phase_split_tsv: Path) -> dict[int, dict[str, Any]]:
    rows = read_tsv(phase_split_tsv)
    clip_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        clip_id = int(row["clip_id"])
        audio_windows = row["audio_windows"]
        source_audio_windows = row["source_audio_windows"]
        clip_rows[clip_id] = {
            **row,
            "clip_id": clip_id,
            "audio_windows": audio_windows,
            "audio_posix_resolved": win_to_posix(audio_windows).resolve().as_posix(),
            "source_audio_windows": source_audio_windows,
            "source_audio_posix": win_to_posix(source_audio_windows).resolve().as_posix(),
        }
    return clip_rows


def build_readme(story_id: str) -> str:
    return (
        f"# {story_id} alignment bundle\n\n"
        "Files in this folder:\n"
        "- `clip_projection.tsv`: corrected text projected onto the existing phase-split clip timing\n"
        "- `manifest_text_clip_projected.tsv`: training-ready orthography manifest using the existing clip WAV files\n"
        "- `line_alignment.tsv`: line/block-level ranges over the source story audio using the transcript blocks as provided\n"
        "- `paragraph_alignment.tsv`: paragraph-level ranges over the source story audio\n"
        "- `line_alignment.vtt`: line/block timing for straightforward phrase highlighting in a website\n"
        "- `sentence_alignment.tsv`: sentence-level ranges over the source story audio\n"
        "- `sentence_alignment.vtt`: sentence subtitles/timing for quick web playback tests\n"
        "- `highlight_approx.json`: token + sentence + paragraph timing payload for a website\n"
        "- `content_lesson_import.json`: single-file lesson payload with transcript + audio + line/sentence/token timings\n"
        "- `summary.json`: counts and source paths\n"
        "- `source_transcript.txt`: corrected transcript used as the input\n\n"
        "Important note:\n"
        "- line/block timing is the safest default for phrase highlighting\n"
        "- sentence and paragraph timing are also usable for phrase highlighting\n"
        "- token timing is approximate, not true forced alignment\n"
        "- for exact word-level highlighting, a dedicated acoustic forced-alignment pass would still be better\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a corrected story transcript onto an existing phase-split clip sequence.")
    parser.add_argument("--story-id", required=True, help="Short story id used in output metadata.")
    parser.add_argument("--title", default=None, help="Optional human-readable title for lesson payloads.")
    parser.add_argument("--phase-split-tsv", required=True, help="Path to the phase_split.tsv file.")
    parser.add_argument("--clip-transcript", required=True, help="Numbered clip transcript file used as the timing backbone.")
    parser.add_argument("--corrected-transcript", required=True, help="Corrected transcript text file.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the alignment bundle into.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_split_tsv = Path(args.phase_split_tsv).expanduser().resolve()
    clip_transcript = Path(args.clip_transcript).expanduser().resolve()
    corrected_transcript = Path(args.corrected_transcript).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_rows = build_clip_rows(phase_split_tsv)
    clip_transcript_rows = parse_numbered_clip_transcript(clip_transcript)
    corrected_text = read_corrected_transcript(corrected_transcript)
    paragraphs = parse_paragraphs(corrected_text)
    paragraph_rows, sentence_rows, corrected_tokens = build_corrected_tokens(paragraphs)
    reference_tokens, reference_clip_ids = build_reference_tokens(clip_transcript_rows)
    align_corrected_tokens_to_clips(reference_tokens, reference_clip_ids, corrected_tokens)
    assign_token_times(corrected_tokens, clip_rows)

    paragraph_alignment = build_group_rows(paragraph_rows, corrected_tokens, clip_rows, "paragraph_id")
    sentence_alignment = build_group_rows(sentence_rows, corrected_tokens, clip_rows, "sentence_id")
    clip_projection_rows = build_clip_projection_rows(corrected_tokens, clip_rows)
    manifest_rows = build_training_manifest_rows(
        clip_projection_rows=clip_projection_rows,
        story_slug=str(args.story_id),
        source_bundle=f"{args.story_id}_corrected_alignment",
    )
    highlight_payload = build_highlight_payload(
        corrected_tokens=corrected_tokens,
        paragraph_rows=paragraph_alignment,
        sentence_rows=sentence_alignment,
        clip_projection_rows=clip_projection_rows,
        clip_rows=clip_rows,
        story_id=str(args.story_id),
    )
    content_lesson_payload = build_content_lesson_payload(
        story_id=str(args.story_id),
        title=str(args.title or args.story_id),
        corrected_text=corrected_text,
        paragraph_rows=paragraph_alignment,
        sentence_rows=sentence_alignment,
        clip_projection_rows=clip_projection_rows,
        highlight_payload=highlight_payload,
    )

    write_tsv(
        output_dir / "clip_projection.tsv",
        clip_projection_rows,
        [
            "clip_id",
            "audio_filename",
            "audio_windows",
            "audio_posix",
            "audio_relpath",
            "start_sec",
            "end_sec",
            "duration_sec",
            "projected_text_full",
            "projected_text",
            "token_count",
            "paragraph_ids",
            "sentence_ids",
        ],
    )
    write_tsv(
        output_dir / "line_alignment.tsv",
        paragraph_alignment,
        ["id", "text_full", "text_projected", "token_count", "clip_start", "clip_end", "clip_count", "start_sec", "end_sec", "duration_sec", "audio_windows", "audio_posix"],
    )
    write_tsv(
        output_dir / "paragraph_alignment.tsv",
        paragraph_alignment,
        ["id", "text_full", "text_projected", "token_count", "clip_start", "clip_end", "clip_count", "start_sec", "end_sec", "duration_sec", "audio_windows", "audio_posix"],
    )
    write_tsv(
        output_dir / "sentence_alignment.tsv",
        sentence_alignment,
        ["id", "text_full", "text_projected", "token_count", "clip_start", "clip_end", "clip_count", "start_sec", "end_sec", "duration_sec", "audio_windows", "audio_posix"],
    )
    write_tsv(output_dir / "manifest_text_clip_projected.tsv", manifest_rows, TEXT_MANIFEST_HEADERS)
    write_vtt(output_dir / "line_alignment.vtt", paragraph_alignment)
    write_vtt(output_dir / "sentence_alignment.vtt", sentence_alignment)
    (output_dir / "highlight_approx.json").write_text(json.dumps(highlight_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "content_lesson_import.json").write_text(
        json.dumps(content_lesson_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "source_transcript.txt").write_text(corrected_text, encoding="utf-8")
    (output_dir / "source_clip_transcript.txt").write_text(
        "\n".join(f"{clip_id}. {clip_transcript_rows[clip_id]}" for clip_id in sorted(clip_transcript_rows)),
        encoding="utf-8",
    )

    summary = {
        "story_id": str(args.story_id),
        "phase_split_tsv": phase_split_tsv.as_posix(),
        "clip_transcript": clip_transcript.as_posix(),
        "corrected_transcript": corrected_transcript.as_posix(),
        "source_audio_windows": clip_rows[min(clip_rows)]["source_audio_windows"],
        "source_audio_posix": clip_rows[min(clip_rows)]["source_audio_posix"],
        "clip_count": len(clip_projection_rows),
        "clip_training_rows": len(manifest_rows),
        "line_count": len(paragraph_alignment),
        "paragraph_count": len(paragraph_alignment),
        "sentence_count": len(sentence_alignment),
        "token_count": len(corrected_tokens),
        "approximate_word_timing": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(build_readme(str(args.story_id)), encoding="utf-8")

    print(f"Wrote alignment bundle: {output_dir}")


if __name__ == "__main__":
    main()
