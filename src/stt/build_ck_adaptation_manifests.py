#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import wave
from pathlib import Path, PureWindowsPath
from typing import Any

from model_paths import stt_model_path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "training_data" / "ck-adaptation-tier1-20260410"
CLIPS_DIR = OUTPUT_DIR / "clips"

EXPLORING_MANIFEST = ROOT_DIR / "audio_corpus" / "training_data" / "manifest.tsv"
MESELE_TSV = ROOT_DIR / "audio_corpus" / "corrected_segments" / "Mesele_kurık_lir_with_unclear_ending.tsv"
MESELE_AUDIO = ROOT_DIR / "audio_corpus" / "audio_wav16k" / "Mesele kurık lir.wav"
SABRI_CLIP_1 = ROOT_DIR / "audio_corpus" / "training_data" / "clips" / "sabri_neweş_0001.wav"
PHASE1A_SPLIT_TSV = SCRIPT_DIR / "transcription_outputs" / "phase1a_processed_multi_ipa_cpu_20260410-133537" / "phase_split.tsv"
PHASE1A_CLIPS_DIR = SCRIPT_DIR / "transcription_outputs" / "phase1a_processed_multi_ipa_cpu_20260410-133537" / "clips"
PHASE1A_STATUS_TSV = SCRIPT_DIR / "transcription_outputs" / "phase1a_first100_multi_ipa_cpu_20260410-144110" / "phase1a_first100_decision_status_20260410.tsv"
PHASE1A_MANUAL_IMPORT_1 = Path("$HOME/Downloads/unresolved-ipa-review-import.jsonl")
PHASE1A_MANUAL_IMPORT_2 = Path("$HOME/Downloads/unresolved-ipa-review-import (1).jsonl")
AK_ANCHOR_MANIFEST = SCRIPT_DIR / "training_data" / "genc-palu-recording_ipa-20260407-124628" / "manifest_ipa.tsv"
AK_TEXT_ANCHOR_MANIFEST = SCRIPT_DIR / "training_data" / "genc-palu-recording_text-20260409-071931-simple-v001" / "manifest_text.tsv"

IPA_MANIFEST_HEADERS = [
    "split",
    "audio",
    "audio_windows",
    "audio_posix",
    "target_ipa",
    "recording_text",
    "recording_translation",
    "recording_type",
    "word_title",
    "word_slug",
    "categories",
    "category_names",
    "speaker_user_id",
    "source_bundle",
    "audio_relpath",
    "source_id",
    "notes",
]

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


MESELE_ROWS: list[dict[str, Any]] = [
    {
        "row_index": 1,
        "orth_full": "Xal o xalcini von ma Kwırık lir ra bıqêd. Ma hın Kwırık lir caverd, ma şı. Ee, xal Kwırık lir ra von yo here yin esta. Von ına her biber.",
        "ipa": "χal o χald͡ʒini von ma kʷɨɾɪkʰ liɾ ɾa bɨqʰed̪ ma hɨn kʷɨɾɪkʰ liɾ d͡ʒavɛɾd̪ ma ʃɨ ɛɛ χal kʷɨɾɪkʰ liɾ ɾa von jo hɛɾe jin ɛst̪ʰa von ɨna hɛɾ bibɛɾ",
    },
    {
        "row_index": 2,
        "orth_full": "Yegat baxçıt best wa bıçer. Wı her gen şın baxçı şın dwir. Hedi xalcini ra vona kılçon poç. Ma gen erzen xwe, haybe xwe.",
        "ipa": "jɛgat̪ʰ baχt͡ʃɨt̪ʰ bɛst̪ʰ wa bɨt͡ʃɛɾ wɨ hɛɾ gɛn ʃɨn baχt͡ʃɨ ʃɨn d̪͡biɾ hɛd̪i χald͡ʒini ɾa vona kʰɨlt͡ʃon pʰot͡ʃ ma gɛn æɾzen χwɛ hajbe χwɛ",
    },
    {
        "row_index": 3,
        "orth_full": "Kwırık lir wa neşnaw. Ma şin. Kwırık liriz yin gweştare ken. Wıle Kwırık lir twe xwe erzen zerrê haybi.",
        "ipa": "kʷɨɾɪkʰ liɾ wa nɛʃnaw ma ʃin kʷɨɾɪkʰ liɾiz jin gʷɛʃt̪aɾe kʰɛn wɨlɛ kʷɨɾɪkʰ liɾ t̪͡pɛ χwɛ æɾzen zɛɾe hajbi",
    },
    {
        "row_index": 4,
        "orth_full": "Xwe erzen zerrê haybi. Xal şın astwar gen. Xal şın mı va astwar. Her gen on. Her gen on. Ey kılçon mılçon ken zerrê haybi. Ken dı.",
        "ipa": "χwɛ æɾzen zɛɾe hajbi χal ʃɨn ast̪ʰwaɾ gɛn χal ʃɨn mɨ va ast̪ʰwaɾ hɛɾ gɛn on hɛɾ gɛn on ɛj kʰɨlt͡ʃon mɨlt͡ʃon kʰɛn zɛɾe hajbi kʰɛn d̪ɨ",
    },
    {
        "row_index": 5,
        "orth_full": "Xalciniz vona: \"Ee, vona mêrik ma çim, ma raştıb, ma çi ra Kwırık lir ra qêdye.\" Ya nışena her. Ya nışena her. Wiz haybe erzen xwe kıft.",
        "ipa": "χald͡ʒiniz vona ɛɛ vona mʲeɾik ma t͡ʃim ma ɾaʃt̪ʰɨb ma t͡ʃi ɾa kʷɨɾɪkʰ liɾ ɾa qʰed̪je ja nɨʃɛna hɛɾ ja nɨʃɛna hɛɾ wiz hajbe æɾzen χwɛ kʰɨft̪ʰ",
    },
    {
        "row_index": 6,
        "orth_full": "Xalız haybe erzen xwe kıft kwen caade. Xal von: \"Yaw cinêk, tı çine eşt ın haybe? Zaf gıron. Yani, tı çine eşte?\" Yaz vona mı piri kılç pot, kılç rwenin.",
        "ipa": "χalɨz hajbe æɾzen χwɛ kʰɨft̪ʰ kʷen d͡ʒaad̪ɛ χal von jaw d͡ʒinek t̪͡pɨ t͡ʃinɛ ɛʃt̪ʰ ɨn hajbe zaf gɨɾon jani t̪͡pɨ t͡ʃinɛ ɛʃt̪ʰ d̪͡ʒɨ jaz vona mɨ t̪͡piɾi kʰɨlt͡ʃ pʰot̪ʰ kʰɨlt͡ʃ ɾwɛnin",
    },
    {
        "row_index": 7,
        "orth_full": "Eye ra belki biyu gıron. Von neyse, von kwen caade. Şın ser yo kwe. Heveri şın ser yo kwe. Miyon yo dow ra viyeren, kutik azyen cı. Çend hep kutik azyen",
        "ipa": "ɛjɛ ɾa bɛlkʰi biju gɨɾon von nɛjsɛ von kʷen d͡ʒaad̪ɛ ʃɨn sɛɾ jo kʷe hɛvɛɾi ʃɨn sɛɾ jo kʷe mijon jo d̪ow ɾa vijɛɾɛn kʰut̪ʰikʰ alzjɛn d͡ʒɨ t͡ʃɛnd̪ hɛpʰ kʰut̪ʰikʰ alzjɛn",
    },
    {
        "row_index": 8,
        "orth_full": "yin. Xal xalcini ra von: \"Cinêk cinêk, bo tı seke? İnke Kwırık lir itiya bibinê, ma ın kutıgon ra qêdnene. Tı va ma Kwırık lir caver. Bo inki ma seken?\" Çıqêdı Kwırık lir.",
        "ipa": "jin χal χald͡ʒini ɾa von d͡ʒinek d͡ʒinek bo t̪͡pɨ sɛkʰɛ inkʰɛ kʷɨɾɪkʰ liɾ itija bibinje ma ɨn kʷʰɨt̪ɨgon ɾa qʰed̪nene t̪͡pɨ va ma kʷɨɾɪkʰ liɾ ɾa d͡ʒavɛɾd̪ bo inkʰi ma sɛkʰɛn t͡ʃɨqʰed̪ɨ kʷɨɾɪkʰ liɾ",
    },
    {
        "row_index": 9,
        "orth_full": "Pê hasyen Kwırık lir ya haybe ra sarê xwe vecen von: \"Xalo mı rınd, ez ho itiya.\" Xal yı non rı, wıle yen war, pey ra kwen kutıgon dıma, kutıgon fetelnen. Xal ben şaş. Neyse, don purye caade ra şın. Xalcini vona: \"Ma ini ra çıqêdı bıqêd?\"",
        "ipa": "pʰe hasjɛn kʷɨɾɪkʰ liɾ ja hajbe ɾa saɾe χwɛ vɛd͡ʒɛn von χalo mɨ ɾʷɨn ɛz ho itija χal jɨ non ɾɨ wɨlɛ jɛn waɾ pʰɛj ɾa kʷen kʷʰɨt̪ɨgon d̪ɨma kʷʰɨt̪ɨgon fɛt̪elnɛn χal bɛn ʃaʃ nɛjsɛ d̪on pʰʊɾjɛ d͡ʒaad̪ɛ ɾa ʃɨn χald͡ʒini vona ma ini ɾa t͡ʃɨqʰed̪ɨ bɨqʰed̪",
    },
    {
        "row_index": 10,
        "orth_full": "İni ra ma bıqêd. Şın ser yo pırd. Şo von ma itiya kwen ra. Xal vona, xalcini vona ma ini ra bıqêd. Vona ma ini, şo ma ini erzen zerrê owk. Wa are mada rakwır.",
        "ipa": "ini ɾa ma bɨqʰed̪ ʃɨn sɛɾ jo pʰɨɾd̪ ʃo von ma itija kʷen ɾa χal vona χald͡ʒini vona ma ini ɾa bɨqʰed̪ vona ma ini ʃo ma ini æɾzen zɛɾe owkʰ wa aɾɛ mad̪a ɾakʷɨɾ",
    },
    {
        "row_index": 11,
        "orth_full": "Xal von la ma çıqêdı, von ını ma hasyen. Xalcini vona: \"E, vono gım.\" Tı von hım. Hani ma otır yo zon, wa wı fa'hım nık. Ma von hım gım, ma ey erzen.",
        "ipa": "χal von la ma t͡ʃɨqʰed̪ɨ von ɨnɨ ma hasjɛn χald͡ʒini vona ɛ vono gɨm t̪ʰɨ von hɨm hani ma ot̪ɨɾ jo zon wa wɨ faħm nɨkʰ ma von hɨm gɨm ma ɛj æɾzen",
    },
    {
        "row_index": 12,
        "orth_full": "Tamom von. Wı ho yin gweştare ken. Şo kwen miyon cılon. Kwırık lir ho wertıda. Kwırık lir oynyen wırd kot ra. Kwırık lir twe wırzen we. Xalcini onciyen ver xwe.",
        "ipa": "t̪ʰamom von wɨ ho jin gʷɛʃt̪aɾe kʰɛn ʃo kʷen mijon d͡ʒɨlon kʷɨɾɪkʰ liɾ ho wɛɾt̪ʰɨd̪a kʷɨɾɪkʰ liɾ ojnjɛn wɨɾd̪ kʰot̪ʰ ɾa kʷɨɾɪkʰ liɾ t̪͡pwɛ wɨɾzɛn wɛ χald͡ʒini ond͡ʒijɛn veɾ χwɛ",
    },
    {
        "row_index": 13,
        "orth_full": "O hete xalciniya kwen ra. Wıle pirik wırzen we von hım. Kwırık liriz von gım: \"Ben lingon xalciniya.\" Wı von gım, wı von hım.",
        "ipa": "o hɛt̪ʰɛ χald͡ʒinija kʷen ɾa wɨlɛ pʰiɾikʰ wɨɾzɛn wɛ von hɨm kʷɨɾɪkʰ liɾiz von gɨm bɛn lingon χald͡ʒinija wɨ von gɨm wɨ von hɨm",
    },
    {
        "row_index": 14,
        "orth_full": "Ve zit gen zerrê owk. Wıle zit gen zerrê owk. Xal Kwırık lir von: \"Wıle ma hım gım va ma tı ra qêdye.\" Wiz dest erzen ye paşt von e, wıle xalo von: \"Ma hım gım va von ma xalcini ra qêdye.\" Oynyen",
        "ipa": "vɛ zit̪ʰ gɛn zɛɾe owkʰ wɨlɛ zit̪ʰ gɛn zɛɾe owkʰ χal kʷɨɾɪkʰ liɾ von wɨlɛ ma hɨm gɨm va ma t̪ʰɨ ɾa qʰed̪je wiz d̪ɛst̪ʰ æɾzen jɛ pʰaʃt̪ʰ von ɛ wɨlɛ χalo von ma hɨm gɨm va von ma χald͡ʒini ɾa qʰed̪je ojnjɛn",
    },
]

SABRI_IPA_ROW_2 = (
    "qʰe sabɾi t̪͡pɛ nɨwɛʃ jani vona wɨle hiɾi hab non wɛ nɪʃko non bɨwuɾ "
    "jani zaf taj non wo sabɾi mɨ nɨwɛʃ mɨsafiɾ ʃaʃ bi von t͡ʃɨqʰedɨ nɨwɛʃ "
    "hiɾi hab non wɛ ja vona sabɾi mɨ pont͡ʃ hab bin nɪzde pont͡ʃ ħab non nɨwuɾsɛ mɨɾd nɨbɛn"
)

SABRI_ORTH_ROW_2 = (
    'Qe, Sabri tpe neweş? Yani vona wıle hiri hab non we, nışko non bıwur. '
    'Yani zaf tay non wo, Sabri mı neweş. Misafir şaş bi: "Von çıqedı neweş?" '
    'Hiri hab non we. Ya vona: "Sabri mı ponc hab bin. Nizde ponc hab non nıwurse, mırd nıben."'
)

README_TEXT = """# C.K. Tier1 Warm-Start Prep

This folder is a prepared, not-yet-run, C.K. adaptation package.

Contents:
- `manifest_ipa_ck_only.tsv`: trusted C.K. IPA set
- `manifest_ipa_ck_plus_ak_anchor.tsv`: same C.K. IPA set plus a small A.K. anchor subset
- `manifest_text_ck_only.tsv`: trusted C.K. orthography set in the simplified orthography target space used by the current best orthography MMS branch
- `manifest_text_ck_plus_ak_anchor.tsv`: same C.K. orthography set plus a small A.K. anchor subset
- `summary.json`: counts, durations, and split information
- `excluded_items.jsonl`: rows intentionally left out of the tier1 set
- `train_ck_mms_ipa_ck_only_wsl.sh` / `.cmd`
- `train_ck_mms_ipa_ck_plus_ak_anchor_wsl.sh` / `.cmd`
- `train_ck_mms_orth_ck_only_wsl.sh` / `.cmd`
- `train_ck_mms_orth_ck_plus_ak_anchor_wsl.sh` / `.cmd`

Preparation choices:
- `Mesele`: rows 1-14 included, final uncertain ending excluded
- `Sabri`: only the second coarse clip included; the first coarse clip was excluded because the uncertain `hefil` clause is embedded inside it
- `phase1a`: only manual accepts plus exact MMS/Whisper agreements included; auto-pattern rows, unresolved rows, and flagged skips excluded
- `Orho's story`: exclude from future C.K.-focused training assembly because it appears to contain O.E.'s speech
- IPA eval: held-out subset of trusted `phase1a` rows only
- Orthography eval: held-out subset of `Exploring a mine` plus two `Mesele` rows

When the GPU is free:
1. Run the desired `.cmd` launcher from Windows, or the `.sh` script from WSL.
2. Compare the resulting C.K. model against the untouched A.K. base on held-out `phase1a` eval rows.
3. Use the winner to bootstrap more `phase1a` review.
"""


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def to_windows_path(path: Path) -> str:
    posix = path.resolve().as_posix()
    parts = posix.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper() + ":"
        tail = "\\".join(parts[3:])
        return drive + ("\\" + tail if tail else "")
    return str(path.resolve())


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_ipa(value: str) -> str:
    return normalize_spaces(value)


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
    text = re.sub(r"[.,!?;:()\"]+", " ", text)
    text = normalize_spaces(text)
    return text


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_elan_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].startswith('"#file:///'):
                continue
            if len(row) < 7:
                continue
            if not row[0].strip() or not row[6].strip():
                continue
            rows.append(
                {
                    "label": row[0].strip(),
                    "start_sec": float(row[3].strip()),
                    "end_sec": float(row[5].strip()),
                    "text": row[6].strip(),
                }
            )
    return rows


def cut_wav_segment(source_path: Path, start_sec: float, end_sec: float, destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_path), "rb") as source:
        rate = source.getframerate()
        sample_width = source.getsampwidth()
        channels = source.getnchannels()
        comp_type = source.getcomptype()
        comp_name = source.getcompname()
        start_frame = max(0, int(round(start_sec * rate)))
        end_frame = max(start_frame, int(round(end_sec * rate)))
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        with wave.open(str(destination), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(rate)
            output.setcomptype(comp_type, comp_name)
            output.writeframes(frames)
    return (end_frame - start_frame) / float(rate)


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def hash_sort_key(value: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest()


def choose_eval_subset(keys: list[str], count: int, seed: int) -> set[str]:
    ordered = sorted(keys, key=lambda item: hash_sort_key(item, seed))
    return set(ordered[: max(1, min(count, len(ordered)))])


def win_to_posix(path_str: str) -> Path:
    windows_path = PureWindowsPath(path_str)
    drive = windows_path.drive.rstrip(":").lower()
    path = Path("/mnt") / drive
    for part in windows_path.parts[1:]:
        if part not in {"\\", "/"}:
            path /= part
    return path


def build_phase1a_trusted_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_rows = {int(row["clip_id"]): row for row in read_tsv(PHASE1A_SPLIT_TSV)}
    status_rows = {int(row["clip_id"]): row for row in read_tsv(PHASE1A_STATUS_TSV)}
    manual_rows_2 = {int(row["recording_id"]): row for row in read_jsonl(PHASE1A_MANUAL_IMPORT_2)}

    trusted: dict[int, str] = {}
    excluded: list[dict[str, Any]] = []
    for clip_id, row in status_rows.items():
        status = row.get("status", "")
        if status in {"manual_accept", "auto_exact"}:
            trusted[clip_id] = normalize_ipa(row.get("chosen_ipa", ""))
        else:
            excluded.append(
                {
                    "source_id": f"phase1a_{clip_id:04d}",
                    "reason": f"phase1a status excluded: {status}",
                }
            )
    for clip_id, row in manual_rows_2.items():
        trusted[clip_id] = normalize_ipa(row.get("recording_ipa", ""))

    trusted_ids = sorted(trusted)
    eval_ids = choose_eval_subset([str(item) for item in trusted_ids], count=max(10, round(len(trusted_ids) * 0.2)), seed=42)

    rows: list[dict[str, Any]] = []
    for clip_id in trusted_ids:
        split_row = split_rows[clip_id]
        audio_path = PHASE1A_CLIPS_DIR / split_row["audio_filename"]
        rows.append(
            {
                "split": "eval" if str(clip_id) in eval_ids else "train",
                "audio": to_windows_path(audio_path),
                "audio_windows": to_windows_path(audio_path),
                "audio_posix": audio_path.resolve().as_posix(),
                "target_ipa": trusted[clip_id],
                "recording_text": "",
                "recording_translation": "",
                "recording_type": "phase1a_trusted",
                "word_title": split_row["audio_filename"],
                "word_slug": "",
                "categories": "quizlet-phase1a",
                "category_names": "Quizlet phase1a",
                "speaker_user_id": "0",
                "source_bundle": "phase1a_reviewed",
                "audio_relpath": f"phase1a/{split_row['audio_filename']}",
                "source_id": f"phase1a_{clip_id:04d}",
                "notes": status_rows.get(clip_id, {}).get("reason", "accepted by user in second import"),
            }
        )
    return rows, excluded


def build_mesele_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tsv_rows = parse_elan_rows(MESELE_TSV)
    included_rows = MESELE_ROWS
    if len(tsv_rows) < len(included_rows) + 1:
        raise RuntimeError("Mesele corrected TSV does not have the expected number of rows")

    rows: list[dict[str, Any]] = []
    excluded = [
        {
            "source_id": "mesele_0015",
            "reason": "final uncertain ending excluded from tier1 set",
        }
    ]
    for item in included_rows:
        tsv_row = tsv_rows[item["row_index"] - 1]
        clip_path = CLIPS_DIR / f"mesele_kurik_lir_{item['row_index']:04d}.wav"
        cut_wav_segment(MESELE_AUDIO, tsv_row["start_sec"], tsv_row["end_sec"], clip_path)
        row = {
            "audio": to_windows_path(clip_path),
            "audio_windows": to_windows_path(clip_path),
            "audio_posix": clip_path.resolve().as_posix(),
            "recording_text": item["orth_full"],
            "recording_translation": "",
            "recording_type": "narrative_segment",
            "word_title": f"Mesele kurık lir row {item['row_index']}",
            "word_slug": "mesele-kurik-lir",
            "categories": "ck-narrative",
            "category_names": "C.K. narrative",
            "speaker_user_id": "0",
            "source_bundle": "ck_narrative_manual",
            "audio_relpath": f"clips/{clip_path.name}",
            "source_id": f"mesele_{item['row_index']:04d}",
            "notes": "trusted story row from corrected TSV and corrected IPA file",
        }
        row["target_ipa"] = normalize_ipa(item["ipa"])
        row["target_text_full"] = item["orth_full"]
        row["target_text"] = simplify_orthography(item["orth_full"])
        rows.append(row)
    return rows, excluded


def build_sabri_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not SABRI_CLIP_1.is_file():
        raise RuntimeError(f"Missing Sabri clip: {SABRI_CLIP_1}")
    included = [
        {
            "audio_path": SABRI_CLIP_1,
            "orth_full": SABRI_ORTH_ROW_2,
            "ipa": SABRI_IPA_ROW_2,
            "source_id": "sabri_0002",
            "notes": "second coarse Sabri clip only; first clip excluded because uncertain clause is embedded inside it",
        }
    ]
    excluded = [
        {
            "source_id": "sabri_0001",
            "reason": "first coarse Sabri clip excluded because uncertain hefil clause is embedded inside the clip",
        }
    ]
    rows: list[dict[str, Any]] = []
    for item in included:
        row = {
            "audio": to_windows_path(item["audio_path"]),
            "audio_windows": to_windows_path(item["audio_path"]),
            "audio_posix": item["audio_path"].resolve().as_posix(),
            "recording_text": item["orth_full"],
            "recording_translation": "",
            "recording_type": "narrative_segment",
            "word_title": "Sabri neweş row 2",
            "word_slug": "sabri-newes",
            "categories": "ck-narrative",
            "category_names": "C.K. narrative",
            "speaker_user_id": "0",
            "source_bundle": "ck_narrative_manual",
            "audio_relpath": f"audio_corpus/training_data/clips/{item['audio_path'].name}",
            "source_id": item["source_id"],
            "notes": item["notes"],
            "target_ipa": normalize_ipa(item["ipa"]),
            "target_text_full": item["orth_full"],
            "target_text": simplify_orthography(item["orth_full"]),
        }
        rows.append(row)
    return rows, excluded


def build_exploring_orth_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_rows = read_tsv(EXPLORING_MANIFEST)
    exploring_rows = [
        row
        for row in source_rows
        if PureWindowsPath(row["audio"]).name.startswith("1_exploring_a_mine_")
    ]
    eval_ids = choose_eval_subset(
        [PureWindowsPath(row["audio"]).stem for row in exploring_rows],
        count=8,
        seed=42,
    )
    for row in exploring_rows:
        audio_path = win_to_posix(row["audio"])
        source_id = PureWindowsPath(row["audio"]).stem
        full_text = row["text"].strip()
        rows.append(
            {
                "split": "eval" if source_id in eval_ids else "train",
                "audio": row["audio"],
                "audio_windows": row["audio"],
                "audio_posix": audio_path.resolve().as_posix(),
                "target_text": simplify_orthography(full_text),
                "target_text_full": full_text,
                "recording_text": full_text,
                "recording_translation": "",
                "recording_type": "narrative_segment",
                "word_title": source_id,
                "word_slug": "exploring-a-mine",
                "speaker_user_id": "0",
                "source_bundle": "audio_corpus_training_manifest",
                "audio_relpath": f"audio_corpus/training_data/clips/{PureWindowsPath(row['audio']).name}",
                "source_id": source_id,
                "notes": "existing C.K. orthography training clip",
            }
        )
    return rows


def make_ipa_manifest_rows(
    phase1a_rows: list[dict[str, Any]],
    mesele_rows: list[dict[str, Any]],
    sabri_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in phase1a_rows:
        item = dict(row)
        item.setdefault("split", row["split"])
        rows.append(item)
    for source_row in mesele_rows:
        item = {
            key: source_row.get(key, "")
            for key in IPA_MANIFEST_HEADERS
        }
        item["split"] = "train"
        rows.append(item)
    for source_row in sabri_rows:
        item = {
            key: source_row.get(key, "")
            for key in IPA_MANIFEST_HEADERS
        }
        item["split"] = "train"
        rows.append(item)
    return rows


def make_text_manifest_rows(
    exploring_rows: list[dict[str, Any]],
    mesele_rows: list[dict[str, Any]],
    sabri_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = list(exploring_rows)
    mesele_eval = choose_eval_subset(
        [f"mesele_{item['row_index']:04d}" for item in MESELE_ROWS],
        count=2,
        seed=42,
    )
    for source_row in mesele_rows:
        item = {
            key: source_row.get(key, "")
            for key in TEXT_MANIFEST_HEADERS
        }
        item["split"] = "eval" if source_row["source_id"] in mesele_eval else "train"
        rows.append(item)
    for source_row in sabri_rows:
        item = {
            key: source_row.get(key, "")
            for key in TEXT_MANIFEST_HEADERS
        }
        item["split"] = "train"
        rows.append(item)
    return rows


def build_ak_anchor_rows(ck_train_count: int) -> list[dict[str, Any]]:
    anchor_source = read_tsv(AK_ANCHOR_MANIFEST)
    train_rows = [row for row in anchor_source if row.get("split") == "train"]
    anchor_count = min(len(train_rows), max(48, min(96, ck_train_count)))
    selected = sorted(
        train_rows,
        key=lambda row: hash_sort_key(row.get("audio_relpath") or row.get("audio_posix") or row.get("audio") or "", 42),
    )[:anchor_count]
    rows: list[dict[str, Any]] = []
    for row in selected:
        item = {key: row.get(key, "") for key in IPA_MANIFEST_HEADERS}
        item["split"] = "train"
        item["source_id"] = f"ak_anchor::{row.get('audio_relpath') or Path(row.get('audio', '')).name}"
        item["notes"] = "small A.K. anchor subset for anti-forgetting mix"
        rows.append(item)
    return rows


def build_ak_anchor_text_rows(ck_train_count: int) -> list[dict[str, Any]]:
    anchor_source = read_tsv(AK_TEXT_ANCHOR_MANIFEST)
    train_rows = [row for row in anchor_source if row.get("split") == "train"]
    anchor_count = min(len(train_rows), max(48, min(96, ck_train_count)))
    selected = sorted(
        train_rows,
        key=lambda row: hash_sort_key(row.get("audio_relpath") or row.get("audio_posix") or row.get("audio") or "", 42),
    )[:anchor_count]
    rows: list[dict[str, Any]] = []
    for row in selected:
        item = {key: row.get(key, "") for key in TEXT_MANIFEST_HEADERS}
        item["split"] = "train"
        item["target_text_full"] = row.get("target_text_full") or row.get("recording_text") or row.get("target_text") or ""
        item["source_id"] = f"ak_anchor::{row.get('audio_relpath') or Path(str(row.get('audio') or '')).name}"
        item["notes"] = "small A.K. anchor subset for anti-forgetting mix"
        rows.append(item)
    return rows


def duration_from_row(row: dict[str, Any]) -> float:
    audio_path = Path(row["audio_posix"])
    if audio_path.suffix.lower() != ".wav":
        return 0.0
    return wav_duration(audio_path)


def write_launchers() -> None:
    ipa_base = stt_model_path(SCRIPT_DIR, "alternatives", "mms-1b-all-ipa-warmstart-v003-20260407-124628")
    orth_base = stt_model_path(
        SCRIPT_DIR,
        "alternatives",
        "mms-1b-all-orthography-zza-simple-v001-20260409-071931",
        "checkpoint-1282",
    )
    ipa_ck_only_output = stt_model_path(SCRIPT_DIR, "alternatives", "mms-1b-all-ipa-ck-tier1-v001-20260410")
    ipa_anchor_output = stt_model_path(
        SCRIPT_DIR,
        "alternatives",
        "mms-1b-all-ipa-ck-tier1-ak-anchor-v001-20260410",
    )
    text_ck_only_output = stt_model_path(
        SCRIPT_DIR,
        "alternatives",
        "mms-1b-all-orthography-ck-tier1-v001-20260410",
    )
    text_anchor_output = stt_model_path(
        SCRIPT_DIR,
        "alternatives",
        "mms-1b-all-orthography-ck-tier1-ak-anchor-v001-20260411",
    )

    ipa_ck_only_manifest = OUTPUT_DIR / "manifest_ipa_ck_only.tsv"
    ipa_anchor_manifest = OUTPUT_DIR / "manifest_ipa_ck_plus_ak_anchor.tsv"
    text_ck_only_manifest = OUTPUT_DIR / "manifest_text_ck_only.tsv"
    text_anchor_manifest = OUTPUT_DIR / "manifest_text_ck_plus_ak_anchor.tsv"

    files = {
        "train_ck_mms_ipa_ck_only_wsl.sh": f"""#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ipa.py" \\
  --manifest "{ipa_ck_only_manifest.as_posix()}" \\
  --base-model "{ipa_base.as_posix()}" \\
  --output-dir "{ipa_ck_only_output.as_posix()}" \\
  --run-name "mms-1b-all-ipa-ck-tier1-v001-20260410" \\
  --epochs 12 \\
  --train-batch-size 1 \\
  --eval-batch-size 1 \\
  --grad-accum 4 \\
  --learning-rate 1e-5 \\
  --warmup-ratio 0.1 \\
  --weight-decay 0.01 \\
  --patience 3 \\
  --dataloader-workers 0 \\
  --freeze-feature-encoder \\
  --gradient-checkpointing \\
  --bf16
""",
        "train_ck_mms_ipa_ck_plus_ak_anchor_wsl.sh": f"""#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ipa.py" \\
  --manifest "{ipa_anchor_manifest.as_posix()}" \\
  --base-model "{ipa_base.as_posix()}" \\
  --output-dir "{ipa_anchor_output.as_posix()}" \\
  --run-name "mms-1b-all-ipa-ck-tier1-ak-anchor-v001-20260410" \\
  --epochs 12 \\
  --train-batch-size 1 \\
  --eval-batch-size 1 \\
  --grad-accum 4 \\
  --learning-rate 1e-5 \\
  --warmup-ratio 0.1 \\
  --weight-decay 0.01 \\
  --patience 3 \\
  --dataloader-workers 0 \\
  --freeze-feature-encoder \\
  --gradient-checkpointing \\
  --bf16
""",
        "train_ck_mms_orth_ck_only_wsl.sh": f"""#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ctc_text.py" \\
  --manifest "{text_ck_only_manifest.as_posix()}" \\
  --base-model "{orth_base.as_posix()}" \\
  --output-dir "{text_ck_only_output.as_posix()}" \\
  --run-name "mms-1b-all-orthography-ck-tier1-v001-20260410" \\
  --epochs 10 \\
  --train-batch-size 1 \\
  --eval-batch-size 1 \\
  --grad-accum 4 \\
  --learning-rate 1e-5 \\
  --warmup-ratio 0.1 \\
  --weight-decay 0.01 \\
  --patience 3 \\
  --dataloader-workers 0 \\
  --freeze-feature-encoder \\
  --gradient-checkpointing \\
  --bf16
""",
        "train_ck_mms_orth_ck_plus_ak_anchor_wsl.sh": f"""#!/usr/bin/env bash
set -euo pipefail

source "$LINUX_STT_ENV/.venv/bin/activate"

python3 "$Z_WORKSPACE/Speech to Text IPA/train_wav2vec2_ctc_text.py" \\
  --manifest "{text_anchor_manifest.as_posix()}" \\
  --base-model "{orth_base.as_posix()}" \\
  --output-dir "{text_anchor_output.as_posix()}" \\
  --run-name "mms-1b-all-orthography-ck-tier1-ak-anchor-v001-20260411" \\
  --epochs 10 \\
  --train-batch-size 1 \\
  --eval-batch-size 1 \\
  --grad-accum 4 \\
  --learning-rate 1e-5 \\
  --warmup-ratio 0.1 \\
  --weight-decay 0.01 \\
  --patience 3 \\
  --dataloader-workers 0 \\
  --freeze-feature-encoder \\
  --gradient-checkpointing \\
  --bf16
""",
    }

    for name, content in files.items():
        path = OUTPUT_DIR / name
        path.write_text(content, encoding="utf-8")

    cmd_map = {
        "train_ck_mms_ipa_ck_only.cmd": "train_ck_mms_ipa_ck_only_wsl.sh",
        "train_ck_mms_ipa_ck_plus_ak_anchor.cmd": "train_ck_mms_ipa_ck_plus_ak_anchor_wsl.sh",
        "train_ck_mms_orth_ck_only.cmd": "train_ck_mms_orth_ck_only_wsl.sh",
        "train_ck_mms_orth_ck_plus_ak_anchor.cmd": "train_ck_mms_orth_ck_plus_ak_anchor_wsl.sh",
    }
    for cmd_name, sh_name in cmd_map.items():
        cmd_path = OUTPUT_DIR / cmd_name
        cmd_path.write_text(
            "@echo off\n"
            f"wsl bash -lc '\"{(OUTPUT_DIR / sh_name).as_posix()}\"'\n",
            encoding="utf-8",
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    phase1a_rows, phase1a_excluded = build_phase1a_trusted_rows()
    mesele_rows, mesele_excluded = build_mesele_rows()
    sabri_rows, sabri_excluded = build_sabri_rows()
    exploring_rows = build_exploring_orth_rows()

    ipa_ck_rows = make_ipa_manifest_rows(phase1a_rows, mesele_rows, sabri_rows)
    ck_train_count = sum(1 for row in ipa_ck_rows if row["split"] == "train")
    ak_anchor_rows = build_ak_anchor_rows(ck_train_count)
    ipa_anchor_rows = list(ipa_ck_rows) + ak_anchor_rows
    text_rows = make_text_manifest_rows(exploring_rows, mesele_rows, sabri_rows)
    text_ck_train_count = sum(1 for row in text_rows if row["split"] == "train")
    ak_anchor_text_rows = build_ak_anchor_text_rows(text_ck_train_count)
    text_anchor_rows = list(text_rows) + ak_anchor_text_rows

    write_tsv(OUTPUT_DIR / "manifest_ipa_ck_only.tsv", ipa_ck_rows, IPA_MANIFEST_HEADERS)
    write_tsv(OUTPUT_DIR / "manifest_ipa_ck_plus_ak_anchor.tsv", ipa_anchor_rows, IPA_MANIFEST_HEADERS)
    write_tsv(OUTPUT_DIR / "manifest_text_ck_only.tsv", text_rows, TEXT_MANIFEST_HEADERS)
    write_tsv(OUTPUT_DIR / "manifest_text_ck_plus_ak_anchor.tsv", text_anchor_rows, TEXT_MANIFEST_HEADERS)

    excluded_rows = phase1a_excluded + mesele_excluded + sabri_excluded
    write_jsonl(OUTPUT_DIR / "excluded_items.jsonl", excluded_rows)
    (OUTPUT_DIR / "README.md").write_text(README_TEXT, encoding="utf-8")
    write_launchers()

    summary = {
        "ipa_ck_only": {
            "rows": len(ipa_ck_rows),
            "train_rows": sum(1 for row in ipa_ck_rows if row["split"] == "train"),
            "eval_rows": sum(1 for row in ipa_ck_rows if row["split"] == "eval"),
            "train_seconds_wav_only": round(sum(duration_from_row(row) for row in ipa_ck_rows if row["split"] == "train"), 3),
            "eval_seconds_wav_only": round(sum(duration_from_row(row) for row in ipa_ck_rows if row["split"] == "eval"), 3),
        },
        "ipa_ck_plus_ak_anchor": {
            "rows": len(ipa_anchor_rows),
            "train_rows": sum(1 for row in ipa_anchor_rows if row["split"] == "train"),
            "eval_rows": sum(1 for row in ipa_anchor_rows if row["split"] == "eval"),
            "ak_anchor_rows": len(ak_anchor_rows),
        },
        "text_ck_only": {
            "rows": len(text_rows),
            "train_rows": sum(1 for row in text_rows if row["split"] == "train"),
            "eval_rows": sum(1 for row in text_rows if row["split"] == "eval"),
            "train_seconds_wav_only": round(sum(duration_from_row(row) for row in text_rows if row["split"] == "train"), 3),
            "eval_seconds_wav_only": round(sum(duration_from_row(row) for row in text_rows if row["split"] == "eval"), 3),
        },
        "text_ck_plus_ak_anchor": {
            "rows": len(text_anchor_rows),
            "train_rows": sum(1 for row in text_anchor_rows if row["split"] == "train"),
            "eval_rows": sum(1 for row in text_anchor_rows if row["split"] == "eval"),
            "ak_anchor_rows": len(ak_anchor_text_rows),
        },
        "phase1a_trusted_ids": [row["source_id"] for row in phase1a_rows],
        "excluded_count": len(excluded_rows),
        "output_dir_windows": to_windows_path(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
