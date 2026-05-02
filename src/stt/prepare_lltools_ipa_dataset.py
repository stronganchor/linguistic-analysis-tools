import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


MANIFEST_HEADERS = [
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
]

STRESS_MARK_PATTERN = re.compile(r"[ˈˌ'’]+")
DENTAL_DIACRITIC = "\u032A"


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_input_dir = script_dir / "zip files from zazacaogren"
    if not default_input_dir.is_dir():
        default_input_dir = script_dir
    default_blacklist = script_dir / "training_blacklists" / "genc-palu-manual-training-blacklist.json"
    parser = argparse.ArgumentParser(
        description="Extract LL Tools bundle zips and build an audio->IPA manifest."
    )
    parser.add_argument(
        "--input-dir",
        default=str(default_input_dir),
        help="Folder containing LL Tools full-bundle zip exports.",
    )
    parser.add_argument(
        "--extract-dir",
        default=str(script_dir / "extracted_bundles"),
        help="Where extracted bundle contents should be stored.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "training_data"),
        help="Where the manifest and summary files should be written.",
    )
    parser.add_argument(
        "--zip",
        action="append",
        dest="zip_paths",
        default=[],
        help="Specific bundle zip(s) to use. Defaults to all .zip files in --input-dir.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Optional speaker_user_id filter. Use 0 to include all speakers.",
    )
    parser.add_argument(
        "--recording-type",
        action="append",
        dest="recording_types",
        default=[],
        help="Optional recording type filter. Repeat for multiple values.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of words to place in eval split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for deterministic word-level train/eval splits.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract bundle zips even if an extracted folder already exists.",
    )
    parser.add_argument(
        "--blacklist",
        default=str(default_blacklist) if default_blacklist.is_file() else "",
        help=(
            "Optional JSON blacklist of recordings or lexical items to exclude from training. "
            "If omitted, no blacklist is applied."
        ),
    )
    return parser.parse_args()


def strip_dental_stop_marks(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    return text.replace(f"t{DENTAL_DIACRITIC}", "t").replace(f"d{DENTAL_DIACRITIC}", "d")


def redentalize_stop_consonants(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    pieces: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in {"t", "d"}:
            next_char = text[index + 1] if index + 1 < len(text) else ""
            pieces.append(char)
            if next_char != DENTAL_DIACRITIC:
                pieces.append(DENTAL_DIACRITIC)
        else:
            pieces.append(char)
        index += 1
    return unicodedata.normalize("NFC", "".join(pieces))


def normalize_ipa_text(
    value: str,
    *,
    strip_dental_stops: bool = False,
    redentalize_stops: bool = False,
) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = STRESS_MARK_PATTERN.sub("", text)
    if strip_dental_stops:
        text = strip_dental_stop_marks(text)
    if redentalize_stops:
        text = redentalize_stop_consonants(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_plain_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_windows_path(path: Path) -> str:
    posix = path.resolve().as_posix()
    parts = posix.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper() + ":"
        remainder = "\\".join(parts[3:])
        return drive + ("\\" + remainder if remainder else "")
    return str(path.resolve())


def list_zip_paths(input_dir: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(p).expanduser().resolve() for p in explicit]
    return sorted(input_dir.glob("*.zip"))


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"Unsafe zip entry path: {member.filename}")
        archive.extractall(destination)


def extract_bundle(zip_path: Path, extract_root: Path, force: bool) -> Path:
    destination = extract_root / zip_path.stem
    if destination.exists() and force:
        shutil.rmtree(destination)
    if not destination.exists():
        destination.mkdir(parents=True, exist_ok=True)
        safe_extract(zip_path, destination)
    return destination


def resolve_bundle_content_root(extracted_bundle_dir: Path) -> Path:
    candidate_files = ("data.json", "metadata.csv", "metadata.jsonl")
    for filename in candidate_files:
        if (extracted_bundle_dir / filename).is_file():
            return extracted_bundle_dir

    matches: list[Path] = []
    for filename in candidate_files:
        matches.extend(extracted_bundle_dir.rglob(filename))

    if not matches:
        return extracted_bundle_dir

    matches = sorted({match.resolve() for match in matches})
    return matches[0].parent


def hash_bucket(key: str, seed: int) -> float:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def choose_split(word_key: str, validation_fraction: float, seed: int) -> str:
    return "eval" if hash_bucket(word_key, seed) < validation_fraction else "train"


def build_category_name_lookup(bundle_data: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for category in bundle_data.get("categories") or []:
        slug = str(category.get("slug") or "").strip()
        name = str(category.get("name") or "").strip()
        if slug:
            lookup[slug] = name
    return lookup


def first_meta_value(meta: dict, key: str) -> str:
    values = meta.get(key) or []
    if not values:
        return ""
    return str(values[0] or "")


def load_training_blacklist(path: Path | None) -> dict[str, object]:
    empty = {
        "recording_ids": set(),
        "word_ids": set(),
        "word_titles": set(),
        "path": "",
        "matched": [],
    }
    if path is None:
        return empty
    if not path.is_file():
        raise SystemExit(f"Blacklist file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    recording_ids: set[str] = set()
    word_ids: set[str] = set()
    word_titles: set[str] = set()
    for entry in data.get("entries") or []:
        if entry.get("exclude_from_training", True) is False:
            continue
        scope = normalize_plain_text(str(entry.get("scope") or ""))
        for value in entry.get("recording_ids") or []:
            text = normalize_plain_text(str(value))
            if text:
                recording_ids.add(text)
        for value in entry.get("word_ids") or []:
            text = normalize_plain_text(str(value))
            if text:
                word_ids.add(text)
        title = normalize_plain_text(str(entry.get("word_title") or ""))
        if title and scope == "word_title":
            word_titles.add(title.casefold())

    return {
        "recording_ids": recording_ids,
        "word_ids": word_ids,
        "word_titles": word_titles,
        "path": str(path),
        "matched": [],
    }


def blacklist_match(
    blacklist: dict[str, object],
    *,
    recording_id: str,
    word_id: str,
    word_title: str,
) -> str:
    recording_ids = blacklist.get("recording_ids") or set()
    word_ids = blacklist.get("word_ids") or set()
    word_titles = blacklist.get("word_titles") or set()
    if recording_id and recording_id in recording_ids:
        return "recording_id"
    if word_id and word_id in word_ids:
        return "word_id"
    if word_title and word_title.casefold() in word_titles:
        return "word_title"
    return ""


def iter_manifest_rows(
    bundle_name: str,
    extracted_dir: Path,
    bundle_data: dict,
    speaker_id: int,
    allowed_recording_types: set[str],
    validation_fraction: float,
    seed: int,
    blacklist: dict[str, object],
) -> list[dict[str, str]]:
    category_name_lookup = build_category_name_lookup(bundle_data)
    rows: list[dict[str, str]] = []

    for word in bundle_data.get("words") or []:
        word_id = normalize_plain_text(str(word.get("origin_id") or ""))
        word_slug = normalize_plain_text(str(word.get("slug") or ""))
        word_title = normalize_plain_text(str(word.get("title") or ""))
        categories = [normalize_plain_text(str(x)) for x in (word.get("categories") or []) if str(x).strip()]
        category_names = [category_name_lookup.get(slug, slug) for slug in categories]
        word_key = word_slug or word_title
        if not word_key:
            continue

        split = choose_split(word_key, validation_fraction, seed)

        for entry in word.get("audio_entries") or []:
            recording_id = normalize_plain_text(str(entry.get("origin_id") or ""))
            match_reason = blacklist_match(
                blacklist,
                recording_id=recording_id,
                word_id=word_id,
                word_title=word_title,
            )
            if match_reason:
                blacklist["matched"].append(
                    {
                        "bundle_name": bundle_name,
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": match_reason,
                    }
                )
                continue
            meta = entry.get("meta") or {}
            recording_type_list = [
                normalize_plain_text(str(x))
                for x in (entry.get("recording_types") or [])
                if str(x).strip()
            ]
            if not recording_type_list:
                recording_type_list = ["unknown"]
            primary_recording_type = recording_type_list[0]
            if allowed_recording_types and primary_recording_type not in allowed_recording_types:
                continue

            entry_speaker_raw = normalize_plain_text(first_meta_value(meta, "speaker_user_id"))
            entry_speaker_id = int(entry_speaker_raw) if entry_speaker_raw.isdigit() else 0
            if speaker_id > 0 and entry_speaker_id != speaker_id:
                continue

            target_ipa = normalize_ipa_text(first_meta_value(meta, "recording_ipa"))
            if not target_ipa:
                continue

            audio_file = entry.get("audio_file") or {}
            audio_relpath = normalize_plain_text(str(audio_file.get("file") or ""))
            if not audio_relpath:
                continue

            audio_path = extracted_dir / audio_relpath
            if not audio_path.is_file():
                continue

            rows.append(
                {
                    "split": split,
                    "audio": to_windows_path(audio_path),
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "target_ipa": target_ipa,
                    "recording_text": normalize_plain_text(first_meta_value(meta, "recording_text")),
                    "recording_translation": normalize_plain_text(first_meta_value(meta, "recording_translation")),
                    "recording_type": primary_recording_type,
                    "word_title": word_title,
                    "word_slug": word_slug,
                    "categories": "|".join(categories),
                    "category_names": "|".join(category_names),
                    "speaker_user_id": str(entry_speaker_id),
                    "source_bundle": bundle_name,
                    "audio_relpath": audio_relpath,
                }
            )

    return rows


def split_recording_types(value: str) -> list[str]:
    text = normalize_plain_text(value)
    if not text:
        return []
    return [item for item in re.split(r"[|,]", text) if normalize_plain_text(item)]


def iter_manifest_rows_from_metadata_csv(
    bundle_name: str,
    extracted_dir: Path,
    metadata_csv_path: Path,
    speaker_id: int,
    allowed_recording_types: set[str],
    validation_fraction: float,
    seed: int,
    blacklist: dict[str, object],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with metadata_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for entry in reader:
            text_field = normalize_plain_text(str(entry.get("text_field") or ""))
            target_ipa = normalize_ipa_text(
                str(entry.get("recording_ipa") or entry.get("text") or "")
            )
            if text_field and text_field != "recording_ipa":
                continue
            if not target_ipa:
                continue

            audio_relpath = normalize_plain_text(str(entry.get("audio") or ""))
            if not audio_relpath:
                continue
            audio_path = extracted_dir / audio_relpath
            if not audio_path.is_file():
                continue

            recording_type_list = [
                normalize_plain_text(item)
                for item in split_recording_types(str(entry.get("recording_types") or ""))
                if normalize_plain_text(item)
            ] or ["unknown"]
            primary_recording_type = recording_type_list[0]
            if allowed_recording_types and primary_recording_type not in allowed_recording_types:
                continue

            entry_speaker_raw = normalize_plain_text(str(entry.get("speaker_user_id") or ""))
            entry_speaker_id = int(entry_speaker_raw) if entry_speaker_raw.isdigit() else 0
            if speaker_id > 0 and entry_speaker_id != speaker_id:
                continue

            word_title = normalize_plain_text(str(entry.get("word_title") or ""))
            word_id = normalize_plain_text(str(entry.get("word_id") or ""))
            recording_id = normalize_plain_text(str(entry.get("recording_id") or ""))
            match_reason = blacklist_match(
                blacklist,
                recording_id=recording_id,
                word_id=word_id,
                word_title=word_title,
            )
            if match_reason:
                blacklist["matched"].append(
                    {
                        "bundle_name": bundle_name,
                        "recording_id": recording_id,
                        "word_id": word_id,
                        "word_title": word_title,
                        "reason": match_reason,
                    }
                )
                continue
            word_slug = normalize_plain_text(str(entry.get("word_slug") or ""))
            word_key = (
                word_slug
                or word_id
                or word_title
            )
            if not word_key:
                continue

            split = choose_split(word_key, validation_fraction, seed)

            rows.append(
                {
                    "split": split,
                    "audio": to_windows_path(audio_path),
                    "audio_windows": to_windows_path(audio_path),
                    "audio_posix": audio_path.resolve().as_posix(),
                    "target_ipa": target_ipa,
                    "recording_text": normalize_plain_text(str(entry.get("recording_text") or "")),
                    "recording_translation": normalize_plain_text(
                        str(entry.get("recording_translation") or "")
                    ),
                    "recording_type": primary_recording_type,
                    "word_title": word_title,
                    "word_slug": word_slug,
                    "categories": "",
                    "category_names": "",
                    "speaker_user_id": str(entry_speaker_id),
                    "source_bundle": bundle_name,
                    "audio_relpath": audio_relpath,
                }
            )

    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_HEADERS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, str]], bundles: list[str], blacklist: dict[str, object]) -> dict:
    by_split = Counter()
    by_type = Counter()
    by_category = Counter()
    by_bundle = Counter()
    unique_words = set()

    for row in rows:
        by_split[row["split"]] += 1
        by_type[row["recording_type"]] += 1
        for name in [x for x in row["category_names"].split("|") if x]:
            by_category[name] += 1
        by_bundle[row["source_bundle"]] += 1
        unique_words.add(row["word_slug"] or row["word_title"])

    return {
        "bundle_files": bundles,
        "total_examples": len(rows),
        "unique_words": len(unique_words),
        "by_split": dict(by_split),
        "by_recording_type": dict(by_type),
        "by_category": dict(by_category),
        "by_bundle": dict(by_bundle),
        "training_blacklist_path": blacklist.get("path") or "",
        "blacklist_excluded_examples": len(blacklist.get("matched") or []),
    }


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    extract_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    blacklist_path = Path(args.blacklist).expanduser().resolve() if args.blacklist else None
    blacklist = load_training_blacklist(blacklist_path)

    zip_paths = list_zip_paths(input_dir, args.zip_paths)
    if not zip_paths:
        raise SystemExit(f"No bundle zip files found in {input_dir}")

    allowed_recording_types = {normalize_plain_text(x) for x in args.recording_types if normalize_plain_text(x)}

    all_rows: list[dict[str, str]] = []
    used_bundles: list[str] = []

    for zip_path in zip_paths:
        if not zip_path.is_file():
            raise SystemExit(f"Missing zip file: {zip_path}")

        extracted_bundle_dir = extract_bundle(zip_path, extract_dir, args.force_extract)
        bundle_content_dir = resolve_bundle_content_root(extracted_bundle_dir)
        data_json_path = bundle_content_dir / "data.json"
        metadata_csv_path = bundle_content_dir / "metadata.csv"

        if data_json_path.is_file():
            with data_json_path.open("r", encoding="utf-8") as handle:
                bundle_data = json.load(handle)

            rows = iter_manifest_rows(
                bundle_name=zip_path.name,
                extracted_dir=bundle_content_dir,
                bundle_data=bundle_data,
                speaker_id=args.speaker_id,
                allowed_recording_types=allowed_recording_types,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                blacklist=blacklist,
            )
        elif metadata_csv_path.is_file():
            rows = iter_manifest_rows_from_metadata_csv(
                bundle_name=zip_path.name,
                extracted_dir=bundle_content_dir,
                metadata_csv_path=metadata_csv_path,
                speaker_id=args.speaker_id,
                allowed_recording_types=allowed_recording_types,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                blacklist=blacklist,
            )
        else:
            raise SystemExit(f"Bundle missing data.json or metadata.csv: {zip_path}")
        all_rows.extend(rows)
        used_bundles.append(zip_path.name)

    if not all_rows:
        raise SystemExit("No audio+IPA examples were found after filtering.")

    all_rows.sort(key=lambda row: (row["split"], row["word_slug"], row["recording_type"], row["audio"]))

    manifest_path = output_dir / "manifest_ipa.tsv"
    summary_path = output_dir / "dataset_summary.json"

    write_manifest(manifest_path, all_rows)

    summary = build_summary(all_rows, used_bundles, blacklist)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
