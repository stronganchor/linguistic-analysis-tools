import argparse
import shutil
from pathlib import Path

from model_paths import resolve_stt_models_dir


def default_model_source(root_dir: Path) -> Path:
    models_dir = resolve_stt_models_dir(root_dir)
    discovered_models: list[Path] = []

    if models_dir.is_dir():
        for candidate in models_dir.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name.lower().startswith("checkpoint-"):
                continue
            if candidate.name.lower() == "mobile":
                continue
            if not (candidate / "config.json").is_file():
                continue
            discovered_models.append(candidate)

    if not discovered_models:
        raise SystemExit(f"No non-mobile model directories with config.json were found in {models_dir}.")

    return max(
        discovered_models,
        key=lambda path: (
            getattr(path.stat(), "st_ctime", 0),
            path.stat().st_mtime,
            path.name.lower(),
        ),
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Copy a selected STT model into the Runpod build context."
    )
    parser.add_argument(
        "--model-source",
        default="",
        help="Optional path to a specific model directory. Defaults to the newest non-mobile model in models/.",
    )
    parser.add_argument(
        "--target-dir",
        default=str(script_dir / "runpod-loadbalancer" / "model"),
        help="Directory that should receive the copied model files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    source_dir = Path(args.model_source).expanduser() if args.model_source else default_model_source(root_dir)
    if not source_dir.is_absolute():
        source_dir = (root_dir / source_dir).resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {source_dir}")
    if not (source_dir / "config.json").is_file():
        raise SystemExit(f"Model directory is missing config.json: {source_dir}")

    target_dir = Path(args.target_dir).expanduser()
    if not target_dir.is_absolute():
        target_dir = (root_dir / target_dir).resolve()

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for entry in source_dir.iterdir():
        if entry.name == "__pycache__":
            continue
        if entry.name.lower().startswith("checkpoint-"):
            continue
        destination = target_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)

    metadata_path = target_dir.parent / "model-source.txt"
    metadata_path.write_text(str(source_dir) + "\n", encoding="utf-8")

    print(f"Copied Runpod model from: {source_dir}")
    print(f"Runpod build model dir:   {target_dir}")
    print(f"Source metadata written:  {metadata_path}")


if __name__ == "__main__":
    main()
