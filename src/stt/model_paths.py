from __future__ import annotations

import os
from pathlib import Path


ENV_VAR = "Z_STT_MODELS_DIR"


def _infer_user_root(script_dir: Path) -> Path:
    resolved = script_dir.resolve()
    parts = resolved.parts
    if len(parts) >= 5 and parts[0] == "/" and parts[1] == "mnt" and parts[3].lower() == "users":
        return Path(*parts[:5])
    return Path.home()


def default_stt_models_dir(script_dir: Path) -> Path:
    return _infer_user_root(script_dir) / "Models" / "Language" / "Z" / "speech_to_text_ipa"


def legacy_stt_models_dir(script_dir: Path) -> Path:
    return (script_dir / "models").resolve()


def resolve_stt_models_dir(script_dir: Path) -> Path:
    raw_override = os.environ.get(ENV_VAR, "").strip()
    if raw_override:
        return Path(raw_override).expanduser().resolve()

    default_dir = default_stt_models_dir(script_dir)
    if default_dir.exists():
        return default_dir.resolve()

    legacy_dir = legacy_stt_models_dir(script_dir)
    if legacy_dir.exists():
        return legacy_dir

    return default_dir


def stt_model_path(script_dir: Path, *parts: str) -> Path:
    return resolve_stt_models_dir(script_dir).joinpath(*parts)
