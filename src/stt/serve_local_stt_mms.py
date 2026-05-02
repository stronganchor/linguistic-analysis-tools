from __future__ import annotations

import argparse
import errno
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from email.parser import BytesParser
from email.policy import default as email_policy_default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEPENDENCY_IMPORT_ERROR = None
try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    DEPENDENCY_IMPORT_ERROR = exc
    torch = None

try:
    import torchaudio
except ModuleNotFoundError:  # pragma: no cover - depends on runtime env
    torchaudio = None

try:
    from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    if DEPENDENCY_IMPORT_ERROR is None:
        DEPENDENCY_IMPORT_ERROR = exc

import librosa

from model_paths import stt_model_path
from prepare_lltools_ipa_dataset import normalize_ipa_text


TARGET_SR = 16000
PYCTC_UNK_LABEL = "⁇"
PRIVATE_USE_START = 0xE000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def default_model_source() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-warmstart-v003-20260407-124628"),
        stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-warmstart-v002-20260406"),
        stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-v001-checkpoint235-export"),
        stt_model_path(script_dir, "alternatives", "mms-1b-all-ipa-v001"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "config.json").is_file():
            return str(candidate)
    return "facebook/mms-1b-all"


def choose_device(value: str) -> str:
    requested = (value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def choose_torch_dtype(device_name: str, precision: str):
    requested = (precision or "").strip().lower()
    if device_name != "cuda":
        return None
    if requested in ("", "auto", "float16", "fp16", "half"):
        return torch.float16
    if requested in ("bfloat16", "bf16"):
        return torch.bfloat16
    if requested in ("float32", "fp32"):
        return torch.float32
    raise SystemExit(f"Unsupported precision: {precision}")


def resolve_feature_dtype(model) -> torch.dtype | None:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return None


def parse_multipart_form_data(content_type: str, body: bytes) -> dict[str, dict[str, object]]:
    message = BytesParser(policy=email_policy_default).parsebytes(
        ("Content-Type: " + content_type + "\r\nMIME-Version: 1.0\r\n\r\n").encode("utf-8")
        + body
    )
    if not message.is_multipart():
        return {}

    fields: dict[str, dict[str, object]] = {}
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        field_name = part.get_param("name", header="content-disposition")
        if not field_name:
            continue

        fields[str(field_name)] = {
            "filename": os.path.basename((part.get_filename() or "").strip()),
            "content": part.get_payload(decode=True) or b"",
            "content_type": part.get_content_type(),
        }

    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a local HTTP endpoint that transcribes uploaded audio to IPA with MMS."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8769)
    parser.add_argument("--model-source", default=default_model_source())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=["auto", "greedy", "beam"],
        help="CTC decoding strategy. Auto uses lltools_ctc_config.json when present.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=64,
        help="Beam width used with pyctcdecode beam search.",
    )
    parser.add_argument(
        "--allowed-origin",
        default="*",
        help="Allowed browser origin. Use * to echo the request origin.",
    )
    parser.add_argument(
        "--api-token",
        default=os.environ.get("LL_TOOLS_STT_API_TOKEN", ""),
        help="Optional bearer token required for POST /transcribe requests.",
    )
    return parser.parse_args()


def load_local_model_config(model_dir: Path) -> dict:
    config_path = model_dir / "lltools_ctc_config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def placeholder_label(token_id: int) -> str:
    return chr(PRIVATE_USE_START + token_id)


def map_ctc_label(token: str, token_id: int, word_delimiter_token: str) -> str:
    if token == "[PAD]":
        return ""
    if token == "[UNK]":
        return PYCTC_UNK_LABEL
    if token == word_delimiter_token:
        return " "
    if token.startswith("<") and token.endswith(">"):
        return placeholder_label(token_id)
    return token


def build_pyctc_labels(
    vocab: dict[str, int],
    word_delimiter_token: str,
    vocab_size: int | None = None,
    added_tokens_decoder: dict | None = None,
) -> list[str]:
    max_vocab_id = max(vocab.values()) if vocab else -1
    max_added_id = -1
    if added_tokens_decoder:
        for token_id in added_tokens_decoder.keys():
            try:
                max_added_id = max(max_added_id, int(token_id))
            except (TypeError, ValueError):
                continue

    label_count = max(max_vocab_id, max_added_id, (vocab_size or 0) - 1) + 1
    labels = [placeholder_label(token_id) for token_id in range(label_count)]

    for token, token_id in vocab.items():
        labels[token_id] = map_ctc_label(token, token_id, word_delimiter_token)

    if added_tokens_decoder:
        for token_id_raw, token_meta in added_tokens_decoder.items():
            try:
                token_id = int(token_id_raw)
            except (TypeError, ValueError):
                continue
            if token_id < 0 or token_id >= len(labels):
                continue
            if any(existing_id == token_id for existing_id in vocab.values()):
                continue
            token = str((token_meta or {}).get("content") or "")
            labels[token_id] = map_ctc_label(token, token_id, word_delimiter_token)
    return labels


def clean_beam_decoded_text(value: str) -> str:
    value = value.replace(PYCTC_UNK_LABEL, "")
    return re.sub(r"[\ue000-\uf8ff]", "", value)


def resolve_ffmpeg_command() -> list[str] | None:
    for name in ("ffmpeg", "ffmpeg.exe"):
        candidate = shutil.which(name)
        if candidate:
            return [candidate]

    try:
        completed = subprocess.run(
            ["cmd.exe", "/c", "where", "ffmpeg"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if completed.returncode != 0:
        return None

    for line in completed.stdout.splitlines():
        candidate = line.strip()
        if candidate:
            return [candidate]
    return None


def path_for_ffmpeg(raw_path: Path, ffmpeg_cmd: list[str]) -> str:
    command_name = (ffmpeg_cmd[0] or "").lower()
    if not command_name.endswith(".exe"):
        return str(raw_path)

    try:
        completed = subprocess.run(
            ["wslpath", "-w", str(raw_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return str(raw_path)

    if completed.returncode != 0:
        return str(raw_path)

    converted = completed.stdout.strip()
    return converted or str(raw_path)


def load_audio_via_ffmpeg(audio_path: Path, original_exc: Exception) -> tuple[torch.Tensor, int]:
    ffmpeg_cmd = resolve_ffmpeg_command()
    if ffmpeg_cmd is None:
        raise RuntimeError(
            f"Error opening {audio_path!r}: {original_exc}. Install ffmpeg or provide WAV audio."
        ) from original_exc

    converted_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            converted_path = Path(handle.name)

        completed = subprocess.run(
            ffmpeg_cmd
            + [
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path_for_ffmpeg(audio_path, ffmpeg_cmd),
                "-f",
                "wav",
                "-ac",
                "1",
                "-ar",
                str(TARGET_SR),
                "-acodec",
                "pcm_s16le",
                path_for_ffmpeg(converted_path, ffmpeg_cmd),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            ffmpeg_error = (completed.stderr or completed.stdout or "").strip()
            if ffmpeg_error == "":
                ffmpeg_error = "ffmpeg failed to decode audio"
            raise RuntimeError(
                f"Error opening {audio_path!r}: {original_exc}. ffmpeg decode failed: {ffmpeg_error}"
            ) from original_exc

        if torchaudio is not None:
            return torchaudio.load(str(converted_path))
        audio_array, sr = librosa.load(str(converted_path), sr=None, mono=False)
        if getattr(audio_array, "ndim", 1) == 1:
            waveform = torch.from_numpy(audio_array.copy()).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_array.copy())
        return waveform, int(sr)
    finally:
        if converted_path is not None and converted_path.exists():
            try:
                converted_path.unlink()
            except OSError:
                pass


def load_audio_file(audio_path: Path) -> tuple[torch.Tensor, int]:
    if torchaudio is None:
        audio_array, sr = librosa.load(str(audio_path), sr=None, mono=False)
        if getattr(audio_array, "ndim", 1) == 1:
            waveform = torch.from_numpy(audio_array.copy()).unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_array.copy())
        return waveform, int(sr)
    try:
        return torchaudio.load(str(audio_path))
    except Exception as exc:
        return load_audio_via_ffmpeg(audio_path, exc)


def resample_waveform(waveform: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return waveform
    if torchaudio is not None:
        return torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.dim() == 1:
        resampled = librosa.resample(waveform.detach().cpu().numpy(), orig_sr=sr, target_sr=target_sr)
        return torch.from_numpy(resampled.copy())
    channels = []
    for channel in waveform:
        resampled = librosa.resample(channel.detach().cpu().numpy(), orig_sr=sr, target_sr=target_sr)
        channels.append(torch.from_numpy(resampled.copy()))
    return torch.stack(channels, dim=0)


def load_model_bundle(
    model_source: str,
    device_name: str,
    precision: str,
    decode_mode: str,
    beam_width: int,
) -> dict:
    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_source)
    model_kwargs = {}
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = Wav2Vec2ForCTC.from_pretrained(model_source, **model_kwargs)
    model = model.to(device_name).eval()

    local_model_dir = Path(model_source).expanduser().resolve()
    if not local_model_dir.exists():
        raise SystemExit(f"Model source must be a local path for the server: {model_source}")

    model_config = load_local_model_config(local_model_dir)
    vocab_path = local_model_dir / "vocab.json"
    tokenizer_config_path = local_model_dir / "tokenizer_config.json"
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    word_delimiter_token = tokenizer_config.get("word_delimiter_token", "|")
    resolved_decode_mode = decode_mode
    if resolved_decode_mode == "auto":
        resolved_decode_mode = str(model_config.get("preferred_decoder") or "beam").strip().lower()

    decoder = None
    if resolved_decode_mode == "beam":
        try:
            from pyctcdecode import build_ctcdecoder
        except ModuleNotFoundError:
            print(
                "pyctcdecode not installed; falling back to greedy CTC decoding.",
                file=sys.stderr,
                flush=True,
            )
            resolved_decode_mode = "greedy"
        else:
            decoder = build_ctcdecoder(
                build_pyctc_labels(
                    vocab,
                    word_delimiter_token,
                    vocab_size=int(model.config.vocab_size),
                    added_tokens_decoder=tokenizer_config.get("added_tokens_decoder"),
                )
            )

    return {
        "feature_extractor": feature_extractor,
        "model": model,
        "device": torch.device(device_name),
        "feature_dtype": resolve_feature_dtype(model),
        "id_to_token": {token_id: token for token, token_id in vocab.items()},
        "blank_id": int(model.config.pad_token_id),
        "word_delimiter_token": word_delimiter_token,
        "decoder_mode": resolved_decode_mode,
        "decoder": decoder,
        "beam_width": int(model_config.get("beam_width") or beam_width),
        "redentalize_output": bool(model_config.get("redentalize_output")),
        "lock": threading.Lock(),
    }


def decode_ctc_ids(
    predicted_ids: torch.Tensor,
    id_to_token: dict[int, str],
    blank_id: int,
    word_delimiter_token: str,
) -> str:
    pieces: list[str] = []
    prev = None
    for item in predicted_ids.tolist():
        if item == prev:
            continue
        prev = item
        if item == blank_id:
            continue
        token = id_to_token.get(item, "")
        if not token or token.startswith("<"):
            continue
        if token == word_delimiter_token:
            pieces.append(" ")
        else:
            pieces.append(token)
    return normalize_ipa_text("".join(pieces))


def transcribe_file(bundle: dict, audio_path: Path) -> str:
    waveform, sr = load_audio_file(audio_path)
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0, keepdim=False)
    if sr != TARGET_SR:
        waveform = resample_waveform(waveform, sr, TARGET_SR)

    features = bundle["feature_extractor"](
        waveform.numpy(),
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    input_values = features.input_values.to(bundle["device"])
    if bundle.get("feature_dtype") is not None:
        input_values = input_values.to(dtype=bundle["feature_dtype"])
    attention_mask = getattr(features, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle["device"])

    with bundle["lock"]:
        with torch.no_grad():
            logits = bundle["model"](
                input_values=input_values,
                attention_mask=attention_mask,
            ).logits

    if bundle.get("decoder") is not None:
        return normalize_ipa_text(
            clean_beam_decoded_text(
                bundle["decoder"].decode(
                    logits[0].float().cpu().numpy(),
                    beam_width=bundle["beam_width"],
                )
            ),
            redentalize_stops=bundle["redentalize_output"],
        )

    predicted_ids = torch.argmax(logits, dim=-1)[0].cpu()
    prediction = decode_ctc_ids(
        predicted_ids=predicted_ids,
        id_to_token=bundle["id_to_token"],
        blank_id=bundle["blank_id"],
        word_delimiter_token=bundle["word_delimiter_token"],
    )
    return normalize_ipa_text(
        prediction,
        redentalize_stops=bundle["redentalize_output"],
    )


def make_handler(bundle: dict, allowed_origin: str, api_token: str):
    class LocalMmsHandler(BaseHTTPRequestHandler):
        server_version = "LLToolsLocalMMS/1.0"

        def _resolve_origin(self) -> str:
            origin = (self.headers.get("Origin") or "").strip()
            if allowed_origin == "*":
                return origin or "*"
            return allowed_origin

        def _send_cors_headers(self) -> None:
            requested_headers = (self.headers.get("Access-Control-Request-Headers") or "").strip()
            self.send_header("Access-Control-Allow-Origin", self._resolve_origin())
            self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
            self.send_header(
                "Access-Control-Allow-Headers",
                requested_headers or "Content-Type",
            )
            self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Vary", "Origin, Access-Control-Request-Headers")

        def _send_json(self, status_code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _require_auth(self) -> bool:
            expected = (api_token or "").strip()
            if expected == "":
                return True

            provided = (self.headers.get("Authorization") or "").strip()
            if provided.lower().startswith("bearer "):
                provided = provided[7:].strip()
            else:
                provided = (self.headers.get("X-LL-Tools-Token") or "").strip()

            if provided == "":
                return False
            return hmac.compare_digest(provided, expected)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_GET(self) -> None:
            normalized_path = self.path.rstrip("/")
            if normalized_path not in ("/transcribe", "/health"):
                self._send_json(404, {"error": "not_found"})
                return

            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "local_mms_stt",
                    "endpoint": "/transcribe",
                },
            )

        def do_HEAD(self) -> None:
            normalized_path = self.path.rstrip("/")
            if normalized_path not in ("/transcribe", "/health"):
                self.send_response(404)
            else:
                self.send_response(200)
            self._send_cors_headers()
            self.end_headers()

        def do_POST(self) -> None:
            if self.path.rstrip("/") != "/transcribe":
                self._send_json(404, {"error": "not_found"})
                return
            if not self._require_auth():
                self.send_response(401)
                self._send_cors_headers()
                self.send_header("WWW-Authenticate", 'Bearer realm="ll-tools-stt"')
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "unauthorized"}).encode("utf-8"))
                return

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type.lower():
                self._send_json(400, {"error": "expected_multipart_form_data"})
                return

            content_length_raw = self.headers.get("Content-Length", "0")
            try:
                content_length = int(content_length_raw)
            except (TypeError, ValueError):
                content_length = 0
            if content_length <= 0:
                self._send_json(400, {"error": "missing_request_body"})
                return

            request_body = self.rfile.read(content_length)
            form = parse_multipart_form_data(content_type, request_body)
            audio_field = form.get("audio")
            if not audio_field:
                self._send_json(400, {"error": "missing_audio"})
                return

            upload_name = str(audio_field.get("filename") or "").strip() or "recording.wav"
            audio_bytes = audio_field.get("content") or b""
            if not isinstance(audio_bytes, (bytes, bytearray)) or len(audio_bytes) == 0:
                self._send_json(400, {"error": "missing_audio"})
                return

            suffix = Path(upload_name).suffix or ".wav"
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    temp_path = Path(handle.name)
                    handle.write(audio_bytes)

                started_at = time.monotonic()
                predicted_ipa = transcribe_file(bundle, temp_path)
                if predicted_ipa == "":
                    self._send_json(422, {"error": "empty_transcript"})
                    return

                self._send_json(
                    200,
                    {
                        "predicted_ipa": predicted_ipa,
                        "text": predicted_ipa,
                        "filename": upload_name,
                        "duration_ms": int((time.monotonic() - started_at) * 1000),
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime safeguard
                self._send_json(
                    500,
                    {
                        "error": "transcription_failed",
                        "message": str(exc),
                    },
                )
            finally:
                if temp_path is not None and temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass

        def log_message(self, format_string: str, *args) -> None:
            sys.stdout.write(
                "%s - - [%s] %s\n"
                % (self.address_string(), self.log_date_time_string(), format_string % args)
            )

    return LocalMmsHandler


def main() -> None:
    if DEPENDENCY_IMPORT_ERROR is not None:
        script_dir = Path(__file__).resolve().parent
        launcher_path = script_dir / "serve_local_stt_mms.cmd"
        package_name = getattr(DEPENDENCY_IMPORT_ERROR, "name", "required package")
        raise SystemExit(
            "Missing Python package "
            + repr(package_name)
            + " in this interpreter. Run "
            + str(launcher_path)
            + " or use the WSL MMS environment."
        )

    args = parse_args()
    device_name = choose_device(args.device)

    print(f"Loading MMS model from: {args.model_source}")
    print(f"Using device:          {device_name}")
    bundle = load_model_bundle(
        args.model_source,
        device_name,
        args.precision,
        args.decoder,
        args.beam_width,
    )

    try:
        server = ThreadingHTTPServer(
            (args.host, args.port),
            make_handler(bundle, args.allowed_origin, args.api_token),
        )
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Port {args.port} is already in use. Stop the other MMS server or run with --port <new_port>."
            ) from exc
        raise
    print(f"Using precision:       {args.precision}")
    print(f"Using decoder:         {bundle['decoder_mode']}")
    if bundle["decoder_mode"] == "beam":
        print(f"Using beam width:      {bundle['beam_width']}")
    print(f"Serving local MMS STT at http://{args.host}:{args.port}/transcribe")
    if args.api_token:
        print("POST /transcribe requires bearer token authentication.")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
