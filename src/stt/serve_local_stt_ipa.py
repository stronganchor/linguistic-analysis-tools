import argparse
import hmac
import json
import os
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
    import librosa
    import torch
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    DEPENDENCY_IMPORT_ERROR = exc

from model_paths import resolve_stt_models_dir, stt_model_path
from prepare_lltools_ipa_dataset import normalize_ipa_text


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def default_model_source() -> str:
    script_dir = Path(__file__).resolve().parent
    models_dir = resolve_stt_models_dir(script_dir)
    discovered_models: list[Path] = []

    if models_dir.is_dir():
        for candidate in models_dir.iterdir():
            if not candidate.is_dir():
                continue
            if candidate.name.lower().startswith("checkpoint-"):
                continue
            if not (candidate / "config.json").is_file():
                continue
            discovered_models.append(candidate)

    if discovered_models:
        newest_model = max(
            discovered_models,
            key=lambda path: (
                getattr(path.stat(), "st_ctime", 0),
                path.stat().st_mtime,
                path.name.lower(),
            ),
        )
        return str(newest_model)

    candidates = [
        stt_model_path(script_dir, "whisper-small-ipa-doga1-from-zazaki-ft002"),
        stt_model_path(script_dir, "whisper-small-ipa-doga1-001"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return "openai/whisper-small"


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
        description="Serve a local HTTP endpoint that transcribes uploaded audio to IPA."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model-source", default=default_model_source())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
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


def load_model_bundle(model_source: str, device_name: str, precision: str, num_beams: int, max_length: int) -> dict:
    torch_dtype = choose_torch_dtype(device_name, precision)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source)
    tokenizer = WhisperTokenizer.from_pretrained(model_source)
    processor = WhisperProcessor.from_pretrained(model_source)
    model_kwargs = {
        "low_cpu_mem_usage": True,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = WhisperForConditionalGeneration.from_pretrained(model_source, **model_kwargs)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=None, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.num_beams = max(1, num_beams)

    device = torch.device(device_name)
    model.to(device)
    model.eval()

    return {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "feature_dtype": resolve_feature_dtype(model),
        "lock": threading.Lock(),
        "num_beams": max(1, num_beams),
        "max_length": max(32, max_length),
        "precision": precision,
    }


def transcribe_file(bundle: dict, audio_path: Path) -> str:
    with bundle["lock"]:
        audio_array, _ = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
        batch = bundle["feature_extractor"](
            [audio_array],
            sampling_rate=TARGET_SR,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = batch["input_features"].to(bundle["device"])
        if bundle.get("feature_dtype") is not None:
            input_features = input_features.to(dtype=bundle["feature_dtype"])

        generate_kwargs = {
            "input_features": input_features,
            "max_length": bundle["max_length"],
            "num_beams": bundle["num_beams"],
        }
        if "attention_mask" in batch:
            generate_kwargs["attention_mask"] = batch["attention_mask"].to(bundle["device"])

        with torch.no_grad():
            generated_ids = bundle["model"].generate(**generate_kwargs)

    prediction = bundle["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    return normalize_ipa_text(prediction)


def make_handler(bundle: dict, allowed_origin: str, api_token: str):
    class LocalIpaHandler(BaseHTTPRequestHandler):
        server_version = "LLToolsLocalIPA/1.0"

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
                    "service": "local_ipa_stt",
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
            sys.stdout.write("%s - - [%s] %s\n" % (
                self.address_string(),
                self.log_date_time_string(),
                format_string % args,
            ))

    return LocalIpaHandler


def main() -> None:
    if DEPENDENCY_IMPORT_ERROR is not None:
        script_dir = Path(__file__).resolve().parent
        launcher_path = script_dir / "serve_local_stt_ipa.cmd"
        package_name = getattr(DEPENDENCY_IMPORT_ERROR, "name", "required package")
        raise SystemExit(
            "Missing Python package "
            + repr(package_name)
            + " in this interpreter. Run "
            + str(launcher_path)
            + " or use ..\\audio_corpus\\venv311\\Scripts\\python.exe."
        )

    args = parse_args()
    device_name = choose_device(args.device)

    print(f"Loading model from: {args.model_source}")
    print(f"Using device:       {device_name}")
    bundle = load_model_bundle(
        args.model_source,
        device_name,
        args.precision,
        args.num_beams,
        args.max_length,
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(bundle, args.allowed_origin, args.api_token))
    print(f"Using precision:    {args.precision}")
    print(f"Beam search:        {bundle['num_beams']}")
    print(f"Max length:         {bundle['max_length']}")
    print(f"Serving local IPA STT at http://{args.host}:{args.port}/transcribe")
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
