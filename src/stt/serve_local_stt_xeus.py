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
    import torch
    import torchaudio
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
    DEPENDENCY_IMPORT_ERROR = exc

from prepare_lltools_ipa_dataset import normalize_ipa_text


TARGET_SR = 16000


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def default_repo_root() -> str:
    return "$LINUX_STT_ENV/alt-models/PhoneticXeus"


def default_work_dir() -> str:
    return "$LINUX_STT_ENV/alt-models/cache/xeus"


def default_checkpoint() -> str:
    candidates = [
        Path(
            "$LINUX_STT_ENV/alt-models/PhoneticXeus/exp/runs/train_ipapack_xeuspr/20260404_zazaki_xeus_longresume_v001/checkpoints/last.ckpt"
        ),
        Path(
            "$LINUX_STT_ENV/alt-models/PhoneticXeus/exp/runs/train_ipapack_xeuspr/20260404_zazaki_xeus_v006/checkpoints/checkpoint-68.ckpt"
        ),
        Path(
            "$HF_HOME/hub/models--changelinglab--PhoneticXeus/snapshots/91240a28a35578f34482341fc46822589da4a6c8/checkpoint-22000.ckpt"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def choose_device(value: str) -> str:
    requested = (value or "").strip().lower()
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        description="Serve a local HTTP endpoint that transcribes uploaded audio to IPA with PhoneticXeus."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--repo-root", default=default_repo_root())
    parser.add_argument("--work-dir", default=default_work_dir())
    parser.add_argument("--checkpoint", default=default_checkpoint())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
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


def load_model_bundle(repo_root: str, work_dir: str, checkpoint: str, device_name: str, dtype: str) -> dict:
    repo_path = Path(repo_root).expanduser().resolve()
    if not repo_path.exists():
        raise SystemExit(f"Missing repo root: {repo_path}")

    sys.path.insert(0, str(repo_path))
    from src.model.xeusphoneme.builders import build_xeus_pr_inference

    inference = build_xeus_pr_inference(
        work_dir=str(Path(work_dir).expanduser().resolve()),
        checkpoint=str(Path(checkpoint).expanduser().resolve()),
        vocab_file=str(repo_path / "src/model/xeusphoneme/resources/ipa_vocab.json"),
        hf_repo=None,
        device=device_name,
        dtype=dtype,
        interctc_use_conditioning=True,
    )

    return {
        "inference": inference,
        "lock": threading.Lock(),
        "device": device_name,
        "dtype": dtype,
    }


def transcribe_file(bundle: dict, audio_path: Path) -> str:
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.dim() > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

    with bundle["lock"]:
        result = bundle["inference"](waveform)

    raw_prediction = ""
    if result:
        raw_prediction = str(result[0].get("processed_transcript") or "")
    return normalize_ipa_text(raw_prediction)


def make_handler(bundle: dict, allowed_origin: str, api_token: str):
    class LocalXeusHandler(BaseHTTPRequestHandler):
        server_version = "LLToolsLocalXeus/1.0"

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
                    "service": "local_xeus_stt",
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

    return LocalXeusHandler


def main() -> None:
    if DEPENDENCY_IMPORT_ERROR is not None:
        script_dir = Path(__file__).resolve().parent
        launcher_path = script_dir / "serve_local_stt_xeus.cmd"
        package_name = getattr(DEPENDENCY_IMPORT_ERROR, "name", "required package")
        raise SystemExit(
            "Missing Python package "
            + repr(package_name)
            + " in this interpreter. Run "
            + str(launcher_path)
            + " or use the WSL PhoneticXeus environment."
        )

    args = parse_args()
    device_name = choose_device(args.device)

    print(f"Loading Xeus checkpoint: {args.checkpoint}")
    print(f"Using repo root:         {args.repo_root}")
    print(f"Using device:            {device_name}")
    bundle = load_model_bundle(
        args.repo_root,
        args.work_dir,
        args.checkpoint,
        device_name,
        args.dtype,
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(bundle, args.allowed_origin, args.api_token),
    )
    print(f"Using dtype:             {args.dtype}")
    print(f"Serving local Xeus STT at http://{args.host}:{args.port}/transcribe")
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
