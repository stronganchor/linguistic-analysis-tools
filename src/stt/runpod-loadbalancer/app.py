import os
import hmac
import tempfile
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from serve_local_stt_ipa import choose_device, load_model_bundle, transcribe_file


MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "/models/current")
REQUEST_DEVICE = os.environ.get("DEVICE", "auto")
API_TOKEN = (
    os.environ.get("LL_TOOLS_STT_API_TOKEN")
    or os.environ.get("HOSTED_STT_API_TOKEN")
    or os.environ.get("RUNPOD_STT_API_TOKEN")
    or ""
).strip()
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
# Bound the complete multipart request before the framework can spool uploads.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
DEBUG_HEALTH = os.environ.get("DEBUG_HEALTH", "").strip().lower() in {"1", "true", "yes"}
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if item.strip()
]

app = FastAPI(title="LL Tools Zazaki STT", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HostedRequestGuard:
    """Authenticate before body parsing and cap actual ASGI bytes, including chunks."""

    def __init__(self, app, max_request_bytes: int):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"].rstrip("/") != "/transcribe":
            await self.app(scope, receive, send)
            return
        from starlette.datastructures import Headers
        from starlette.formparsers import MultiPartException

        headers = Headers(scope=scope)
        try:
            _require_auth(headers.get("authorization"), headers.get("x-ll-tools-token"))
            lengths = headers.getlist("content-length")
            if lengths:
                if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit():
                    raise HTTPException(status_code=400, detail="invalid_content_length")
                if int(lengths[0]) > self.max_request_bytes:
                    raise HTTPException(status_code=413, detail="request_too_large")
        except HTTPException as exc:
            await JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)(scope, receive, send)
            return

        total = 0
        exceeded = False

        async def limited_receive():
            nonlocal total, exceeded
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_request_bytes:
                    exceeded = True
                    # Older Starlette releases clean partial spools only for this
                    # parser exception. Replace their resulting 400 with 413 below.
                    raise MultiPartException("request_too_large")
            return message

        async def limited_send(message):
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except HTTPException as exc:
            if not exceeded:
                await JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)(scope, receive, send)
        except MultiPartException:
            if not exceeded:
                raise
        if exceeded:
            await JSONResponse({"detail": "request_too_large"}, status_code=413)(scope, receive, send)


app.add_middleware(HostedRequestGuard, max_request_bytes=MAX_REQUEST_BYTES)

STATE = {
    "bundle": None,
    "loading": False,
    "ready": False,
    "error": "",
    "model_source": MODEL_SOURCE,
    "device": choose_device(REQUEST_DEVICE),
    "started_at": time.time(),
}
STATE_LOCK = threading.Lock()


def _load_bundle_background() -> None:
    with STATE_LOCK:
        if STATE["loading"] or STATE["ready"]:
            return
        STATE["loading"] = True
        STATE["error"] = ""

    try:
        bundle = load_model_bundle(STATE["model_source"], STATE["device"])
    except Exception:
        with STATE_LOCK:
            STATE["bundle"] = None
            STATE["ready"] = False
            STATE["loading"] = False
            STATE["error"] = traceback.format_exc()
        return

    with STATE_LOCK:
        STATE["bundle"] = bundle
        STATE["ready"] = True
        STATE["loading"] = False
        STATE["error"] = ""


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=_load_bundle_background, daemon=True).start()


def _status_payload() -> dict:
    with STATE_LOCK:
        return {
            "status": "ok" if STATE["ready"] else ("loading" if STATE["loading"] else "error"),
            "service": "ll_tools_zazaki_stt",
            "endpoint": "/transcribe",
            "model_source": STATE["model_source"],
            "device": STATE["device"],
            "ready": bool(STATE["ready"]),
            "loading": bool(STATE["loading"]),
            "uptime_seconds": int(max(0, time.time() - STATE["started_at"])),
            "auth_configured": bool(API_TOKEN),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "error": STATE["error"] if (DEBUG_HEALTH and not STATE["ready"]) else ("model_failed" if STATE["error"] else ""),
        }


@app.get("/ping")
def ping() -> Response:
    with STATE_LOCK:
        if STATE["ready"]:
            return Response(status_code=200)
        if STATE["error"]:
            return Response(status_code=503)
        return Response(status_code=204)


@app.get("/health")
def health() -> JSONResponse:
    payload = _status_payload()
    if payload["ready"]:
        return JSONResponse(payload, status_code=200)
    if payload["loading"]:
        return JSONResponse(payload, status_code=202)
    return JSONResponse(payload, status_code=503)


def _require_auth(authorization: str | None, x_ll_tools_token: str | None) -> None:
    if API_TOKEN == "":
        raise HTTPException(status_code=503, detail="api_token_not_configured")

    provided = (authorization or "").strip()
    if provided.lower().startswith("bearer "):
        provided = provided[7:].strip()
    else:
        provided = (x_ll_tools_token or "").strip()

    if provided == "" or not hmac.compare_digest(provided.encode(), API_TOKEN.encode()):
        raise HTTPException(
            status_code=401,
            detail="unauthorized",
            headers={"WWW-Authenticate": 'Bearer realm="ll-tools-stt"'},
        )


async def _read_upload_limited(audio: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await audio.read(1024 * 1024)
        if not chunk:
            break

        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="audio_too_large")

        chunks.append(chunk)

    return b"".join(chunks)


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    x_ll_tools_token: str | None = Header(default=None, alias="X-LL-Tools-Token"),
) -> JSONResponse:
    _require_auth(authorization, x_ll_tools_token)

    with STATE_LOCK:
        if STATE["loading"]:
            raise HTTPException(status_code=503, detail="model_loading")
        if STATE["error"]:
            raise HTTPException(status_code=503, detail="model_failed")
        bundle = STATE["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="model_unavailable")

    upload_name = (audio.filename or "").strip() or "recording.wav"
    audio_bytes = await _read_upload_limited(audio)
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="missing_audio")

    suffix = Path(upload_name).suffix or ".wav"
    temp_path = None
    started_at = time.monotonic()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            temp_path = Path(handle.name)
            handle.write(audio_bytes)

        predicted_ipa = transcribe_file(bundle, temp_path)
        if predicted_ipa == "":
            raise HTTPException(status_code=422, detail="empty_transcript")

        return JSONResponse(
            {
                "predicted_ipa": predicted_ipa,
                "text": predicted_ipa,
                "filename": upload_name,
                "duration_ms": int((time.monotonic() - started_at) * 1000),
                "model_source": STATE["model_source"],
            },
            status_code=200,
        )
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
