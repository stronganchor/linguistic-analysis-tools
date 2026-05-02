import os
import tempfile
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from serve_local_stt_ipa import choose_device, load_model_bundle, transcribe_file


MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "/models/current")
REQUEST_DEVICE = os.environ.get("DEVICE", "auto")

app = FastAPI(title="LL Tools Zazaki STT", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            "error": STATE["error"] if not STATE["ready"] else "",
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


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    with STATE_LOCK:
        if STATE["loading"]:
            raise HTTPException(status_code=503, detail="model_loading")
        if STATE["error"]:
            raise HTTPException(status_code=503, detail="model_failed")
        bundle = STATE["bundle"]

    if bundle is None:
        raise HTTPException(status_code=503, detail="model_unavailable")

    upload_name = (audio.filename or "").strip() or "recording.wav"
    audio_bytes = await audio.read()
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
