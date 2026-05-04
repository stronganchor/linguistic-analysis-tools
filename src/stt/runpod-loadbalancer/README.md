# Runpod Load Balancer STT Deployment

This folder packages the Genç-Palu Whisper-small IPA model as a direct HTTP STT service for Runpod.

This is intended for a **Runpod Load Balancer endpoint**, not the queue-based `runsync` API. A load-balancer endpoint is the better fit because LL Tools already knows how to call a normal hosted HTTP endpoint like:

```text
https://your-endpoint.runpod.ai/transcribe
```

## Why this shape

- low-latency speaking game requests
- same `/health` and `/transcribe` contract you already use locally
- no new LL Tools plugin integration path needed

## Files

- `app.py`: FastAPI STT service
- `Dockerfile`: GPU container image
- `requirements.txt`: inference dependencies
- `build_runpod_image.cmd`: prepares the chosen model and builds a Linux AMD64 image

## Prepare the model

Before each build, copy the current model into the fixed Docker build context:

```bat
"%Z_WORKSPACE%\audio_corpus\venv311\Scripts\python.exe" ^
  "%Z_WORKSPACE%\Speech to Text IPA\prepare_runpod_model.py"
```

By default, that picks the newest non-mobile model from `models\`.

To force a specific model:

```bat
"%Z_WORKSPACE%\audio_corpus\venv311\Scripts\python.exe" ^
  "%Z_WORKSPACE%\Speech to Text IPA\prepare_runpod_model.py" ^
  --model-source "%Z_MODELS_DIR%\speech_to_text_ipa\whisper-small-ipa-nostress-hayvanlar1-v006"
```

## Build the image

Example with a Docker Hub tag:

```bat
"%Z_WORKSPACE%\Speech to Text IPA\runpod-loadbalancer\build_runpod_image.cmd" yourdockerhubuser/lltools-zazaki-stt:latest
```

Then push it:

```bat
docker push yourdockerhubuser/lltools-zazaki-stt:latest
```

## Deploy on Runpod

In Runpod:

1. Go to `Serverless`
2. Choose `Custom deployment`
3. Choose the **Load Balancing** endpoint type if available
4. Use your pushed Docker image
5. Set the container port to `8000`
6. Set the health path to `/ping`
7. Use a GPU worker type suitable for low-latency inference
8. Add an environment variable named `LL_TOOLS_STT_API_TOKEN` with a long random token
9. Keep **Max workers** at `0` whenever the endpoint is not actively needed

For a proof of concept with a few users, start with one worker and a smaller GPU like an L4 or RTX 4000 Ada if available. Increase only if cold starts or latency are a problem.

## App endpoints

- `GET /ping`
- `GET /health`
- `POST /transcribe` with multipart field `audio` and a bearer token matching `LL_TOOLS_STT_API_TOKEN`

Response:

```json
{
  "predicted_ipa": "kʷɛ ho t͡ʃada",
  "text": "kʷɛ ho t͡ʃada",
  "filename": "recording.mp3",
  "duration_ms": 742
}
```

## LL Tools settings

You do not need a new plugin feature for Runpod.

Use the existing `Hosted STT API` settings:

1. `Lesson Transcription Provider`: `Hosted STT API`
2. `Custom STT Output`: `Secondary transcription field (IPA)`
3. `Custom STT Endpoint`: your Runpod endpoint plus `/transcribe`
4. `Hosted API Token`: the same token you set in `LL_TOOLS_STT_API_TOKEN`
5. `Speaking Game STT Provider`: `Hosted STT API`
6. `Speaking Game Comparison Target`: `IPA`

Example endpoint:

```text
https://your-runpod-hostname/transcribe
```

## Notes

- Keep the endpoint private and use the token field in LL Tools for the hosted STT token. Do not use or publish your Runpod account API key as the hosted STT token.
- This image bakes the model into the container. That is slower to rebuild, but simplest for now.
- Later, if you want faster builds and rollouts, move the model to object storage and download it during container start.
