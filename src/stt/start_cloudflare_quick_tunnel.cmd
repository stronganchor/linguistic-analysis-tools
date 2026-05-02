@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "CLOUDFLARED_EXE=%SCRIPT_DIR%cloudflared.exe"
set "UPSTREAM_URL=http://127.0.0.1:8765"

if not exist "%CLOUDFLARED_EXE%" (
    echo Missing cloudflared.exe at:
    echo %CLOUDFLARED_EXE%
    exit /b 1
)

echo Starting Cloudflare Quick Tunnel for the hosted MMS STT server at %UPSTREAM_URL%
echo Copy the public https://...trycloudflare.com URL that appears below
echo and use that URL plus /transcribe in LL Tools.
echo.

"%CLOUDFLARED_EXE%" tunnel --url %UPSTREAM_URL%
