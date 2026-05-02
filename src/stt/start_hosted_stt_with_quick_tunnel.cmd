@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SERVER_CMD=%SCRIPT_DIR%serve_hosted_stt_mms.cmd"
set "TUNNEL_CMD=%SCRIPT_DIR%start_cloudflare_quick_tunnel.cmd"

if not exist "%SERVER_CMD%" (
    echo Missing server launcher at:
    echo %SERVER_CMD%
    exit /b 1
)

if not exist "%TUNNEL_CMD%" (
    echo Missing tunnel launcher at:
    echo %TUNNEL_CMD%
    exit /b 1
)

echo Starting hosted MMS STT server in a separate window...
start "LL Tools Hosted MMS STT" "%SERVER_CMD%"

echo Waiting for the STT server to come up...
timeout /t 20 /nobreak >nul

echo Starting Cloudflare Quick Tunnel...
call "%TUNNEL_CMD%"
