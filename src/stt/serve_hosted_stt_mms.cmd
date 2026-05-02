@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "SERVER_CMD=%SCRIPT_DIR%serve_local_stt_mms.cmd"
set "TOKEN_FILE=%SCRIPT_DIR%hosted_stt_server.token"

if /I "%~1"=="--stop" (
    call "%SERVER_CMD%" --stop
    endlocal
    exit /b 0
)

if not exist "%SERVER_CMD%" (
    echo Missing local MMS launcher:
    echo %SERVER_CMD%
    exit /b 1
)

set "LL_TOOLS_STT_API_TOKEN="
if exist "%TOKEN_FILE%" (
    <"%TOKEN_FILE%" set /p LL_TOOLS_STT_API_TOKEN=
)

if "%LL_TOOLS_STT_API_TOKEN%"=="" (
    echo Missing hosted STT token.
    echo Create or update:
    echo %TOKEN_FILE%
    exit /b 1
)

call "%SERVER_CMD%" --host 0.0.0.0 --port 8765 --api-token "%LL_TOOLS_STT_API_TOKEN%" %*

endlocal
