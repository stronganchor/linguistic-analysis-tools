@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if /I "%~1"=="--stop" (
    bash -lc "pkill -f 'python3 .*serve_local_stt_mms.py' || true"
    endlocal
    exit /b 0
)
bash -lc "source $LINUX_STT_ENV/.venv/bin/activate && python3 '$Z_WORKSPACE/Speech to Text IPA/serve_local_stt_mms.py' %*"

endlocal
