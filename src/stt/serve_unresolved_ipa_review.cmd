@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if /I "%~1"=="--stop" (
    bash -lc "pkill -f 'python3 .*serve_unresolved_ipa_review.py' || true"
    endlocal
    exit /b 0
)

bash -lc "python3 - <<'PY'\nimport json, sys, urllib.request\ntry:\n    with urllib.request.urlopen('http://127.0.0.1:8770/api/bootstrap', timeout=1.5) as r:\n        obj = json.load(r)\n    if isinstance(obj, dict) and obj.get('total_rows') is not None:\n        print('RUNNING')\n    else:\n        print('NOTRUNNING')\nexcept Exception:\n    print('NOTRUNNING')\nPY" > "%TEMP%\\unresolved_ipa_review_status.txt"
set /p CODEx_REVIEW_STATUS=<"%TEMP%\\unresolved_ipa_review_status.txt"
del "%TEMP%\\unresolved_ipa_review_status.txt" >nul 2>nul

if /I "%CODEX_REVIEW_STATUS%"=="RUNNING" (
    start "" "http://127.0.0.1:8770/"
    echo Unresolved IPA review UI already running at http://127.0.0.1:8770/
    endlocal
    exit /b 0
)

bash -lc "python3 '$Z_WORKSPACE/Speech to Text IPA/serve_unresolved_ipa_review.py' --open-browser %*"

endlocal
