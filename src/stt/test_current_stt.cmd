@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "DEFAULT_ENDPOINT=http://127.0.0.1:8769/transcribe"
set "DEFAULT_AUDIO=%SCRIPT_DIR%extracted_bundles_20260403\ll-tools-stt-training-genc-palu-recording_ipa-20260403-090111\audio\15356-Agac_introduction_15356_1773210711.wav"

set "ENDPOINT=%~1"
if "%ENDPOINT%"=="" set "ENDPOINT=%DEFAULT_ENDPOINT%"

set "AUDIO_FILE=%~2"
if "%AUDIO_FILE%"=="" set "AUDIO_FILE=%DEFAULT_AUDIO%"

if not exist "%AUDIO_FILE%" (
    echo Missing audio file:
    echo %AUDIO_FILE%
    exit /b 1
)

echo Testing STT endpoint:
echo %ENDPOINT%
echo.
echo Using audio file:
echo %AUDIO_FILE%
echo.

curl.exe -sS -X POST -F "audio=@%AUDIO_FILE%;type=audio/wav" "%ENDPOINT%"
echo.

endlocal
