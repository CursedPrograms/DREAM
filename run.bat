@echo off
REM run.bat - DREAM's entry point. Sets up .\venv311 (Python 3.11) the first
REM time (and again whenever requirements.txt changes), then starts BOTH:
REM   app.py           - DREAM's web server: dashboard, sensor board,
REM                      RIFT registration (in its own minimised window)
REM   scripts\dream.py - DREAM herself (avatar, wake word, voice)
REM   memory_server.py - the fleet's long-term memory (Hindsight), which TINA uses
REM                      too - only if Config.DREAM.Memory.Hindsight is true
REM Closing DREAM closes the server too. Arguments go to dream.py (e.g. --web).
REM Model weights for lip-sync are separate: download_weights.bat.
setlocal
cd /d "%~dp0"
set "VENV=%~dp0venv311"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating venv311 ...
    py -3.11 -m venv "%VENV%" || goto :nopython
)
if not exist "%PY%" goto :nopython

REM (Re)install only when requirements.txt changed since the last install.
fc /b requirements.txt "%VENV%\requirements.installed" >nul 2>&1
if errorlevel 1 (
    echo Installing requirements - the first time takes a while ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt openai-whisper piper-tts pathvalidate sounddevice soundfile faster-whisper || goto :fail
    copy /y requirements.txt "%VENV%\requirements.installed" >nul
)

where ollama >nul 2>&1 || echo Ollama not found - install it from https://ollama.com for DREAM's voice chat.

REM The fleet's memory: its own environment (CPU only, no PyTorch), set up the first time.
"%PY%" -c "import json,sys; sys.exit(0 if json.load(open('config.json',encoding='utf-8'))['Config']['DREAM'].get('Memory',{}).get('Hindsight') else 1)" && call :memory

start "DREAM Dashboard" /min "%PY%" app.py
"%PY%" scripts\dream.py %*
taskkill /fi "WINDOWTITLE eq DREAM Dashboard*" /t /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq DREAM Memory*" /t /f >nul 2>&1
exit /b

:memory
if not exist "hindsight-venv\Scripts\hindsight-api.exe" (
    echo Setting up the fleet's memory server - the first time takes a while ...
    uv venv hindsight-venv --python 3.12 || (echo uv not found - skipping the memory server & exit /b)
    uv pip install --python hindsight-venv\Scripts\python.exe hindsight-api-slim pg0-embedded onnxruntime tokenizers huggingface-hub flashrank || exit /b
)
start "DREAM Memory (Hindsight)" /min hindsight-venv\Scripts\python.exe memory_server.py
exit /b

:nopython
echo Python was not found. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then run this again.
pause
exit /b 1

:fail
echo Setup failed - see the errors above.
pause
exit /b 1
