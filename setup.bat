@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
REM ============================================================
REM  ComCentre - Windows Venv + Dependencies Setup
REM ============================================================

REM Set the FRIDAY_DIR to your project folder
SET FRIDAY_DIR=C:\Users\cursed\Desktop\ComCentre

REM ---------------------------
REM Check Python 3.12
REM ---------------------------
python --version | findstr "3.12" >nul
IF ERRORLEVEL 1 (
    ECHO ❌ Python 3.12 not found! Please install Python 3.12 before running this script.
    EXIT /B 1
)

REM ---------------------------
REM Create venv
REM ---------------------------
ECHO 🐍 Creating venv with Python 3.12...
python -m venv "%FRIDAY_DIR%\venv"

SET VENV_PYTHON=%FRIDAY_DIR%\venv\Scripts\python.exe
SET VENV_PIP=%FRIDAY_DIR%\venv\Scripts\pip.exe

REM ---------------------------
REM Upgrade pip
REM ---------------------------
ECHO ⚙️ Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

REM ---------------------------
REM Install requirements
REM ---------------------------
ECHO 📦 Installing requirements.txt...
"%VENV_PIP%" install -r "%FRIDAY_DIR%\requirements.txt"

REM ---------------------------
REM Install additional packages
REM ---------------------------
ECHO 📦 Installing additional packages...
"%VENV_PIP%" install openai-whisper piper-tts pathvalidate sounddevice soundfile numpy requests faster-whisper

REM ---------------------------
REM Ollama check
REM ---------------------------
where ollama >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO 🧠 Ollama not found! Please install Ollama manually: https://ollama.com
) ELSE (
    ECHO ✔ Ollama already installed
    ollama --version
)

ECHO ✅ Setup complete!
PAUSE
