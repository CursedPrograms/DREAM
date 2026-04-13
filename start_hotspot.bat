@echo off
SETLOCAL ENABLEDELAYEDEXPANSION
REM ============================================================
REM  ComCentre - Master Start Script (Windows)
REM ============================================================

REM Change this to your ComCentre/FRIDAY directory
SET FRIDAY_DIR=C:\Users\cursed\Desktop\ComCentre

ECHO.
ECHO ╔══════════════════════════════════════╗
ECHO ║          Starting ComCentre...       ║
ECHO ╚══════════════════════════════════════╝
ECHO.

REM ---------------------------
REM Check Python venv
REM ---------------------------
SET VENV_PYTHON=%FRIDAY_DIR%\venv\Scripts\python.exe

IF NOT EXIST "%VENV_PYTHON%" (
    ECHO ❌ venv not found. Run setup first.
    EXIT /B 1
)

REM ---------------------------
REM 1. Start hotspot (Windows version must be adapted)
REM ---------------------------
ECHO [1/3] Bringing up WiFi hotspot...
CALL "%FRIDAY_DIR%\hotspot.bat"

REM ---------------------------
REM 2. Start Ollama if installed
REM ---------------------------
ECHO [2/3] Starting Ollama...
where ollama >nul 2>&1
IF ERRORLEVEL 1 (
    ECHO Ollama not found. Please install Ollama manually.
) ELSE (
    tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
    IF ERRORLEVEL 1 (
        START "" /B ollama serve
        TIMEOUT /T 3 >nul
        ECHO   → Ollama started
    ) ELSE (
        ECHO   → Ollama already running
    )
)

REM ---------------------------
REM 3. Start web server
REM ---------------------------
ECHO [3/3] Starting web server on http://192.168.4.1 ...
CD /D "%FRIDAY_DIR%"

REM Logging
SET LOG_FILE=%FRIDAY_DIR%\server.log
ECHO 📜 Logging webserver output to %LOG_FILE%

REM Run webserver using venv Python
"%VENV_PYTHON%" webserver.py > "%LOG_FILE%" 2>&1

REM ---------------------------
REM Stop hotspot after server exits
REM ---------------------------
ECHO.
ECHO [*] Web server stopped. Tearing down hotspot...
CALL "%FRIDAY_DIR%\stop_hotspot.bat"

ECHO [*] ComCentre session ended.
PAUSE
