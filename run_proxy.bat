@echo off
REM DREAM's body on this PC: her sensor board, webcam and window (dream_proxy.py). The AI runs on the AI PC.
"%~dp0venv311\Scripts\python.exe" "%~dp0dream_proxy.py" %*
if errorlevel 1 pause
