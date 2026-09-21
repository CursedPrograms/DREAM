@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM  DREAM (C++) - one-time setup
REM
REM  Gets everything dream.exe needs that isn't in the repo:
REM    1. Build tools   C++ compiler + CMake + Ninja (one winget package) and Git
REM    2. Speech model  Whisper "tiny.en" (77 MB)      -> cpp_dream\models\
REM    3. Voice         Piper (22 MB) + a voice (63 MB) -> piper\ and voices\
REM    4. Ollama        the app, and the phi3:mini model
REM  Only what's missing is downloaded, so it is safe to run again.
REM  Afterwards run build.bat, then build\dream.exe.
REM ============================================================

cd /d "%~dp0"
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "NEWTOOLS=0"
set "PROBLEMS=0"

echo.
echo DREAM (C++) setup
echo Repo: %ROOT%
echo.

REM ---------------------------------------------------------------
echo [1/4] Build tools
where winget >nul 2>&1
if errorlevel 1 (
    set "HAVE_WINGET=0"
) else (
    set "HAVE_WINGET=1"
)

set "NEED_TOOLCHAIN=0"
where g++   >nul 2>&1 || set "NEED_TOOLCHAIN=1"
where cmake >nul 2>&1 || set "NEED_TOOLCHAIN=1"
where ninja >nul 2>&1 || set "NEED_TOOLCHAIN=1"
if "!NEED_TOOLCHAIN!"=="1" (
    if "!HAVE_WINGET!"=="1" (
        echo   Installing the C++ toolchain ^(WinLibs GCC + CMake + Ninja^)...
        winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT --accept-package-agreements --accept-source-agreements
        set "NEWTOOLS=1"
    ) else (
        echo   MISSING: g++, cmake or ninja, and winget isn't available to install them.
        echo   Install WinLibs ^(https://winlibs.com^) - it includes all three - and add it to PATH.
        set "PROBLEMS=1"
    )
) else (
    echo   g++, cmake, ninja: found
)
where git >nul 2>&1
if errorlevel 1 (
    if "!HAVE_WINGET!"=="1" (
        echo   Installing Git ^(needed to fetch whisper.cpp^)...
        winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
        set "NEWTOOLS=1"
    ) else (
        echo   MISSING: git - install it from https://git-scm.com
        set "PROBLEMS=1"
    )
) else (
    echo   git: found
)

REM ---------------------------------------------------------------
echo.
echo [2/4] Whisper speech model
set "MODEL=%~dp0models\ggml-tiny.en.bin"
call :download "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" "!MODEL!" 70000000 "Whisper model, 77 MB"

REM ---------------------------------------------------------------
echo.
echo [3/4] Piper voice
set "PIPER="
if exist "%ROOT%\venv311\Scripts\piper.exe" set "PIPER=%ROOT%\venv311\Scripts\piper.exe"
if exist "%ROOT%\venv\Scripts\piper.exe"    set "PIPER=%ROOT%\venv\Scripts\piper.exe"
if exist "%ROOT%\piper\piper.exe"           set "PIPER=%ROOT%\piper\piper.exe"
if defined PIPER (
    echo   piper.exe: found ^(!PIPER!^)
) else (
    call :download "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip" "%~dp0piper_windows_amd64.zip" 20000000 "Piper, 22 MB"
    if exist "%~dp0piper_windows_amd64.zip" (
        echo   Unpacking Piper into %ROOT%\piper ...
        REM Windows' own tar.exe: if Git is on PATH first, "tar" is GNU tar, which reads C:\... as a remote host.
        "%SystemRoot%\System32\tar.exe" -xf "%~dp0piper_windows_amd64.zip" -C "%ROOT%"
        if exist "%ROOT%\piper\piper.exe" (
            del "%~dp0piper_windows_amd64.zip" >nul 2>&1
            echo   piper.exe: installed
        ) else (
            echo   PROBLEM: unpacking Piper failed. The download is kept at %~dp0piper_windows_amd64.zip
            set "PROBLEMS=1"
        )
    )
)

set "HAVE_VOICE=0"
if exist "%ROOT%\voices\*.onnx" set "HAVE_VOICE=1"
if "!HAVE_VOICE!"=="1" (
    echo   voice ^(voices\*.onnx^): found
) else (
    if not exist "%ROOT%\voices" mkdir "%ROOT%\voices"
    set "VOICE_BASE=https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"
    call :download "!VOICE_BASE!/en_US-amy-medium.onnx"      "%ROOT%\voices\en_US-amy-medium.onnx"      50000000 "voice en_US-amy-medium, 63 MB"
    call :download "!VOICE_BASE!/en_US-amy-medium.onnx.json" "%ROOT%\voices\en_US-amy-medium.onnx.json" 1000     "voice settings"
)

REM ---------------------------------------------------------------
echo.
echo [4/4] Ollama ^(the language model^)
set "OLLAMA="
where ollama >nul 2>&1 && set "OLLAMA=ollama"
if not defined OLLAMA if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not defined OLLAMA (
    if "!HAVE_WINGET!"=="1" (
        echo   Installing Ollama...
        winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    ) else (
        echo   MISSING: Ollama - install it from https://ollama.com
        set "PROBLEMS=1"
    )
)
if defined OLLAMA (
    echo   Pulling phi3:mini ^(skipped if you already have it^)...
    "!OLLAMA!" pull phi3:mini
    if errorlevel 1 (
        echo   Couldn't pull the model. Start Ollama ^(open the Ollama app^), then run:  ollama pull phi3:mini
        set "PROBLEMS=1"
    )
)

REM ---------------------------------------------------------------
echo.
echo ============================================================
if "!PROBLEMS!"=="1" (
    echo  Setup finished with problems - see the messages above, fix them, and run setup.bat again.
) else (
    echo  Setup complete.
)
if "!NEWTOOLS!"=="1" (
    echo.
    echo  New tools were installed. CLOSE THIS WINDOW and open a new one before building,
    echo  so Windows picks them up.
)
echo.
echo  Next:  build.bat      ^(compiles dream.exe - a few minutes the first time^)
echo         build\dream.exe --selftest      ^(checks every part works^)
echo         build\dream.exe                 ^(runs DREAM^)
echo ============================================================
exit /b 0


REM ===============================================================
REM  :download URL DEST MIN_BYTES LABEL   - fetch with curl unless DEST already exists
REM ===============================================================
:download
set "DL_URL=%~1"
set "DL_DEST=%~2"
set "DL_MIN=%~3"
set "DL_LABEL=%~4"
if exist "!DL_DEST!" (
    echo   !DL_LABEL!: already downloaded
    exit /b 0
)
for %%D in ("%DL_DEST%") do if not exist "%%~dpD" mkdir "%%~dpD"
echo   Downloading !DL_LABEL! ...
curl -L -f -sS --retry 3 -o "%DL_DEST%.part" "%DL_URL%"
set "DL_SIZE=0"
if exist "%DL_DEST%.part" for %%S in ("%DL_DEST%.part") do set "DL_SIZE=%%~zS"
if !DL_SIZE! GEQ %DL_MIN% (
    move /y "%DL_DEST%.part" "%DL_DEST%" >nul
    echo   !DL_LABEL!: done
    exit /b 0
)
del "%DL_DEST%.part" >nul 2>&1
echo   PROBLEM: couldn't download !DL_LABEL!.
echo            Check your internet connection, or fetch it yourself: !DL_URL!
set "PROBLEMS=1"
exit /b 1
