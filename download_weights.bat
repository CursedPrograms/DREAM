@echo off
REM download_weights.bat - downloads everything DREAM needs that isn't in git.
REM Safe to run again: anything already there is skipped. Each download is retried,
REM then tried through the hf-mirror.com mirror, before it counts as failed.
REM   1. MuseTalk lip-sync weights   -> models\
REM   2. Whisper speech model        -> the one WhisperModel in config.json names (pick it in options.py)
REM   3. Piper voice                 -> voices\ (Voice in config.json)
REM   4. ffmpeg                      -> installed with winget if missing
REM   5. Ollama models               -> LlmModel, VisionModel, and Memory's LlmModel if memory is on
REM                                     (if Ollama is installed); and the face models (YuNet + SFace) -> models\faces
REM   6. EMOCA 3D head test          -> emoca_test\ (optional, asks first: non-commercial licence)
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "VENV=%~dp0venv311"
set "PY=%VENV%\Scripts\python.exe"
set "HF=%VENV%\Scripts\hf.exe"
set "M=models"
set "TAR=%SystemRoot%\System32\tar.exe"
set "CURL=%SystemRoot%\System32\curl.exe"
set "FAILED="

echo.
echo [1/6] Lip-sync weights (MuseTalk)
if not exist "%PY%" (
    echo   Creating venv311 ...
    py -3.11 -m venv "%VENV%" || goto :nopython
)
if not exist "%PY%" goto :nopython
if not exist "%HF%" (
    echo   Installing the Hugging Face downloader ...
    "%PY%" -m pip install -q "huggingface_hub[hf_xet]"
)
if not exist "%HF%" (
    echo   Could not install huggingface_hub - skipping the lip-sync weights
    set "FAILED=!FAILED!, lip-sync weights"
    goto :whisper
)
call :hf "MuseTalk v1.5"  TMElyralab/MuseTalk                  "%M%"                   musetalkV15/unet.pth musetalkV15/musetalk.json
call :hf "MuseTalk v1.0"  TMElyralab/MuseTalk                  "%M%"                   musetalk/pytorch_model.bin musetalk/musetalk.json
call :hf "SD VAE"         stabilityai/sd-vae-ft-mse            "%M%\sd-vae"            config.json diffusion_pytorch_model.bin
call :hf "Whisper (lip-sync)" openai/whisper-tiny                "%M%\whisper"           config.json pytorch_model.bin preprocessor_config.json
call :hf "DWPose"         yzd-v/DWPose                         "%M%\dwpose"            dw-ll_ucoco_384.pth
call :hf "SyncNet"        ByteDance/LatentSync                 "%M%\syncnet"           latentsync_syncnet.pt
call :hf "Face parsing"   ManyOtherFunctions/face-parse-bisent "%M%\face-parse-bisent" 79999_iter.pth resnet18-5c106cde.pth

:whisper
echo.
echo [2/6] Whisper speech model (WhisperModel in config.json)
"%PY%" scripts\whisper_model.py --download
if errorlevel 1 set "FAILED=!FAILED!, Whisper"
REM dream.exe (cpp_dream) uses whisper.cpp's copy of the same model
if exist "cpp_dream\build\dream.exe" (
    for /f "delims=" %%G in ('""%PY%" scripts\whisper_model.py --ggml"') do set "GGML=%%G"
    if defined GGML call :get "Whisper for dream.exe (!GGML!)" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/!GGML!" "cpp_dream\models\!GGML!"
)

:voice
echo.
echo [3/6] Piper voice (Voice in config.json)
"%PY%" scripts\dream_models.py --download-voice
if errorlevel 1 set "FAILED=!FAILED!, voice"

echo.
echo [4/6] ffmpeg
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    echo   ffmpeg: already installed
    goto :ollama
)
where winget >nul 2>&1
if errorlevel 1 (
    echo   ffmpeg: winget not found - install ffmpeg from https://ffmpeg.org/download.html
    set "FAILED=!FAILED!, ffmpeg"
    goto :ollama
)
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo   ffmpeg: FAILED
    set "FAILED=!FAILED!, ffmpeg"
) else (
    echo   ffmpeg: installed - open a new terminal so it is on PATH
)

:ollama
echo.
echo [5/6] Ollama models (LlmModel and VisionModel in config.json, and Memory's if memory is on)
echo   Face recognition models:
"%PY%" -c "import sys; sys.path.insert(0, 'scripts'); from dream_mind.faces import download_models; sys.exit(0 if download_models() else 1)"
if errorlevel 1 set "FAILED=!FAILED!, face models"
where ollama >nul 2>&1
if errorlevel 1 (
    echo   Ollama is not installed - get it from https://ollama.com, then run this again
    set "FAILED=!FAILED!, Ollama"
    goto :emoca
)
set "LLMS="
for /f "delims=" %%L in ('""%PY%" scripts\dream_models.py --llm"') do set "LLMS=%%L"
for /f "delims=" %%L in ('""%PY%" scripts\dream_models.py --vision"') do if /i not "%%L"=="!LLMS!" set "LLMS=!LLMS! %%L"
for /f "delims=" %%L in ('""%PY%" scripts\dream_models.py --memory-llm"') do if /i not "%%L"=="!LLMS!" set "LLMS=!LLMS! %%L"
for %%L in (!LLMS!) do call :pull %%L

:emoca
echo.
echo [6/6] EMOCA 3D head test (optional)
set "EM=emoca_test"
if exist "%EM%\emoca\gdl" if exist "%EM%\assets\FLAME\geometry\generic_model.pkl" if exist "%EM%\assets\EMOCA\models\EMOCA_v2_lr_mse_20\cfg.yaml" if exist "%EM%\assets\ear\ear_landmarker.pth" (
    echo   EMOCA: already there
    goto :done
)
echo   EMOCA and FLAME are free for non-commercial research only. Register and accept
echo   their licences first: https://flame.is.tue.mpg.de and https://emoca.is.tue.mpg.de
echo   The ear landmarker (from flame-head-tracker, i-Bug ears) is research use only too.
choice /c YN /t 60 /d N /m "  Download them now, about 600 MB"
if errorlevel 2 (
    echo   EMOCA: skipped
    goto :done
)
if not exist "%EM%\emoca\gdl" (
    where git >nul 2>&1 && git clone --depth 1 https://github.com/radekd91/emoca "%EM%\emoca"
)
if not exist "%EM%\emoca\gdl" if not exist "%EM%\emoca" (
    call :get "EMOCA code" "https://codeload.github.com/radekd91/emoca/zip/refs/heads/release/EMOCA_v2" "%EM%\emoca.zip"
    if exist "%EM%\emoca.zip" (
        "%TAR%" -xf "%EM%\emoca.zip" -C "%EM%"
        if exist "%EM%\emoca-release-EMOCA_v2" move "%EM%\emoca-release-EMOCA_v2" "%EM%\emoca" >nul
    )
)
if not exist "%EM%\emoca\gdl" (
    echo   EMOCA code: FAILED - if %EM%\emoca is a broken download, delete it and run this again
    set "FAILED=!FAILED!, EMOCA code"
)
call :zip "FLAME head model" "https://download.is.tue.mpg.de/emoca/assets/FLAME.zip"                            "%EM%\assets\FLAME.zip"                                  "%EM%\assets"               "%EM%\assets\FLAME\geometry\generic_model.pkl"
call :zip "EMOCA model"      "https://download.is.tue.mpg.de/emoca/assets/EMOCA/models/EMOCA_v2_lr_mse_20.zip" "%EM%\assets\EMOCA\models\EMOCA_v2_lr_mse_20.zip"   "%EM%\assets\EMOCA\models" "%EM%\assets\EMOCA\models\EMOCA_v2_lr_mse_20\cfg.yaml"
call :get "Ear landmarker"   "https://github.com/PeizhiYan/flame-head-tracker/releases/download/resource/ear_landmarker.pth" "%EM%\assets\ear\ear_landmarker.pth"

:done
echo.
if defined FAILED (
    echo Not everything finished:!FAILED:~1!
    echo Run download_weights.bat again to retry - anything already downloaded is skipped.
    pause
    exit /b 1
)
echo Everything is downloaded.
pause
exit /b 0


REM ---------------------------------------------------------------- helpers

REM :hf "name" repo local-dir file...   (Hugging Face; retried, then via the mirror)
:hf
set "NAME=%~1"
set "REPO=%~2"
set "DIR=%~3"
set "FILES="
shift
shift
shift
:hf_args
if "%~1"=="" goto :hf_go
set "FILES=!FILES! %~1"
shift
goto :hf_args
:hf_go
call :hf_missing
if not defined MISSING (
    echo   %NAME%: already there
    exit /b 0
)
echo   %NAME%: downloading ...
for %%E in (https://huggingface.co https://huggingface.co https://hf-mirror.com) do (
    call :hf_missing
    if defined MISSING (
        set "HF_ENDPOINT=%%E"
        "%HF%" download %REPO% !FILES! --local-dir "%DIR%"
    )
)
set "HF_ENDPOINT="
call :hf_missing
if defined MISSING (
    echo   %NAME%: FAILED, still missing:!MISSING!
    set "FAILED=!FAILED!, %NAME%"
) else (
    echo   %NAME%: OK
)
exit /b 0

:hf_missing
set "MISSING="
for %%F in (!FILES!) do (
    set "F=%%F"
    if not exist "%DIR%\!F:/=\!" set "MISSING=!MISSING! %%F"
)
exit /b 0

REM :pull model   (Ollama; starts the server and retries once if the pull fails)
:pull
ollama list 2>nul | findstr /b /i /c:"%~1 " /c:"%~1:latest " >nul
if not errorlevel 1 (
    echo   %~1: already there
    exit /b 0
)
ollama pull %~1
if errorlevel 1 (
    echo   Starting the Ollama server and trying again ...
    start "Ollama" /min ollama serve
    timeout /t 5 /nobreak >nul
    ollama pull %~1
)
if errorlevel 1 (
    echo   %~1: FAILED
    set "FAILED=!FAILED!, %~1"
)
exit /b 0

REM :get "name" url out-file   (curl, resumes a broken download)
:get
if exist "%~3" (
    echo   %~1: already there
    exit /b 0
)
if not exist "%~dp3" mkdir "%~dp3"
echo   %~1: downloading ...
"%CURL%" -fL --retry 3 --retry-delay 5 -C - -o "%~3.part" "%~2"
if not errorlevel 1 move /y "%~3.part" "%~3" >nul
if exist "%~3" (
    echo   %~1: OK
) else (
    echo   %~1: FAILED
    set "FAILED=!FAILED!, %~1"
)
exit /b 0

REM :zip "name" url zip-file unpack-into check-file
:zip
if exist "%~5" (
    echo   %~1: already there
    exit /b 0
)
if not exist "%~4" mkdir "%~4"
call :get "%~1" "%~2" "%~3"
if not exist "%~3" exit /b 0
"%TAR%" -xf "%~3" -C "%~4"
if exist "%~5" (
    echo   %~1: unpacked
) else (
    echo   %~1: could not unpack %~3
    set "FAILED=!FAILED!, %~1"
)
exit /b 0

:nopython
echo Python 3.11 was not found. Install it from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH"), then run this again.
pause
exit /b 1
