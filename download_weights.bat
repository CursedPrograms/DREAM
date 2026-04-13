@echo off
setlocal enabledelayedexpansion

:: --- 1. SETTINGS ---
set "CheckpointsDir=models"
set "HF_ENDPOINT=https://hf-mirror.com"

:: --- 2. VIRTUAL ENV SETUP ---
echo [1/3] Setting up Python Virtual Environment...
if not exist "venv" (
    py -3.11 -m venv venv
)
call venv\Scripts\activate

:: Ensure we have the latest downloader tools
python -m pip install --upgrade pip
pip install -U "huggingface_hub[hf_xet]"

:: --- 3. DIRECTORY CREATION ---
echo [2/3] Preparing folders...
for %%d in (musetalk, syncnet, dwpose, face-parse-bisent, sd-vae-ft-mse, whisper) do (
    if not exist "%CheckpointsDir%\%%d" mkdir "%CheckpointsDir%\%%d"
)

:: --- 4. CLEAN DOWNLOADS ---
echo [3/3] Downloading weights (excluding metadata files)...

:: Use --exclude to block .md, .gitignore, and .gitattributes
:: Use --local-dir-use-symlinks False to ensure actual files are downloaded, not shortcuts

:: MuseTalk (Core)
huggingface-cli download TMElyralab/MuseTalk --local-dir %CheckpointsDir% --exclude "*.md" ".gitignore" ".gitattributes" --local-dir-use-symlinks False

:: SD-VAE
huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae-ft-mse --include "config.json" "diffusion_pytorch_model.bin" --local-dir-use-symlinks False

:: Whisper Tiny
huggingface-cli download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper --include "config.json" "pytorch_model.bin" "preprocessor_config.json" --local-dir-use-symlinks False

:: DWPose
huggingface-cli download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose --include "dw-ll_ucoco_384.pth" --local-dir-use-symlinks False

:: SyncNet (LatentSync)
huggingface-cli download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet --include "latentsync_syncnet.pt" --local-dir-use-symlinks False

echo.
echo ===================================================
echo DONE: All weights downloaded to .\%CheckpointsDir%\
echo ===================================================
pause