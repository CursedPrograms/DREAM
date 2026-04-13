@echo off
setlocal enabledelayedexpansion

:: --- 1. SETTINGS ---
set "CheckpointsDir=models"
set "HF_ENDPOINT=https://hf-mirror.com"

:: --- 2. VIRTUAL ENV SETUP ---
echo [1/3] Setting up Python Virtual Environment...
if not exist "venv311" (
    py -3.11 -m venv venv
)
call venv311\Scripts\activate

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
:: --- 4. CLEAN DOWNLOADS ---
echo [3/3] Downloading weights...

:: MuseTalk (Core) - Just download the repo, exclude handled by default logic
hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%

:: SD-VAE - Directly specifying files, so no --include needed
hf download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir %CheckpointsDir%\sd-vae-ft-mse

:: Whisper Tiny
hf download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json --local-dir %CheckpointsDir%\whisper

:: DWPose
hf download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir %CheckpointsDir%\dwpose

:: SyncNet (LatentSync)
hf download ByteDance/LatentSync latentsync_syncnet.pt --local-dir %CheckpointsDir%\syncnet

echo.
echo ===================================================
echo DONE: All weights downloaded to .\%CheckpointsDir%\
echo ===================================================
pause