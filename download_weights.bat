@echo off
setlocal enabledelayedexpansion

set "CheckpointsDir=models"
set "HF_ENDPOINT=https://hf-mirror.com"
set "HF_HUB_ENABLE_HF_TRANSFER=1"

echo [1/3] Setting up Python Virtual Environment...
if not exist "venv311" (
    py -3.11 -m venv venv311
)
call venv311\Scripts\activate

python -m pip install --upgrade pip
pip install -U "huggingface_hub[hf_xet]"

echo [2/3] Preparing folders...
for %%d in (musetalk, syncnet, dwpose, face-parse-bisent, sd-vae-ft-mse, whisper) do (
    if not exist "%CheckpointsDir%\%%d" mkdir "%CheckpointsDir%\%%d"
)

echo [3/3] Downloading weights...

hf download TMElyralab/MuseTalk --local-dir %CheckpointsDir%\musetalk ^
    --include "*.pth" "*.pt" "*.json" "*.yaml"

hf download stabilityai/sd-vae-ft-mse --local-dir %CheckpointsDir%\sd-vae ^
    --include "config.json" "diffusion_pytorch_model.bin"

hf download openai/whisper-tiny --local-dir %CheckpointsDir%\whisper ^
    --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

hf download yzd-v/DWPose --local-dir %CheckpointsDir%\dwpose ^
    --include "dw-ll_ucoco_384.pth"

hf download ByteDance/LatentSync --local-dir %CheckpointsDir%\syncnet ^
    --include "latentsync_syncnet.pt"

hf download ManyOtherFunctions/face-parse-bisent --local-dir %CheckpointsDir%\face-parse-bisent ^
    --include "79999_iter.pth" "resnet18-5c106cde.pth"

echo.
echo ===================================================
echo DONE: All weights downloaded to .\%CheckpointsDir%\
echo ===================================================
pause