#!/bin/bash

# Exit on error
set -e

# Set the checkpoints directory
CheckpointsDir="models"

# Create directories efficiently
mkdir -p "$CheckpointsDir"/{musetalk,musetalkV15,syncnet,dwpose,face-parse-bisent,sd-vae,whisper}

# Ensure downloader tools are present
pip install -U "huggingface_hub[cli]" gdown

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# --- Download Logic ---

# 1. MuseTalk (V1.0 & V1.5)
# Using --local-dir-use-symlinks False to ensure we get actual files
huggingface-cli download TMElyralab/MuseTalk \
  --local-dir "$CheckpointsDir" \
  --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin" "musetalkV15/musetalk.json" "musetalkV15/unet.pth" \
  --local-dir-use-symlinks False

# 2. SD VAE
huggingface-cli download stabilityai/sd-vae-ft-mse \
  --local-dir "$CheckpointsDir/sd-vae" \
  --include "config.json" "diffusion_pytorch_model.bin" \
  --local-dir-use-symlinks False

# 3. Whisper
huggingface-cli download openai/whisper-tiny \
  --local-dir "$CheckpointsDir/whisper" \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json" \
  --local-dir-use-symlinks False

# 4. DWPose
huggingface-cli download yzd-v/DWPose \
  --local-dir "$CheckpointsDir/dwpose" \
  --include "dw-ll_ucoco_384.pth" \
  --local-dir-use-symlinks False

# 5. SyncNet
huggingface-cli download ByteDance/LatentSync \
  --local-dir "$CheckpointsDir/syncnet" \
  --include "latentsync_syncnet.pt" \
  --local-dir-use-symlinks False

# 6. Face Parse (Google Drive & Direct Link)
echo "Downloading Face Parse weights..."
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$CheckpointsDir/face-parse-bisent/79999_iter.pth"
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o "$CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth"

echo "------------------------------------------------"
echo "✅ All weights have been downloaded successfully!"
echo "------------------------------------------------"