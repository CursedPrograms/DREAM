#!/bin/bash

set -e  # stop on error

echo "🐍 Creating venv with Python 3.12..."

python3.12 -m venv venv

echo "⚙️ Using venv Python..."
VENV_PYTHON="./venv/bin/python"
VENV_PIP="./venv/bin/pip"

echo "📦 Upgrading pip..."
$VENV_PYTHON -m pip install --upgrade pip

echo "📦 Installing requirements.txt..."
$VENV_PIP install -r requirements.txt

echo "📦 Installing additional packages..."
$VENV_PIP install openai-whisper piper-tts pathvalidate sounddevice soundfile numpy requests faster-whisper

# ---- OLLAMA INSTALL ----
if ! command -v ollama &> /dev/null
then
    echo "🧠 Installing Ollama..."
    sudo snap install ollama
else
    echo "✔ Ollama already installed"
fi

echo "🔍 Ollama version:"
ollama --version

echo "✅ Setup complete!"
