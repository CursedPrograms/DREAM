#!/bin/bash
set -e  # Stop on any error

# ============================================================
#  ComCentre - Master Start Script
# ============================================================

FRIDAY_DIR="/home/cursed/Desktop/ComCentre"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║          Starting ComCentre...       ║"
echo "╚══════════════════════════════════════╝"
echo ""

VENV_PYTHON="$FRIDAY_DIR/venv/bin/python"

# ---------------------------
# Trap: ensure hotspot stops on exit
# ---------------------------
trap 'echo ""; echo "[*] Stopping hotspot..."; bash "$FRIDAY_DIR/stop_hotspot.sh"' EXIT

# 1. Start hotspot
echo "[1/3] Bringing up WiFi hotspot..."
bash "$FRIDAY_DIR/hotspot.sh"

# 2. Start Ollama
echo "[2/3] Starting Ollama..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/dev/null &
    sleep 3
    echo "  → Ollama started"
else
    echo "  → Ollama already running"
fi

# 3. Start web server
echo "[3/3] Starting web server on http://192.168.4.1 ..."
cd "$FRIDAY_DIR"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ venv not found. Run setup first."
    exit 1
fi

# Optional: log output
LOG_FILE="$FRIDAY_DIR/server.log"
echo "📜 Logging webserver output to $LOG_FILE"
$VENV_PYTHON webserver.py | tee "$LOG_FILE"

# ---------------------------
# When webserver exits, EXIT trap triggers
# ---------------------------
echo ""
echo "[*] ComCentre session ended."
