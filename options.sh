#!/bin/bash
# options.sh - the options menu for this bot (options.py edits config.json).
# Uses the bot's own venv when there is one, and installs pygame into it if missing.
cd "$(dirname "$0")" || exit 1

PY=""
for v in venv venv311 .venv scripts/.venv; do
    if [ -z "$PY" ] && [ -x "$v/bin/python" ]; then PY="$v/bin/python"; fi
done
[ -z "$PY" ] && PY="python3"

"$PY" -c "import pygame" 2>/dev/null || "$PY" -m pip install pygame
"$PY" options.py "$@" || read -r -p "Press Enter to close..."
