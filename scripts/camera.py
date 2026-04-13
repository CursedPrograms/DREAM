#!/usr/bin/env python3
# camera.py

import cv2
import os
import sys
from datetime import datetime
from rich.console import Console
import time

console = Console()

# ── Platform ───────────────────────────────────────────────────────────────────
IS_WINDOWS = sys.platform == "win32"

_state = {"running": True, "enabled": True}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "output", "i_was_asked_to_take_this")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Camera backend ─────────────────────────────────────────────────────────────
# On Windows, CAP_DSHOW avoids a long timeout when no camera is present.
# On Linux, the default backend (V4L2) works fine.
_backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
cap = cv2.VideoCapture(0, _backend)

if not cap.isOpened():
    console.print("[yellow]Webcam not detected — surveillance disabled[/yellow]")
    _state["enabled"] = False

# ── Video codec ────────────────────────────────────────────────────────────────
# XVID works on both platforms but mp4v + .mp4 is more universally playable.
# Use mp4v on Windows (better media player support), XVID on Linux.
if IS_WINDOWS:
    VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    VIDEO_EXT    = ".mp4"
else:
    VIDEO_FOURCC = cv2.VideoWriter_fourcc(*'XVID')
    VIDEO_EXT    = ".avi"


def take_picture():
    """Capture a single photo and save it."""
    if not _state["enabled"]:
        console.print("[red]Cannot take picture — webcam disabled[/red]")
        return None

    ret, frame = cap.read()
    if not ret:
        console.print("[red]Capture failed[/red]")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename  = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    console.print(f"[green]Photo saved:[/green] {filename}")
    return filename


def take_video(duration=10):
    """Record video for `duration` seconds."""
    if not _state["enabled"]:
        console.print("[red]Cannot take video — webcam disabled[/red]")
        return None

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename     = os.path.join(SAVE_DIR, f"video_{timestamp}{VIDEO_EXT}")

    out = cv2.VideoWriter(filename, VIDEO_FOURCC, 20.0, (frame_width, frame_height))

    console.print(f"[green]Recording video for {duration} seconds...[/green]")

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            console.print("[red]Frame capture failed[/red]")
            break
        out.write(frame)
        time.sleep(0.05)  # ~20 FPS

    out.release()
    console.print(f"[green]Video saved:[/green] {filename}")
    return filename


def release_camera():
    """Release the webcam when done."""
    cap.release()
    console.print("[dim]Camera released[/dim]")


if __name__ == "__main__":
    take_picture()
    take_video(5)
    release_camera()