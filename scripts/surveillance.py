#!/usr/bin/env python3
# surveillance.py

import threading
import time
import cv2
import os
from datetime import datetime
from rich.console import Console
import numpy as np

console = Console()
# Added a 'blind_mode' flag to state
_state = {"running": True, "enabled": True, "blind_mode": False}

def start_surveillance():
    """Starts the camera loop. If no camera is found, DREAM continues in blind mode."""
    def loop():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FOLDER_NAME = "eyes_on_you"
        SAVE_DIR = os.path.join(BASE_DIR, "output", FOLDER_NAME)
        os.makedirs(SAVE_DIR, exist_ok=True)

        # Try to capture camera (Using DirectShow for Windows stability)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            console.print("[yellow]Surveillance: No webcam detected. DREAM is now in 'Blind Mode'.[/yellow]")
            _state["enabled"] = False
            _state["blind_mode"] = True
            # We don't return; we let the thread finish naturally or wait for a signal
            return 

        console.print(f"[green]Surveillance started — saving to '{FOLDER_NAME}'[/green]")

        while _state["running"] and _state["enabled"]:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Surveillance: Capture failed. Retrying in 30s...[/red]")
                time.sleep(30)
                continue

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            console.print(f"[dim]Surveillance: {filename}[/dim]")

            # Check every 10 minutes, but check _state["running"] more often for responsiveness
            for _ in range(600): 
                if not _state["running"]: break
                time.sleep(1)

        cap.release()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_motion_detection():
    """Starts motion detection. If no camera is found, exits loop but keeps system alive."""
    def loop():
        if _state["blind_mode"]:
            return # Exit silently if surveillance already confirmed no camera

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        FOLDER_NAME = "motion_alerts"
        SAVE_DIR = os.path.join(BASE_DIR, "output", FOLDER_NAME)
        os.makedirs(SAVE_DIR, exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            console.print("[yellow]Motion detection: Disabled (No Camera).[/yellow]")
            return

        console.print(f"[green]Motion detection started — saving to '{FOLDER_NAME}'[/green]")

        ret, prev_frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

        while _state["running"]:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            if np.sum(thresh) > 10000:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(SAVE_DIR, f"motion_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                console.print(f"[bold red]Motion detected: {filename}[/bold red]")

            prev_gray = gray
            time.sleep(1)

        cap.release()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t