#!/usr/bin/env python3
"""
surveillance_live_feed.py
─────────────────────────
Live webcam surveillance with real-time AI analysis.

  • Opens webcam and displays annotated live feed in a window
  • Runs detection every N frames (configurable) to stay smooth
  • Motion-triggered saves: raw + annotated frame + JSON report
  • Press  Q  to quit,  S  to force-save current frame,  P  to pause

Usage:
    python3 surveillance_live_feed.py
    python3 surveillance_live_feed.py --camera 1       # use camera index 1
    python3 surveillance_live_feed.py --interval 5     # analyse every 5 frames
    python3 surveillance_live_feed.py --conf 0.4       # detection confidence
    python3 surveillance_live_feed.py --no-display     # headless / SSH mode
"""

import argparse
import os
import sys
import time
import threading
import cv2
import numpy as np
from datetime import datetime
from rich.console import Console

# surveillance_core.py must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from surveillance_core import analyse_frame, save_json, Models, console

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Smart Live Surveillance Feed")
    p.add_argument("--camera",     type=int,   default=0,    help="Camera device index (default: 0)")
    p.add_argument("--interval",   type=int,   default=10,   help="Analyse every N frames (default: 10)")
    p.add_argument("--conf",       type=float, default=0.45, help="Detection confidence threshold (default: 0.45)")
    p.add_argument("--no-display", action="store_true",      help="Headless mode — no window, just save on motion")
    p.add_argument("--motion-only",action="store_true",      help="Only save frames when motion is detected")
    p.add_argument("--output",     type=str,   default=None, help="Output directory (default: output/live_feed/)")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  MOTION DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class MotionDetector:
    def __init__(self, threshold=10_000, cooldown_sec=5):
        self.threshold   = threshold
        self.cooldown    = cooldown_sec
        self._prev_gray  = None
        self._last_alert = 0

    def update(self, frame):
        """Returns (motion_detected: bool, motion_score: int)."""
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, 0

        diff   = cv2.absdiff(self._prev_gray, gray)
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, 2)
        score  = int(np.sum(thresh))
        self._prev_gray = gray

        now = time.time()
        if score > self.threshold and (now - self._last_alert) > self.cooldown:
            self._last_alert = now
            return True, score

        return False, score

# ─────────────────────────────────────────────────────────────────────────────
#  ASYNC ANALYSER
#  Runs heavy inference in a background thread so the display stays smooth.
# ─────────────────────────────────────────────────────────────────────────────

class AsyncAnalyser:
    def __init__(self, conf):
        self.conf       = conf
        self._lock      = threading.Lock()
        self._pending   = None          # frame waiting for analysis
        self._result    = (None, None)  # (annotated, report)
        self._thread    = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._pending = frame.copy()

    def latest(self):
        with self._lock:
            return self._result

    def _worker(self):
        while True:
            frame = None
            with self._lock:
                if self._pending is not None:
                    frame = self._pending
                    self._pending = None
            if frame is not None:
                try:
                    annotated, report = analyse_frame(frame, "live_feed")
                    with self._lock:
                        self._result = (annotated, report)
                except Exception as e:
                    console.print(f"[red]Analysis error: {e}[/red]")
            else:
                time.sleep(0.02)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR   = args.output or os.path.join(BASE_DIR, "output", "live_feed")
    JSON_DIR  = os.path.join(OUT_DIR, "json")
    os.makedirs(OUT_DIR,  exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)

    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   Smart Surveillance — Live Feed     ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════╝[/bold cyan]")
    console.print()
    console.print(f"  Camera     : {args.camera}")
    console.print(f"  Interval   : every {args.interval} frames")
    console.print(f"  Confidence : {args.conf}")
    console.print(f"  Output     : {OUT_DIR}")
    console.print(f"  Display    : {'OFF (headless)' if args.no_display else 'ON'}")
    console.print()

    # Pre-load models before opening camera
    Models.get()
    console.print()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        console.print(f"[red]Cannot open camera {args.camera}[/red]")
        sys.exit(1)

    # Try to set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    console.print(f"[green]Camera opened at {actual_w}×{actual_h}[/green]")
    console.print("[dim]Controls: Q = quit  |  S = save frame  |  P = pause[/dim]")
    console.print()

    motion_det = MotionDetector()
    analyser   = AsyncAnalyser(args.conf)

    frame_count  = 0
    paused       = False
    display_frame = None   # last annotated frame for window

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    console.print("[red]Camera read failed — exiting.[/red]")
                    break

                frame_count += 1
                motion_detected, motion_score = motion_det.update(frame)

                # Submit to background analyser on interval or motion
                if frame_count % args.interval == 0 or motion_detected:
                    analyser.submit(frame)

                # Get latest analysis result (may be from a previous frame)
                annotated, report = analyser.latest()

                # Save on motion (or every frame if not motion-only)
                should_save = motion_detected or (not args.motion_only and frame_count % 300 == 0)

                if should_save and report is not None:
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    prefix = "motion_" if motion_detected else "periodic_"
                    raw_path  = os.path.join(OUT_DIR, f"{prefix}{ts}.jpg")
                    ann_path  = os.path.join(OUT_DIR, f"{prefix}{ts}_annotated.jpg")
                    cv2.imwrite(raw_path, frame)
                    if annotated is not None:
                        cv2.imwrite(ann_path, annotated)
                    json_path = save_json(report, JSON_DIR, prefix=f"{prefix}{ts}_")
                    trigger = "[bold red]MOTION[/bold red]" if motion_detected else "[dim]periodic[/dim]"
                    console.print(f"  {trigger} saved → {os.path.basename(raw_path)}")
                    console.print(f"         scene: {report.get('scene_summary', '')}")
                    console.print(f"         json : {os.path.basename(json_path)}")

                # Build display frame
                if not args.no_display:
                    display_frame = annotated if annotated is not None else frame
                    # Status bar at bottom
                    bar = display_frame.copy()
                    h_bar = 24
                    cv2.rectangle(bar, (0, bar.shape[0]-h_bar), (bar.shape[1], bar.shape[0]), (20,20,20), -1)
                    status = (f"Frame {frame_count}  |  Motion: {'YES' if motion_detected else 'no'} "
                              f"(score {motion_score:,})  |  {'PAUSED' if paused else 'LIVE'}")
                    cv2.putText(bar, status, (8, bar.shape[0]-7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)
                    display_frame = bar

            # Display
            if not args.no_display and display_frame is not None:
                cv2.imshow("Smart Surveillance — Live Feed  [Q=quit  S=save  P=pause]", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    console.print("[yellow]Q pressed — shutting down.[/yellow]")
                    break
                elif key == ord('p'):
                    paused = not paused
                    console.print(f"[yellow]{'Paused' if paused else 'Resumed'}[/yellow]")
                elif key == ord('s') and not paused:
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    raw_path = os.path.join(OUT_DIR, f"manual_{ts}.jpg")
                    cv2.imwrite(raw_path, frame)
                    if report:
                        ann_path = os.path.join(OUT_DIR, f"manual_{ts}_annotated.jpg")
                        if annotated is not None:
                            cv2.imwrite(ann_path, annotated)
                        save_json(report, JSON_DIR, prefix=f"manual_{ts}_")
                    console.print(f"[green]Manual save: {os.path.basename(raw_path)}[/green]")
            else:
                time.sleep(0.03)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — shutting down.[/yellow]")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        console.print("[green]Done.[/green]")


if __name__ == "__main__":
    main()
