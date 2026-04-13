#!/usr/bin/env python3
"""
smart_surveillance.py
─────────────────────
Fully OFFLINE smart surveillance system.
• Motion detection triggers frame capture
• Each captured frame is analysed locally using OpenCV Haar cascades
• Detected humans get bounding boxes + heuristic estimates (age range, gender, skin tone)
• Detected cats/animals get labelled bounding boxes
• Every analysed frame is saved as:
    output/motion_alerts/<timestamp>.jpg          ← raw frame
    output/motion_alerts/<timestamp>_annotated.jpg ← frame with drawn boxes
    output/motion_alerts/json/<timestamp>.json     ← full analysis report

NOTE ON ESTIMATES
  Age, gender, and skin-tone estimates are produced by lightweight heuristics
  (face-region brightness, eye spacing, contour ratios).  They are rough
  approximations — not deep-learning predictions — and should be treated as
  indicative only.
"""

import threading
import time
import cv2
import os
import json
import math
import numpy as np
from datetime import datetime
from rich.console import Console

console = Console()
_state = {"running": True, "enabled": True}

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD CLASSIFIERS  (all ship with opencv-python — 100 % offline)
# ─────────────────────────────────────────────────────────────────────────────
CASCADE_DIR = cv2.data.haarcascades

def _load(name):
    path = os.path.join(CASCADE_DIR, name)
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        console.print(f"[yellow]Warning: could not load {name}[/yellow]")
    return clf

face_cascade      = _load("haarcascade_frontalface_default.xml")
profile_cascade   = _load("haarcascade_profileface.xml")
eye_cascade       = _load("haarcascade_eye.xml")
fullbody_cascade  = _load("haarcascade_fullbody.xml")
upperbody_cascade = _load("haarcascade_upperbody.xml")
cat_cascade       = _load("haarcascade_frontalcatface_extended.xml")

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE FOR ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────
COL_HUMAN  = (0,   255,  0)    # green  – humans
COL_FACE   = (255, 200,  0)    # amber  – face box
COL_ANIMAL = (0,   200, 255)   # cyan   – animals
COL_BODY   = (180,  0,  255)   # purple – body silhouette
COL_TEXT_BG= (0,   0,    0)    # black label bg
FONT       = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────────────────────────────
#  HEURISTIC ESTIMATORS
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_age_range(face_roi_gray: np.ndarray, face_w: int) -> str:
    """
    Very rough age bracket from face-region texture variance.
    High variance → more wrinkles/texture → older.
    Low variance + small box → child.
    """
    if face_roi_gray is None or face_roi_gray.size == 0:
        return "unknown"
    variance = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
    if face_w < 60:
        return "child (est. <14)"
    if variance < 40:
        return "young adult (est. 15-30)"
    if variance < 120:
        return "adult (est. 30-50)"
    return "older adult (est. 50+)"


def _estimate_gender(face_roi_gray: np.ndarray, face_w: int, face_h: int) -> str:
    """
    Heuristic: count eye detections + face aspect ratio.
    Wider faces relative to height lean male in aggregate statistics.
    Two clear eyes detected → higher confidence.
    This is a rough proxy — accuracy is limited without a trained model.
    """
    if face_roi_gray is None or face_roi_gray.size == 0:
        return "unknown"
    aspect = face_w / max(face_h, 1)
    eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(15, 15))
    eye_count = len(eyes)
    # Wider + eyes clearly detected
    if aspect > 0.88 and eye_count >= 2:
        return "likely male"
    if aspect < 0.82 and eye_count >= 1:
        return "likely female"
    return "indeterminate"


def _estimate_skin_tone(face_roi_bgr: np.ndarray) -> str:
    """
    Classifies average skin luminance from the YCrCb colour space into
    broad descriptive bands.  This is a colorimetric measurement, not
    ethnicity classification.
    """
    if face_roi_bgr is None or face_roi_bgr.size == 0:
        return "unknown"
    ycrcb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2YCrCb)
    y_mean = float(np.mean(ycrcb[:, :, 0]))   # luminance
    cr_mean = float(np.mean(ycrcb[:, :, 1]))  # red chrominance

    if y_mean > 180 and cr_mean > 148:
        return "very light"
    if y_mean > 140 and cr_mean > 140:
        return "light"
    if y_mean > 100 and cr_mean > 130:
        return "medium"
    if y_mean > 70:
        return "medium-dark"
    return "dark"


def _body_posture(bw: int, bh: int) -> str:
    """Guess posture from bounding-box aspect ratio."""
    ratio = bh / max(bw, 1)
    if ratio > 2.5:
        return "standing"
    if ratio > 1.2:
        return "crouching or sitting"
    return "lying / crawling"


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_frame(frame: np.ndarray, image_path: str) -> dict:
    """
    Full offline analysis of a single BGR frame.
    Returns a structured dict and writes an annotated image alongside the original.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated = frame.copy()

    result = {
        "timestamp": datetime.now().isoformat(),
        "source_image": os.path.basename(image_path),
        "frame_size": {"width": w, "height": h},
        "humans": [],
        "animals": [],
        "scene_summary": "",
        "threat_level": "none",
    }

    # ── 1. FULL-BODY DETECTION ───────────────────────────────────────────────
    bodies = fullbody_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 120))
    upper_bodies = upperbody_cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 60))

    # Merge full-body and upper-body hits, dedup by overlap
    all_bodies = list(bodies) + list(upper_bodies)

    # ── 2. FACE DETECTION (frontal + profile) ────────────────────────────────
    faces_front = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    faces_profile = profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    all_faces = list(faces_front) + list(faces_profile)

    # ── 3. BUILD HUMAN RECORDS ───────────────────────────────────────────────
    # Each detected face becomes a human record; if a body box exists nearby we attach it
    human_id = 1
    used_body_indices = set()

    for (fx, fy, fw, fh) in all_faces:
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        face_bgr  = frame[fy:fy+fh, fx:fx+fw]

        age_range  = _estimate_age_range(face_gray, fw)
        gender     = _estimate_gender(face_gray, fw, fh)
        skin_tone  = _estimate_skin_tone(face_bgr)

        # Find best-matching body box
        body_box = None
        best_overlap = 0
        for i, (bx, by, bw, bh) in enumerate(all_bodies):
            if i in used_body_indices:
                continue
            # Face should lie inside or near the top of the body box
            cx = fx + fw // 2
            cy = fy + fh // 2
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                overlap = fw * fh  # face is inside → use it
                if overlap > best_overlap:
                    best_overlap = overlap
                    body_box = (bx, by, bw, bh)
                    best_body_idx = i

        if body_box and best_overlap > 0:
            used_body_indices.add(best_body_idx)
            bx, by, bw, bh = body_box
            posture = _body_posture(bw, bh)
        else:
            bx, by, bw, bh = fx, fy, fw, fh * 3  # extrapolate body from face
            posture = "partial view"

        # Draw body bounding box
        cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), COL_HUMAN, 2)
        # Draw face box
        cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), COL_FACE, 2)

        # Label
        label_lines = [
            f"Person #{human_id}",
            f"Age: {age_range}",
            f"Gender: {gender}",
            f"Skin: {skin_tone}",
            f"Posture: {posture}",
        ]
        _draw_label_block(annotated, bx, by, label_lines, COL_HUMAN)

        result["humans"].append({
            "id": human_id,
            "age_estimate": age_range,
            "gender_estimate": gender,
            "skin_tone": skin_tone,
            "posture": posture,
            "face_bounding_box": {"x": int(fx), "y": int(fy), "w": int(fw), "h": int(fh)},
            "body_bounding_box": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
        })
        human_id += 1

    # Bodies without a matched face (partial detections)
    for i, (bx, by, bw, bh) in enumerate(all_bodies):
        if i in used_body_indices:
            continue
        posture = _body_posture(bw, bh)
        cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), COL_BODY, 2)
        _draw_label_block(annotated, bx, by, [f"Person #{human_id}", "face not visible", f"Posture: {posture}"], COL_BODY)
        result["humans"].append({
            "id": human_id,
            "age_estimate": "unknown (face not visible)",
            "gender_estimate": "unknown",
            "skin_tone": "unknown",
            "posture": posture,
            "face_bounding_box": None,
            "body_bounding_box": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
        })
        human_id += 1

    # ── 4. CAT / ANIMAL DETECTION ────────────────────────────────────────────
    cats = cat_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    animal_id = 1
    for (ax, ay, aw, ah) in cats:
        cv2.rectangle(annotated, (ax, ay), (ax+aw, ay+ah), COL_ANIMAL, 2)
        label = [f"Animal #{animal_id}", "Species: Cat", "Behaviour: present"]
        _draw_label_block(annotated, ax, ay, label, COL_ANIMAL)
        result["animals"].append({
            "id": animal_id,
            "species": "Cat (Felis catus)",
            "detection_method": "Haar cascade (frontal face)",
            "behaviour": "stationary or approaching camera",
            "bounding_box": {"x": int(ax), "y": int(ay), "w": int(aw), "h": int(ah)},
        })
        animal_id += 1

    # ── 5. SCENE SUMMARY ─────────────────────────────────────────────────────
    n_humans = len(result["humans"])
    n_animals = len(result["animals"])
    parts = []
    if n_humans:
        parts.append(f"{n_humans} person(s) detected")
    if n_animals:
        parts.append(f"{n_animals} animal(s) detected")
    if not parts:
        parts.append("no persons or animals detected")

    result["scene_summary"] = "; ".join(parts)
    result["threat_level"] = "low" if n_humans > 0 else "none"

    # ── 6. SAVE ANNOTATED IMAGE ──────────────────────────────────────────────
    base, ext = os.path.splitext(image_path)
    annotated_path = base + "_annotated" + ext
    cv2.imwrite(annotated_path, annotated)
    result["annotated_image"] = os.path.basename(annotated_path)

    return result


def _draw_label_block(img, x, y, lines, colour):
    """Draw a filled label block above a bounding box."""
    scale = 0.42
    thickness = 1
    pad = 3
    line_h = 14
    block_h = len(lines) * line_h + pad * 2
    block_w = max((len(l) for l in lines), default=10) * 7 + pad * 2
    # Clamp to image
    lx = max(x, 0)
    ly = max(y - block_h, 0)
    cv2.rectangle(img, (lx, ly), (lx + block_w, ly + block_h), COL_TEXT_BG, -1)
    for i, line in enumerate(lines):
        ty = ly + pad + (i + 1) * line_h - 2
        cv2.putText(img, line, (lx + pad, ty), FONT, scale, colour, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  JSON WRITER
# ─────────────────────────────────────────────────────────────────────────────

def save_json_report(analysis: dict, json_dir: str):
    os.makedirs(json_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    path = os.path.join(json_dir, f"{ts}.json")
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    console.print(f"[cyan]  JSON report → {path}[/cyan]")
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  SURVEILLANCE LOOP  (periodic snapshot every 10 min)
# ─────────────────────────────────────────────────────────────────────────────

def start_surveillance():
    """Starts the camera loop in a background thread that saves + analyses images periodically."""
    def loop():
        BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
        SAVE_DIR  = os.path.join(BASE_DIR, "output", "eyes_on_you")
        JSON_DIR  = os.path.join(SAVE_DIR, "json")
        os.makedirs(SAVE_DIR, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            console.print("[yellow]Surveillance disabled: No webcam detected[/yellow]")
            _state["enabled"] = False
            return

        console.print("[green]Surveillance started — saving to 'eyes_on_you'[/green]")

        while _state["running"] and _state["enabled"]:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Capture failed[/red]")
                break

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            img_path  = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
            cv2.imwrite(img_path, frame)
            console.print(f"[dim]Surveillance frame: {img_path}[/dim]")

            analysis = analyse_frame(frame, img_path)
            save_json_report(analysis, JSON_DIR)
            _print_summary(analysis)

            time.sleep(600)

        cap.release()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  MOTION DETECTION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def start_motion_detection():
    """Starts a motion-detection webcam loop that analyses every triggered frame."""
    def loop():
        BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
        SAVE_DIR  = os.path.join(BASE_DIR, "output", "motion_alerts")
        JSON_DIR  = os.path.join(SAVE_DIR, "json")
        os.makedirs(SAVE_DIR, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            console.print("[yellow]Motion detection disabled: No webcam detected[/yellow]")
            return

        console.print("[green]Motion detection started — saving to 'motion_alerts'[/green]")

        ret, prev_frame = cap.read()
        if not ret:
            console.print("[red]Initial capture failed[/red]")
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        last_alert = 0  # prevent spamming on continuous motion

        while _state["running"]:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Capture failed[/red]")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

            diff   = cv2.absdiff(prev_gray, gray_blur)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_score = int(np.sum(thresh))

            if motion_score > 10_000 and (time.time() - last_alert) > 5:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                img_path  = os.path.join(SAVE_DIR, f"motion_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)

                console.print(f"[bold red]▶ Motion detected (score {motion_score:,}): {img_path}[/bold red]")

                # ── AI analysis ──────────────────────────────────────────────
                analysis = analyse_frame(frame, img_path)
                analysis["motion_score"] = motion_score
                json_path = save_json_report(analysis, JSON_DIR)
                _print_summary(analysis)

                last_alert = time.time()

            prev_gray = gray_blur
            time.sleep(1)

        cap.release()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


# ─────────────────────────────────────────────────────────────────────────────
#  RICH CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(analysis: dict):
    console.print(f"  [bold]Scene:[/bold] {analysis['scene_summary']}")
    console.print(f"  [bold]Threat:[/bold] {analysis['threat_level']}")

    for h in analysis["humans"]:
        bb = h["face_bounding_box"]
        bb_str = f"face@({bb['x']},{bb['y']})" if bb else "no face"
        console.print(
            f"  [green]👤 Person #{h['id']}[/green] | "
            f"Age: {h['age_estimate']} | "
            f"Gender: {h['gender_estimate']} | "
            f"Skin tone: {h['skin_tone']} | "
            f"Posture: {h['posture']} | {bb_str}"
        )

    for a in analysis["animals"]:
        bb = a["bounding_box"]
        console.print(
            f"  [cyan]🐾 Animal #{a['id']}[/cyan] | "
            f"Species: {a['species']} | "
            f"Behaviour: {a['behaviour']} | "
            f"box@({bb['x']},{bb['y']})"
        )

    if not analysis["humans"] and not analysis["animals"]:
        console.print("  [dim]No persons or animals detected in frame.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold cyan]Smart Surveillance System — fully offline[/bold cyan]")
    console.print("Press [bold]Ctrl+C[/bold] to stop.\n")

    t1 = start_surveillance()
    t2 = start_motion_detection()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down…[/yellow]")
        _state["running"] = False
        time.sleep(2)
        console.print("[green]Done.[/green]")
