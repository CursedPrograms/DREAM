#!/usr/bin/env python3
"""
surveillance_core.py
────────────────────
Shared analysis engine for both surveillance_live_feed.py and surveillance_image_feed.py.

Model stack (all loaded from resources/weights/):
  ┌─────────────────────────────────────────────────────────┐
  │  Detection                                              │
  │    YOLOv8n (.pt)          — people, animals, vehicles  │
  │    MobileNetSSD (.caffe)  — fallback if YOLO missing   │
  │    YuNet (.onnx)          — precise face detection      │
  │    Haar cascades          — last-resort fallback        │
  │                                                         │
  │  Per-face analysis (ONNX via onnxruntime)               │
  │    age_googlenet.onnx     — 8 age brackets             │
  │    gender_googlenet.onnx  — Female / Male              │
  │    emotion-ferplus-8.onnx — 8 emotions                 │
  └─────────────────────────────────────────────────────────┘
"""

import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from datetime import datetime
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

def _find_weights_dir():
    """Walk up from this file looking for resources/weights."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in [here, os.path.dirname(here)]:
        candidate = os.path.join(base, "resources", "weights")
        if os.path.isdir(candidate):
            return candidate
    # Fallback: create it next to this file
    path = os.path.join(here, "resources", "weights")
    os.makedirs(path, exist_ok=True)
    return path

WEIGHTS_DIR = _find_weights_dir()

def W(filename):
    return os.path.join(WEIGHTS_DIR, filename)

CASCADE_DIR = cv2.data.haarcascades

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS & DRAWING
# ─────────────────────────────────────────────────────────────────────────────

COL = {
    "person":  (50,  220,  50),   # green
    "face":    (0,   200, 255),   # amber-yellow
    "animal":  (255, 180,   0),   # cyan-blue
    "vehicle": (180,   0, 255),   # purple
    "object":  (200, 200, 200),   # grey
    "unknown": (100, 100, 100),
}
FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

def draw_box(img, x1, y1, x2, y2, colour, label_lines, alpha=0.35):
    """Draws a semi-transparent filled rectangle + label block."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Semi-transparent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Solid border
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)

    # Label block above box
    scale, thick, pad, line_h = 0.45, 1, 4, 16
    block_w = max((len(l) for l in label_lines), default=1) * 8 + pad * 2
    block_h = len(label_lines) * line_h + pad * 2
    lx = x1
    ly = max(0, y1 - block_h)
    cv2.rectangle(img, (lx, ly), (lx + block_w, ly + block_h), (15, 15, 15), -1)
    cv2.rectangle(img, (lx, ly), (lx + block_w, ly + block_h), colour, 1)
    for i, line in enumerate(label_lines):
        ty = ly + pad + (i + 1) * line_h - 3
        cv2.putText(img, line, (lx + pad, ty), FONT, scale, colour, thick, cv2.LINE_AA)

def draw_hud(img, detections):
    """Top-right HUD with counts."""
    h, w = img.shape[:2]
    n_people  = sum(1 for d in detections if d["category"] == "person")
    n_animals = sum(1 for d in detections if d["category"] == "animal")
    n_other   = len(detections) - n_people - n_animals

    lines = [
        f"People : {n_people}",
        f"Animals: {n_animals}",
        f"Other  : {n_other}",
        datetime.now().strftime("%H:%M:%S"),
    ]
    pad, line_h = 6, 18
    bw = 160
    bh = len(lines) * line_h + pad * 2
    bx = w - bw - 8
    by = 8
    overlay = img.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (180, 180, 180), 1)
    for i, line in enumerate(lines):
        ty = by + pad + (i + 1) * line_h - 3
        cv2.putText(img, line, (bx + pad, ty), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER  (lazy, singleton)
# ─────────────────────────────────────────────────────────────────────────────

class Models:
    _instance = None

    def __init__(self):
        self.yolo         = None   # ultralytics YOLO
        self.ssd_net      = None   # cv2 DNN caffe MobileNetSSD
        self.yunet        = None   # cv2.FaceDetectorYN
        self.age_sess     = None   # onnxruntime session
        self.gender_sess  = None
        self.emotion_sess = None
        self.face_haar    = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml"))
        self.body_haar    = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_fullbody.xml"))
        self.upper_haar   = cv2.CascadeClassifier(os.path.join(CASCADE_DIR, "haarcascade_upperbody.xml"))
        self._load_all()

    def _load_all(self):
        # ── YOLOv8n ──────────────────────────────────────────────────────────
        yolo_path = W("yolov8n.pt")
        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_path)
                console.print("[green]  ✔ YOLOv8n loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]  ⚠ YOLOv8n failed: {e}[/yellow]")
        else:
            console.print(f"[yellow]  ⚠ yolov8n.pt not found — run download_weights.sh[/yellow]")

        # ── MobileNetSSD fallback ─────────────────────────────────────────────
        ssd_model  = W("MobileNetSSD_deploy.caffemodel")
        ssd_proto  = W("MobileNetSSD_deploy.prototxt")
        if os.path.exists(ssd_model) and os.path.exists(ssd_proto):
            try:
                self.ssd_net = cv2.dnn.readNetFromCaffe(ssd_proto, ssd_model)
                console.print("[green]  ✔ MobileNetSSD loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]  ⚠ MobileNetSSD failed: {e}[/yellow]")

        # ── YuNet face detector ───────────────────────────────────────────────
        yunet_path = W("face_detection_yunet_2023mar.onnx")
        if os.path.exists(yunet_path):
            try:
                self.yunet = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320))
                console.print("[green]  ✔ YuNet face detector loaded[/green]")
            except Exception as e:
                console.print(f"[yellow]  ⚠ YuNet failed: {e}[/yellow]")

        # ── Age / Gender / Emotion (ONNX) ─────────────────────────────────────
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 3

        for attr, fname in [("age_sess",     "age_googlenet.onnx"),
                             ("gender_sess",  "gender_googlenet.onnx"),
                             ("emotion_sess", "emotion-ferplus-8.onnx")]:
            path = W(fname)
            if os.path.exists(path):
                try:
                    setattr(self, attr, ort.InferenceSession(path, opts,
                                providers=["CPUExecutionProvider"]))
                    console.print(f"[green]  ✔ {fname} loaded[/green]")
                except Exception as e:
                    console.print(f"[yellow]  ⚠ {fname} failed: {e}[/yellow]")
            else:
                console.print(f"[yellow]  ⚠ {fname} not found[/yellow]")

    @classmethod
    def get(cls):
        if cls._instance is None:
            console.print("[bold cyan]Loading models...[/bold cyan]")
            cls._instance = cls()
        return cls._instance


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL MAPS
# ─────────────────────────────────────────────────────────────────────────────

AGE_BUCKETS   = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDER_LABELS = ["Female", "Male"]
EMOTION_LABELS= ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]

COCO_ANIMALS  = {14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",
                 19:"cow",20:"elephant",21:"bear",22:"zebra",23:"giraffe"}
COCO_VEHICLES = {1:"bicycle",2:"car",3:"motorcycle",5:"bus",6:"train",7:"truck"}
COCO_PEOPLE   = {0:"person"}

SSD_CLASSES   = ["background","aeroplane","bicycle","bird","boat","bottle",
                 "bus","car","cat","chair","cow","diningtable","dog","horse",
                 "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
SSD_ANIMALS   = {"bird","cat","cow","dog","horse","sheep"}
SSD_VEHICLES  = {"bicycle","bus","car","motorbike","train"}

# ─────────────────────────────────────────────────────────────────────────────
#  PER-FACE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_age_gender(face_bgr):
    """224×224 BGR blob with ImageNet mean subtraction."""
    resized = cv2.resize(face_bgr, (224, 224))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (224, 224), (104, 117, 123))
    return blob.astype(np.float32)

def _preprocess_emotion(face_bgr):
    """64×64 greyscale, float32."""
    grey = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, (64, 64)).astype(np.float32)
    return resized.reshape(1, 1, 64, 64)

def analyse_face(face_bgr):
    """
    Returns dict with age, gender, emotion from ONNX models.
    Falls back gracefully if any model is missing.
    """
    m = Models.get()
    result = {"age": "unknown", "gender": "unknown", "emotion": "unknown",
              "gender_conf": 0.0, "emotion_conf": 0.0}

    if face_bgr is None or face_bgr.size == 0:
        return result

    # ── Age ───────────────────────────────────────────────────────────────────
    if m.age_sess:
        try:
            blob = _preprocess_age_gender(face_bgr)
            out = m.age_sess.run(None, {"data_0": blob})[0][0]
            idx = int(np.argmax(out))
            result["age"] = AGE_BUCKETS[idx]
        except Exception:
            pass

    # ── Gender ────────────────────────────────────────────────────────────────
    if m.gender_sess:
        try:
            blob = _preprocess_age_gender(face_bgr)
            out = m.gender_sess.run(None, {"data_0": blob})[0][0]
            probs = np.exp(out) / np.sum(np.exp(out))   # softmax
            idx = int(np.argmax(probs))
            result["gender"]      = GENDER_LABELS[idx]
            result["gender_conf"] = float(probs[idx])
        except Exception:
            pass

    # ── Emotion ───────────────────────────────────────────────────────────────
    if m.emotion_sess:
        try:
            blob = _preprocess_emotion(face_bgr)
            out  = m.emotion_sess.run(None, {"Input3": blob})[0][0]
            probs = np.exp(out) / np.sum(np.exp(out))
            idx = int(np.argmax(probs))
            result["emotion"]      = EMOTION_LABELS[idx]
            result["emotion_conf"] = float(probs[idx])
        except Exception:
            pass

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  FACE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_faces(frame):
    """Returns list of (x, y, w, h) face boxes using YuNet or Haar fallback."""
    m   = Models.get()
    h, w = frame.shape[:2]
    faces = []

    if m.yunet:
        try:
            m.yunet.setInputSize((w, h))
            _, detections = m.yunet.detect(frame)
            if detections is not None:
                for d in detections:
                    fx, fy, fw, fh = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                    faces.append((fx, fy, fw, fh))
            return faces
        except Exception:
            pass

    # Haar fallback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raw = m.face_haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    for (x, y, w2, h2) in (raw if len(raw) else []):
        faces.append((x, y, w2, h2))
    return faces

# ─────────────────────────────────────────────────────────────────────────────
#  OBJECT / ANIMAL / PERSON DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_objects(frame, conf_threshold=0.45):
    """
    Returns list of dicts:
      { label, category, confidence, x1, y1, x2, y2 }
    category: person | animal | vehicle | object
    """
    m   = Models.get()
    h, w = frame.shape[:2]
    detections = []

    # ── YOLOv8n (preferred) ───────────────────────────────────────────────────
    if m.yolo:
        try:
            results = m.yolo.predict(frame, conf=conf_threshold, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label  = results.names[cls_id]

                if cls_id in COCO_PEOPLE:
                    cat = "person"
                elif cls_id in COCO_ANIMALS:
                    cat = "animal"
                elif cls_id in COCO_VEHICLES:
                    cat = "vehicle"
                else:
                    cat = "object"

                detections.append({"label": label, "category": cat,
                                   "confidence": conf,
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            return detections
        except Exception as e:
            console.print(f"[yellow]YOLO inference error: {e}[/yellow]")

    # ── MobileNetSSD fallback ─────────────────────────────────────────────────
    if m.ssd_net:
        try:
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                         0.007843, (300, 300), 127.5)
            m.ssd_net.setInput(blob)
            out = m.ssd_net.forward()
            for i in range(out.shape[2]):
                conf  = float(out[0, 0, i, 2])
                if conf < conf_threshold:
                    continue
                cls_id = int(out[0, 0, i, 1])
                if cls_id >= len(SSD_CLASSES):
                    continue
                label = SSD_CLASSES[cls_id]
                x1 = int(out[0, 0, i, 3] * w)
                y1 = int(out[0, 0, i, 4] * h)
                x2 = int(out[0, 0, i, 5] * w)
                y2 = int(out[0, 0, i, 6] * h)

                if label == "person":
                    cat = "person"
                elif label in SSD_ANIMALS:
                    cat = "animal"
                elif label in SSD_VEHICLES:
                    cat = "vehicle"
                else:
                    cat = "object"

                detections.append({"label": label, "category": cat,
                                   "confidence": conf,
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            return detections
        except Exception as e:
            console.print(f"[yellow]SSD inference error: {e}[/yellow]")

    # ── Haar body fallback ────────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for cascade in [m.body_haar, m.upper_haar]:
        raw = cascade.detectMultiScale(gray, 1.05, 3, minSize=(60, 120))
        for (x, y, bw, bh) in (raw if len(raw) else []):
            detections.append({"label": "person", "category": "person",
                                "confidence": 0.5,
                                "x1": x, "y1": y, "x2": x+bw, "y2": y+bh})

    return detections

# ─────────────────────────────────────────────────────────────────────────────
#  FULL FRAME ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_frame(frame, source_name="frame"):
    """
    Runs full detection + face analysis on a BGR frame.
    Returns (annotated_frame, analysis_dict).
    """
    annotated = frame.copy()
    detections = detect_objects(frame)
    faces      = detect_faces(frame)
    h_img, w_img = frame.shape[:2]

    report = {
        "timestamp":    datetime.now().isoformat(),
        "source":       source_name,
        "frame_size":   {"w": w_img, "h": h_img},
        "people":       [],
        "animals":      [],
        "vehicles":     [],
        "other_objects":[],
        "scene_summary": "",
        "threat_level": "none",
    }

    # Map each face to the nearest person detection box
    face_used = set()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label    = det["label"].capitalize()
        cat      = det["category"]
        conf     = det["confidence"]
        colour   = COL.get(cat, COL["unknown"])

        if cat == "person":
            # Find the best face inside this person box
            best_face = None
            best_area = 0
            for i, (fx, fy, fw, fh) in enumerate(faces):
                if i in face_used:
                    continue
                cx, cy = fx + fw // 2, fy + fh // 2
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = fw * fh
                    if area > best_area:
                        best_area = area
                        best_face = i
            
            face_info = {"age": "unknown", "gender": "unknown", "emotion": "unknown",
                         "gender_conf": 0.0, "emotion_conf": 0.0}
            face_box  = None

            if best_face is not None:
                face_used.add(best_face)
                fx, fy, fw, fh = faces[best_face]
                face_box = {"x": fx, "y": fy, "w": fw, "h": fh}
                face_crop = frame[max(0,fy):fy+fh, max(0,fx):fx+fw]
                face_info = analyse_face(face_crop)
                # Draw face box in different colour
                cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), COL["face"], 2)

            label_lines = [
                f"Person  {conf*100:.0f}%",
                f"Age    : {face_info['age']}",
                f"Gender : {face_info['gender']} ({face_info['gender_conf']*100:.0f}%)" if face_info['gender'] != 'unknown' else "Gender : unknown",
                f"Emotion: {face_info['emotion']} ({face_info['emotion_conf']*100:.0f}%)" if face_info['emotion'] != 'unknown' else "Emotion: unknown",
            ]
            draw_box(annotated, x1, y1, x2, y2, colour, label_lines)

            report["people"].append({
                "label": "person", "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "face_bbox": face_box,
                **face_info,
            })

        elif cat == "animal":
            label_lines = [f"{label}  {conf*100:.0f}%", "Category: Animal"]
            draw_box(annotated, x1, y1, x2, y2, colour, label_lines)
            report["animals"].append({
                "species": label, "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        elif cat == "vehicle":
            label_lines = [f"{label}  {conf*100:.0f}%", "Category: Vehicle"]
            draw_box(annotated, x1, y1, x2, y2, colour, label_lines)
            report["vehicles"].append({
                "type": label, "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        else:
            label_lines = [f"{label}  {conf*100:.0f}%"]
            draw_box(annotated, x1, y1, x2, y2, colour, label_lines)
            report["other_objects"].append({
                "label": label, "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    # Orphan faces (detected but not inside any person box)
    for i, (fx, fy, fw, fh) in enumerate(faces):
        if i in face_used:
            continue
        face_crop = frame[max(0,fy):fy+fh, max(0,fx):fx+fw]
        face_info = analyse_face(face_crop)
        cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), COL["face"], 2)
        label_lines = [
            "Person (face only)",
            f"Age: {face_info['age']}",
            f"Gender: {face_info['gender']}",
            f"Emotion: {face_info['emotion']}",
        ]
        draw_box(annotated, fx, fy, fx+fw, fy+fh, COL["person"], label_lines)
        report["people"].append({
            "label": "person (face only)", "confidence": 0.7,
            "bbox": {"x1": fx, "y1": fy, "x2": fx+fw, "y2": fy+fh},
            "face_bbox": {"x": fx, "y": fy, "w": fw, "h": fh},
            **face_info,
        })

    # HUD overlay
    draw_hud(annotated, detections)

    # Summary
    n_p = len(report["people"])
    n_a = len(report["animals"])
    n_v = len(report["vehicles"])
    parts = []
    if n_p: parts.append(f"{n_p} person(s)")
    if n_a: parts.append(f"{n_a} animal(s)")
    if n_v: parts.append(f"{n_v} vehicle(s)")
    report["scene_summary"]  = ", ".join(parts) if parts else "empty scene"
    report["threat_level"]   = "low" if n_p else "none"

    return annotated, _to_python(report)


def _to_python(obj):
    """Recursively convert numpy scalars/arrays to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(report, json_dir, prefix=""):
    os.makedirs(json_dir, exist_ok=True)
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    name = f"{prefix}{ts}.json"
    path = os.path.join(json_dir, name)
    with open(path, "w") as f:
        json.dump(_to_python(report), f, indent=2)
    return path
