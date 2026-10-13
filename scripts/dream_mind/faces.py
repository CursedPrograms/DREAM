"""
faces.py - knowing who's there.

Tell her your name while you face the camera ("my name is Kimmie") and she keeps
a face print of you: 128 numbers from OpenCV's SFace model, never a picture. From
then on, the camera frames surveillance.py publishes (frames.py) are checked
every couple of seconds, and when she sees you again she knows it's you: she can
greet you by name, and "what do you see?" becomes "I see Kimmie...".

  * Only people who introduced themselves are known. Anyone else is just
    "someone I don't know".
  * Each person keeps a few prints (different light, angles), added as she sees
    them, so recognising gets better with time.
  * "Forget my face" deletes that person's prints; nothing else is kept.

Models (OpenCV Zoo, downloaded once into models/faces/, about 39 MB):
  YuNet  finds faces          face_detection_yunet_2023mar.onnx
  SFace  turns one into a print   face_recognition_sface_2021dec.onnx
"""

import os
import threading
import time
import urllib.request

import numpy as np

from . import store

try:
    import cv2
except ImportError:
    cv2 = None

FILE = "faces.json"
MODEL_DIR = store.ROOT / "models" / "faces"
MODELS = {
    "detector": ("face_detection_yunet_2023mar.onnx",
                 "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    "recognizer": ("face_recognition_sface_2021dec.onnx",
                   "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
}

MATCH = 0.45            # cosine similarity to count as the same person (OpenCV suggests 0.363; that let
                        # 2 of 146 strangers through in a test, real matches scored 0.55-0.96)
NEW_LOOK = 0.75         # a print less alike than this to all of theirs is worth keeping too
MAX_PRINTS = 12         # per person
MIN_FACE_PX = 40        # smaller faces are too far away to tell apart
MAX_WIDTH = 640         # frames are looked at this size (a webcam frame already is)


def download_models(quiet=False):
    """Fetch the two models if they're missing. True when both are there."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in MODELS.values():
        path = MODEL_DIR / fname
        if path.is_file():
            if not quiet:
                print(f"  {fname}: already there")
            continue
        print(f"  {fname}: downloading ...")
        try:
            urllib.request.urlretrieve(url, str(path) + ".part")
            os.replace(str(path) + ".part", path)
        except Exception as e:
            print(f"  {fname}: FAILED ({e})")
            return False
        print(f"  {fname}: OK")
    return True


class Faces:
    def __init__(self):
        self.known = {}             # name -> [print (np.float32, 128)]
        self._load()
        self.detector = self.recognizer = None
        self._lock = threading.Lock()
        self.seen = {}              # name -> last time she saw them
        self.last_view = []         # [(name or None, similarity)] from the newest frame looked at
        self.last_view_ts = 0.0

    # ------------------------------------------------------------ storage
    def _load(self):
        d = store.read_json(FILE, {}) or {}
        self.known = {n: [np.array(p, dtype=np.float32) for p in prints] for n, prints in (d.get("people") or {}).items()}

    def _save(self):
        store.write_json(FILE, {"people": {n: [p.round(5).tolist() for p in prints] for n, prints in self.known.items()}})

    # ------------------------------------------------------------ models
    def ready(self):
        """Load the models on first use (downloading them if needed)."""
        if cv2 is None or not hasattr(cv2, "FaceRecognizerSF"):
            return False
        if self.detector is not None:
            return True
        if not download_models(quiet=True):
            return False
        self.detector = cv2.FaceDetectorYN.create(str(MODEL_DIR / MODELS["detector"][0]), "", (320, 320), 0.8)
        self.recognizer = cv2.FaceRecognizerSF.create(str(MODEL_DIR / MODELS["recognizer"][0]), "")
        return True

    def _faces(self, frame):
        """Every face in the frame, biggest first: [(print, box)]."""
        if frame.shape[1] > MAX_WIDTH:   # YuNet misses faces that fill a big picture; at webcam size it finds them
            s = MAX_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, found = self.detector.detect(frame)
            if found is None:
                return []
            out = []
            for f in sorted(found, key=lambda f: f[2] * f[3], reverse=True):
                if min(f[2], f[3]) < MIN_FACE_PX:
                    continue
                aligned = self.recognizer.alignCrop(frame, f)
                feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
                out.append((feat / (np.linalg.norm(feat) or 1.0), tuple(int(v) for v in f[:4])))
            return out

    def _match(self, feat):
        best, best_sim = None, 0.0
        for name, prints in self.known.items():
            sim = max(float(np.dot(feat, p)) for p in prints)
            if sim > best_sim:
                best, best_sim = name, sim
        return (best, best_sim) if best_sim >= MATCH else (None, best_sim)

    # ------------------------------------------------------------ looking
    def look(self, frame, now=None):
        """Who's in this frame: [(name or None, similarity)], biggest face first.
        Known faces are marked as seen, and a new look at them is learned."""
        now = now or time.time()
        if frame is None or not self.ready():
            return []
        view = []
        changed = False
        for feat, _box in self._faces(frame):
            name, sim = self._match(feat)
            view.append((name, sim))
            if name:
                self.seen[name] = now
                prints = self.known[name]
                if len(prints) < MAX_PRINTS and max(float(np.dot(feat, p)) for p in prints) < NEW_LOOK and sim >= MATCH + 0.1:
                    prints.append(feat)     # clearly them, but from a new angle or light: remember this look too
                    changed = True
        if changed:
            self._save()
        self.last_view, self.last_view_ts = view, now
        return view

    def enroll(self, frame, name):
        """Learn the biggest face in the frame as `name`. Returns "new", "again",
        "no_camera", "no_face" or "unavailable"."""
        if not self.ready():
            return "unavailable"
        if frame is None:
            return "no_camera"
        found = self._faces(frame)
        if not found:
            return "no_face"
        feat = found[0][0]
        again = name in self.known
        self.known.setdefault(name, [])
        if len(self.known[name]) >= MAX_PRINTS:
            self.known[name].pop(0)
        self.known[name].append(feat)
        self.seen[name] = time.time()
        self._save()
        return "again" if again else "new"

    def forget(self, name=None):
        """Forget one person's face, or everyone's. Returns who was forgotten."""
        gone = list(self.known) if name is None else [n for n in self.known if n.lower() == name.lower()]
        for n in gone:
            del self.known[n]
            self.seen.pop(n, None)
        if gone:
            self._save()
        return gone

    def in_view(self, now=None, within_s=20):
        """Names she has seen in the last `within_s` seconds."""
        now = now or time.time()
        return [n for n, t in self.seen.items() if now - t <= within_s]

    def strangers_in_view(self, now=None, within_s=20):
        now = now or time.time()
        if now - self.last_view_ts > within_s:
            return 0
        return sum(1 for n, _ in self.last_view if n is None)
