"""
vision.py - perception of the room.

surveillance.py already saves a photo every ten minutes and a frame whenever
something moves. This is the part that *looks* at them. She keeps a running
picture of what the room is normally like, and each new photo is measured
against it: brighter or darker than usual? someone in view? changed? What's
unusual is remembered and can be mentioned; what's ordinary is quietly folded
into her sense of normal. Everything stays on this machine.

If there's no webcam (surveillance.py's "blind mode") there are simply no
photos, and she carries on without eyes.
"""

import os
import time
from pathlib import Path

import numpy as np

from . import store

try:
    import cv2
except ImportError:  # no OpenCV: no eyes, but everything else still works
    cv2 = None

OBSERVATIONS = "visual.jsonl"
STATE = "vision_state.json"
THUMB = (16, 9)
MAX_PER_SCAN = 6
CORE_SIGNIFICANCE = 0.75

SCRIPTS = store.ROOT / "scripts"
PHOTO_DIRS = [("photo", SCRIPTS / "output" / "eyes_on_you"), ("motion", SCRIPTS / "output" / "motion_alerts")]


def set_photo_dirs(dirs):
    """dirs: [(kind, path)] - the self-test uses fake ones."""
    global PHOTO_DIRS
    PHOTO_DIRS = [(k, Path(p)) for k, p in dirs]


def _load_face_detectors():
    if cv2 is None:
        return None, None
    face = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    smile = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml"))
    return (face if not face.empty() else None), (smile if not smile.empty() else None)


class Vision:
    def __init__(self):
        self.available = cv2 is not None
        self.face_cascade, self.smile_cascade = _load_face_detectors()
        s = store.read_json(STATE, {}) or {}
        self.last_mtime = s.get("last_mtime", 0.0)
        self.normal = np.array(s["normal"], dtype=np.float32) if s.get("normal") else None   # running mean thumbnail
        self.normal_n = s.get("normal_n", 0)
        self.last_thumb = np.array(s["last_thumb"], dtype=np.float32) if s.get("last_thumb") else None
        self.last_brightness = s.get("last_brightness")

    def _save(self):
        store.write_json(STATE, {
            "last_mtime": self.last_mtime, "normal_n": self.normal_n,
            "normal": self.normal.round(2).tolist() if self.normal is not None else None,
            "last_thumb": self.last_thumb.round(2).tolist() if self.last_thumb is not None else None,
            "last_brightness": self.last_brightness,
        })

    # ------------------------------------------------------------ looking
    def scan(self, now=None):
        """Look at any photos taken since last time. Returns the new observations."""
        if not self.available:
            return []
        now = now or time.time()
        found = []
        for kind, folder in PHOTO_DIRS:
            if not folder.is_dir():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() == ".jpg":
                    try:
                        mtime = f.stat().st_mtime
                    except OSError:
                        continue
                    if mtime > self.last_mtime:
                        found.append((mtime, kind, f))
        found.sort()
        observations = []
        for mtime, kind, f in found[:MAX_PER_SCAN]:
            obs = self._observe(f, kind, mtime)
            if obs:
                observations.append(obs)
                store.append_jsonl(OBSERVATIONS, obs)
            self.last_mtime = max(self.last_mtime, mtime)
        if len(found) > MAX_PER_SCAN:   # a backlog (she was off): skip ahead rather than grind through it
            self.last_mtime = found[-1][0]
        if found:
            self._save()
        return observations

    def _observe(self, path, kind, mtime):
        img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if img is None:
            return None
        thumb = cv2.resize(img, THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)
        brightness = float(thumb.mean())

        delta = float(np.abs(thumb - self.last_thumb).mean()) if self.last_thumb is not None else 0.0
        anomaly = float(np.abs(thumb - self.normal).mean()) if self.normal is not None and self.normal_n >= 5 else 0.0
        d_bright = (brightness - self.last_brightness) if self.last_brightness is not None else 0.0

        faces, smiling = 0, False
        if self.face_cascade is not None:
            found = self.face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            faces = len(found)
            if faces and self.smile_cascade is not None:   # a rough read of the expression
                x, y, w, h = max(found, key=lambda r: r[2] * r[3])
                lower = img[y + h // 2:y + h, x:x + w]
                if lower.size:
                    smiling = len(self.smile_cascade.detectMultiScale(lower, scaleFactor=1.7, minNeighbors=20)) > 0

        # what she now considers normal absorbs this (a slow average)
        self.normal = thumb.copy() if self.normal is None else 0.9 * self.normal + 0.1 * thumb
        self.normal_n += 1
        self.last_thumb, self.last_brightness = thumb, brightness

        # how much this is worth remembering: a big change from normal is a core
        # memory; someone showing up or motion is notable; the ordinary is not.
        significance = max(0.7 * min(1.0, delta / 35.0), 0.85 * min(1.0, anomaly / 30.0),
                           0.6 if faces else 0.0, 0.65 if kind == "motion" else 0.0)
        if kind == "motion":
            text = "something moved in the room"
        elif faces:
            text = "someone was in view" + (" and looked like they were smiling" if smiling else "")
        elif abs(d_bright) > 25:
            text = "the light changed, the room got " + ("brighter" if d_bright > 0 else "darker")
        elif anomaly > 20:
            text = "the room looked different from how it usually does"
        elif delta > 20:
            text = "something in the room had changed"
        else:
            text = "nothing unusual"
        return {"ts": mtime, "path": str(path), "kind": kind, "brightness": round(brightness, 1),
                "delta": round(delta, 1), "anomaly": round(anomaly, 1), "faces": faces, "smiling": smiling,
                "significance": round(significance, 2), "text": text,
                "core": significance >= CORE_SIGNIFICANCE, "spoken": False}

    # ------------------------------------------------------------ remembering
    def observations(self, limit=50):
        return store.read_jsonl(OBSERVATIONS)[-limit:]

    def unspoken_notable(self, min_significance=0.6, max_age_h=6, now=None):
        """The most striking thing she saw recently and hasn't mentioned."""
        now = now or time.time()
        cands = [o for o in self.observations(40)
                 if not o.get("spoken") and o["significance"] >= min_significance and now - o["ts"] < max_age_h * 3600]
        return max(cands, key=lambda o: o["significance"]) if cands else None

    def mark_spoken(self, obs):
        items = store.read_jsonl(OBSERVATIONS)
        for o in items:
            if o["ts"] == obs["ts"] and o["path"] == obs["path"]:
                o["spoken"] = True
        store.write_jsonl(OBSERVATIONS, items)

    def latest_summary(self, now=None):
        now = now or time.time()
        obs = self.observations(1)
        if not obs:
            return ""
        o = obs[-1]
        ago = int((now - o["ts"]) / 60)
        when = "just now" if ago < 2 else f"{ago} minutes ago" if ago < 90 else f"{ago // 60} hours ago"
        return f"You last looked around {when}: {o['text']}."

    def core_memories(self, n=5):
        return sorted([o for o in store.read_jsonl(OBSERVATIONS) if o.get("core")], key=lambda o: -o["significance"])[:n]

    def presence_pattern(self):
        """When she usually sees someone: the hours with the most faces, or []."""
        hours = {}
        for o in store.read_jsonl(OBSERVATIONS):
            if o.get("faces"):
                h = time.localtime(o["ts"]).tm_hour
                hours[h] = hours.get(h, 0) + 1
        return [h for h, c in sorted(hours.items(), key=lambda kv: -kv[1])[:3] if c >= 2]
