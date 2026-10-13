"""
sight.py - looking at something and saying what she sees.

"What do you see?", "look at this", "what am I holding?": she takes a picture
with the webcam and a vision model (VisionModel in config.json, picked in
options.py) describes it, in her own voice. If she recognises anyone in it
(faces.py), the model is told who they are, so she says "Kimmie" instead of
"a person".

The picture comes from the frames surveillance.py publishes (frames.py), or,
with no camera watcher running (look.py on its own), from the webcam directly.
Each look is kept in scripts/output/looks/ (the photo, and what she said about it).

She also looks on her own (mind.py): a photo a minute while she's awake and
idle (AutoLook in config.json; "stop looking around" turns it off). When it
shows something new she writes herself one past-tense sentence about it
(observe) and remembers it (vision.py), and it can turn up in her dreams.
"""

import base64
import json
import os
import sys
import time

import requests

from . import frames, llm, store

try:
    import cv2
except ImportError:
    cv2 = None

LOOKS_DIR = store.ROOT / "scripts" / "output" / "looks"
MAX_WIDTH = 768
TIMEOUT_S = 180


def _models():
    scripts = str(store.ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import dream_models
    return dream_models


def take_picture(camera_index=0):
    """A fresh BGR frame: the shared one if the camera watcher is running, else
    straight from the webcam. None if there's no camera."""
    frame, _ = frames.latest(3.0)
    if frame is not None or cv2 is None:
        return frame
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(8):          # the first frames are dark while the camera adjusts its exposure
            ok, f = cap.read()
            if ok:
                frame = f
        return frame
    finally:
        cap.release()


def _jpeg_b64(frame):
    if frame.shape[1] > MAX_WIDTH:
        s = MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def _persona():
    try:
        cfg = json.loads((store.ROOT / "config.json").read_text(encoding="utf-8"))["Config"]["DREAM"]
        name = cfg.get("CharName", "DREAM")
        return cfg.get("SystemPrompt", "You are {name}.").replace("{name}", name)
    except (OSError, ValueError, KeyError):
        return "You are DREAM."


def describe(frame, question="", people=(), strangers=0):
    """What she says about this picture. `people`: names of those she recognises in it."""
    if frame is None:
        return "I can't see anything. My camera isn't on."
    img = _jpeg_b64(frame)
    if img is None:
        return "I couldn't take that picture."
    who = ""
    if people:
        who = f"You recognise the person in the picture: {', '.join(people)}. Call them by name. "
        if strangers:
            who += "There's also someone you don't know. "
    elif strangers:
        who = "There's someone in the picture you don't know. "
    ask = question.strip() or "What do you see?"
    prompt = (f"{_persona()}\n\nThis is what your webcam sees right now: you're looking at it with your own eyes. {who}"
              f"They said: \"{ask}\"\nAnswer it out loud in two or three short sentences, about what is actually in "
              "the picture. Plain sentences only: no lists, no markdown.")
    model = _models().vision_model()
    try:
        r = requests.post(llm.OLLAMA_URL, json={"model": model, "prompt": prompt, "images": [img], "stream": False,
                                                "options": {"num_predict": 160, "temperature": 0.6}}, timeout=TIMEOUT_S)
        r.raise_for_status()
        text = (r.json().get("response") or "").strip()
    except requests.RequestException as e:
        print(f"[sight] {model}: {e}")
        return f"I tried to look, but my eyes aren't working. Is {model} pulled in Ollama?"
    text = " ".join(text.replace("*", "").split())
    return text or "I looked, but I'm not sure what I'm seeing."


def thumb(frame):
    """A tiny grey version of the picture, to tell whether the scene changed."""
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA)
    return small.astype("float32")


def same_scene(a, b, threshold=10.0):
    return float(abs(a - b).mean()) < threshold


def autolook_settings():
    """(enabled, every_s) from Config/DREAM/AutoLook in config.json."""
    try:
        cfg = json.loads((store.ROOT / "config.json").read_text(encoding="utf-8"))["Config"]["DREAM"].get("AutoLook") or {}
    except (OSError, ValueError, KeyError, AttributeError):
        cfg = {}
    return bool(cfg.get("Enabled", True)), max(15, int(cfg.get("EverySeconds", 60) or 60))


def observe(frame, people=(), strangers=0):
    """One past-tense sentence about this picture, for her memory (not spoken).
    None if the model is busy (talking or dreaming comes first) or failed."""
    if frame is None:
        return None
    img = _jpeg_b64(frame)
    if img is None:
        return None
    if people:
        who = f"The person in the picture is {', '.join(people)}: use the name."
        example = f"{people[0]} was at the desk, playing a guitar."
    else:
        who = "There's someone in it you don't know." if strangers else ""
        example = "The room was empty and the lamp was on."
    prompt = (f"This is a photo from your webcam. {who} Write one short sentence in the past tense about what was "
              f"happening, as a note to remember it by, like: \"{example}\" Only the sentence.")
    if not llm._lock.acquire(blocking=False):
        return None
    try:
        r = requests.post(llm.OLLAMA_URL, json={"model": _models().vision_model(), "prompt": prompt, "images": [img],
                                                "stream": False, "options": {"num_predict": 60, "temperature": 0.3}},
                          timeout=TIMEOUT_S)
        r.raise_for_status()
        text = " ".join((r.json().get("response") or "").replace("*", "").split()).strip('"')
    except requests.RequestException as e:
        print(f"[sight] look: {e}")
        return None
    finally:
        llm._lock.release()
    return text or None


def keep(frame, said, question=""):
    """Save the photo and what she said about it."""
    if frame is None or cv2 is None:
        return None
    LOOKS_DIR.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("look_%Y%m%d_%H%M%S")
    cv2.imwrite(str(LOOKS_DIR / f"{stem}.jpg"), frame)
    (LOOKS_DIR / f"{stem}.txt").write_text((f"Q: {question}\n" if question else "") + said + "\n", encoding="utf-8")
    return LOOKS_DIR / f"{stem}.jpg"
