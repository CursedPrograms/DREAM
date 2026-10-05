"""
dream_visions.py - her dreams, seen.

dream_images.py paints a dream from her memories and walks latent space for it,
but neither knows what the dream *says*. This hands the dream's own words to the
Image-Generator (github.com/CursedPrograms/Image-Generator, checked out next to
DREAM; its scripts/generate.py runs SDXL-Turbo without asking questions):

  vision   text-to-image from the dream itself -> vision_<stamp>_<cycle>.jpg
  frames   every few frames of the latent walk, re-imagined with the same
           prompt (image-to-image), so the dream world takes on its subject
  video    when she wakes, every frame of the night - the walks and the
           visions, dream by dream - becomes dreams_<stamp>.mp4

All of it lives in the images folder (output/dreams). The generator runs in its
own process with DREAM's Python, so it can be killed the moment she wakes and
its memory goes away with it. Which model: config.json's DREAM.DreamImageModel -
"sdxl-turbo" (best), "sd-turbo" (a quarter the size: 4 GB GPU, 8 GB RAM), any
other Hugging Face id, or "auto": SDXL-Turbo with 12 GB of RAM or more, else SD-Turbo.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import store

try:
    import cv2
except ImportError:
    cv2 = None

VISUALIZE = True                  # selftest turns this off: it shouldn't need a GPU and a model download
GENERATOR = Path(os.environ.get("DREAM_IMAGE_GENERATOR", store.ROOT.parent / "Image-Generator")) / "scripts" / "generate.py"
VISION_SIZE = 512
VISION_FRAMES = 8                 # latent-walk frames re-imagined per dream
STRENGTH = {"pleasant": 0.45, "strange": 0.55, "nightmare": 0.6}   # how far each frame strays from the walk
BLEND = 0.6                       # frames are mixed with the vision, up to this much by the last: the walk turns into the dream
MIN_FREE_RAM_GB = 2.5
FIRST_RUN_TIMEOUT_S = 1800        # the first run may still be downloading the model
TIMEOUT_S = 900
CHECK_CACHE = "dream_visions.json"

TONE_STYLE = {
    "pleasant":  "soft warm light, gentle colours, peaceful surreal dreamscape",
    "strange":   "surreal dreamlike painting, shifting impossible architecture, mist",
    "nightmare": "dark unsettling nightmare, deep shadows, eerie red light, surreal horror",
}

# the video
FPS = 24
WALK_REPEAT = 2                   # latent-walk frames at 12 per second
VISION_HOLD_S = 1.0
FADE_S = 0.4


def images_dir() -> Path:
    d = store.IMAGES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def frames_dir(dream) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(dream["ts"]))
    return images_dir() / "frames" / f"{stamp}_{dream.get('cycle', 0)}"


def model():
    try:
        with open(store.ROOT / "config.json", encoding="utf-8") as f:
            chosen = json.load(f)["Config"]["DREAM"].get("DreamImageModel", "auto")
    except (OSError, ValueError, KeyError):
        chosen = "auto"
    if chosen and chosen != "auto":
        return chosen if "/" in chosen else f"stabilityai/{chosen}"
    try:
        import psutil
        total = psutil.virtual_memory().total / 1e9
    except Exception:
        total = 0
    return "stabilityai/sdxl-turbo" if total >= 12 else "stabilityai/sd-turbo"


def _free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0


def usable():
    """Whether the generator can run right now. The import check runs once a day
    in a subprocess (a broken torch can take the process down with it)."""
    if not VISUALIZE or not GENERATOR.exists() or _free_ram_gb() < MIN_FREE_RAM_GB:
        return False
    cached = store.read_json(CHECK_CACHE, None)
    if cached and time.time() - cached.get("ts", 0) < 86400:
        return bool(cached.get("ok"))
    try:
        ok = subprocess.run([sys.executable, "-c", "import torch, diffusers"], capture_output=True, timeout=120).returncode == 0
    except (subprocess.SubprocessError, OSError):
        ok = False
    store.write_json(CHECK_CACHE, {"ts": time.time(), "ok": ok, "ran": (cached or {}).get("ran", False)})
    return ok


def prompt_for(dream):
    """The dream's words, trimmed to what the text encoder reads (~75 tokens), in its tone's style."""
    text = re.sub(r"\s+", " ", dream["text"]).strip()
    text = re.sub(r"\b(I'm|I was|myself|my|me|I)\b", "", text, flags=re.I)   # 'I' means nothing to a picture
    words = re.sub(r"\s+", " ", text).split()[:45]
    return f"{TONE_STYLE.get(dream['tone'], TONE_STYLE['strange'])}, {' '.join(words)}"


def save_world(dream, world_bgr):
    """Keep the dream world's frames for the night's video. Returns the folder."""
    d = frames_dir(dream)
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(world_bgr):
        big = cv2.resize(f, (VISION_SIZE, VISION_SIZE), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(d / f"world_{i:03d}.jpg"), big, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return d


def visualize(dream, stop=None):
    """Turn the dream's words into pictures. Needs save_world() first. Returns the
    vision's file name (inside images_dir()), or None."""
    if cv2 is None or not usable() or (stop is not None and stop.is_set()):
        return None
    d = frames_dir(dream)
    world = sorted(d.glob("world_*.jpg"))
    picks = [world[int(i * len(world) / VISION_FRAMES)] for i in range(VISION_FRAMES)] if len(world) >= VISION_FRAMES else world
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(dream["ts"]))
    vision = f"vision_{stamp}_{dream.get('cycle', 0)}.jpg"
    job = {
        "prompt": prompt_for(dream),
        "keyframe": str(images_dir() / vision),
        "frames": [[str(p), str(d / p.name.replace("world_", "vision_"))] for p in picks],
        "strength": STRENGTH.get(dream["tone"], 0.55),
        "size": VISION_SIZE,
        "seed": int(dream["ts"]) % 2**31,
        "blend": BLEND,
    }
    job_path = d / "job.json"
    job_path.write_text(json.dumps(job, indent=1), encoding="utf-8")

    first = not (store.read_json(CHECK_CACHE, {}) or {}).get("ran")
    proc = subprocess.Popen([sys.executable, str(GENERATOR), "--job", str(job_path), "--model", model()],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + (FIRST_RUN_TIMEOUT_S if first else TIMEOUT_S)
    while proc.poll() is None:
        if (stop is not None and stop.is_set()) or time.time() > deadline:
            proc.terminate()
            break
        time.sleep(0.5)
    if proc.returncode == 0:
        cached = store.read_json(CHECK_CACHE, {}) or {}
        store.write_json(CHECK_CACHE, dict(cached, ran=True))
    dream["vision_prompt"] = job["prompt"]
    return vision if (images_dir() / vision).exists() else None


# ---------------------------------------------------------------- the night's video
def _read(path):
    img = cv2.imread(str(path))
    return None if img is None else cv2.resize(img, (VISION_SIZE, VISION_SIZE), interpolation=cv2.INTER_AREA)


def _dream_frames(dream):
    """This dream's part of the video, as BGR frames: the walk, then the visions crossfading."""
    d = frames_dir(dream)
    out = []
    for p in sorted(d.glob("world_*.jpg")):
        img = _read(p)
        if img is not None:
            out += [img] * WALK_REPEAT
    stills = []
    if dream.get("vision"):
        stills.append(_read(images_dir() / dream["vision"]))
    stills += [_read(p) for p in sorted(d.glob("vision_*.jpg"))]
    stills = [s for s in stills if s is not None]
    hold, fade = int(VISION_HOLD_S * FPS), int(FADE_S * FPS)
    for s in stills:
        if out:   # crossfade in from whatever came before
            prev = out[-1].astype(np.float32)
            for k in range(1, fade + 1):
                a = k / (fade + 1)
                out.append(((1 - a) * prev + a * s.astype(np.float32)).astype(np.uint8))
        out += [s] * hold
    return out


def session_video(dreams, started):
    """Every frame of the night, dream by dream, as one MP4. Returns its path, or None."""
    if cv2 is None or not dreams:
        return None
    path = images_dir() / f"dreams_{time.strftime('%Y%m%d_%H%M%S', time.localtime(started))}.mp4"
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(str(path), fps=FPS, codec="libx264", quality=7, macro_block_size=16)
    except Exception as e:
        print(f"[mind] can't write the dream video: {e}")
        return None
    n = 0
    try:
        for dream in dreams:
            for f in _dream_frames(dream):
                writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                n += 1
    finally:
        writer.close()
    if n == 0:
        path.unlink(missing_ok=True)
        return None
    print(f"[mind] the night's dreams: {path.name} ({n / FPS:.0f} s)")
    return path


def session_video_later(dreams, started):
    """session_video() in the background, so waking up isn't held up by it."""
    threading.Thread(target=session_video, args=(list(dreams), started), daemon=True, name="dream-video").start()
