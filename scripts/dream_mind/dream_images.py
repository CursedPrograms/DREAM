"""
dream_images.py - what she sees while she dreams.

This absorbs the old dreaming scripts. latent_space.py is DREAM's dreaming
engine: it walks a path through the latent space of a (randomly initialised)
decoder network, in modes like spiral, pulse, random_walk and interpolate, and
saves the frames; deep_dream_batch.py was meant to trippify those frames
afterwards. Nothing ever started them together, and neither knew anything about
what she had lived. Now each dream the mind writes is also painted:

  world    latent_space.py walks a path chosen by the dream's tone (a spiral for
           the strange ones, a slow pulse for pleasant ones, an erratic walk for
           nightmares), seeded by the dream's own text so the same dream always
           looks the same, and graded in colour to match
  memory   the most striking thing she has seen (a core visual memory), else the
           newest photo, else - if she has no eyes at all - shapes from the dream
           itself, is painted and bleeds through the world
  output   a still and an animated GIF of the walk
  extra    if TensorFlow works and there's memory to spare, the still is also run
           through DeepDream (the old scripts' InceptionV3 layers, per tone)

If PyTorch is missing, she paints with a pure OpenCV/NumPy renderer instead
(feedback zoom and rotation, kaleidoscope folds, warping, colour drift).

The painting happens in the sleep thread and is abandoned the moment she wakes.
"""

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import store

try:
    import cv2
except ImportError:
    cv2 = None

MAX_SIDE = 512
SCRIPTS = store.ROOT / "scripts"
LATENT_FRAMES = 40
LATENT_SIZE = 224
GIF_SIZE = 192
GIF_FRAME_MS = 90
MEMORY_BLEED = 0.20              # how much of her own memory shows through the dream world
# The dream's tone decides how she walks through latent space (see latent_space.py).
TONE_MODE = {"pleasant": "pulse", "strange": "spiral", "nightmare": "random_walk"}
MIN_FREE_RAM_GB = 1.2            # below this TensorFlow is skipped (it would page the machine to a crawl)
# DeepDream's memory grows with the picture, so use as large an image as the free RAM allows.
TF_SIDES = ((3.0, 512), (1.8, 320), (1.2, 224))   # (free GB at least, longest image side)
TF_TIMEOUT_S = 600
ENGINE_CACHE = "dream_engine.json"

# The old scripts' layer choices, by tone: mixed3-5 are geometry, eyes and animals;
# mixed7-8 are the aggressive, "fried" layers nightmare_dreamer.py used.
TF_PRESETS = {
    "pleasant":  {"layers": ["mixed3", "mixed4"], "octaves": 3, "steps": 12, "step_size": 0.01, "scale": 1.3},
    "strange":   {"layers": ["mixed3", "mixed4", "mixed5"], "octaves": 4, "steps": 15, "step_size": 0.01, "scale": 1.3},
    "nightmare": {"layers": ["mixed7", "mixed8"], "octaves": 4, "steps": 20, "step_size": 0.8, "scale": 2.4},
}

# OpenCV renderer: how each tone bends the picture each frame.
STYLES = {
    "pleasant":  dict(frames=22, zoom=1.010, rot=0.7, hue=2.0, sat=1.10, val=1.00, blend=0.90, warp=2.5, fold=0, edges=0.0, vignette=0.15, noise=0.0, tint=None),
    "strange":   dict(frames=28, zoom=1.022, rot=1.6, hue=6.0, sat=1.18, val=1.00, blend=0.86, warp=5.0, fold=6, edges=0.10, vignette=0.25, noise=0.0, tint=None),
    "nightmare": dict(frames=30, zoom=1.030, rot=-2.2, hue=0.0, sat=1.05, val=0.955, blend=0.84, warp=9.0, fold=0, edges=0.30, vignette=0.60, noise=0.06, tint=(0, 0, 255)),
}


def images_dir() -> Path:
    d = store.IMAGES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- the source
def _seed_for(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def blind_canvas(seed_text, size=(384, 512)):
    """Something to dream on when she has no photos: soft coloured blobs, seeded
    by the dream so no two are alike."""
    rng = np.random.default_rng(_seed_for(seed_text))
    h, w = size
    canvas = np.zeros((h, w, 3), np.float32)
    for _ in range(9):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(0.08, 0.30) * max(w, h)
        col = rng.uniform(0.15, 1.0, 3)
        ys, xs = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * r * r)))
        canvas += blob[..., None] * col[None, None, :]
    canvas = np.clip(canvas / max(canvas.max(), 1e-6), 0, 1)
    return (canvas * 255).astype(np.uint8)


def choose_source(mind, dream):
    """(image BGR uint8, description) for this dream to be painted from."""
    if cv2 is not None:
        try:
            for obs in mind.vision.core_memories(3):
                img = cv2.imread(obs["path"])
                if img is not None:
                    return img, f"a photo she remembers ({obs['text']})"
            recent = mind.vision.observations(10)
            for obs in reversed(recent):
                img = cv2.imread(obs["path"])
                if img is not None:
                    return img, "the last thing she saw"
        except Exception:
            pass
    return blind_canvas(dream["text"]), "shapes from the dream itself"


# ---------------------------------------------------------------- OpenCV renderer
def _hsv_shift(img, hue, sat, val, tint):
    hsv = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 1)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if tint is not None:   # the nightmare bleeds toward red
        t = np.array(tint, np.float32) / 255.0
        out = out * 0.94 + t[None, None, :] * 0.06 * out.mean(axis=2, keepdims=True)
    return out


def _kaleido(img, folds):
    """Mirror-fold the picture into `folds` wedges around the centre."""
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    wedge = 2 * math.pi / folds
    ang = np.abs(((ang % wedge) - wedge / 2.0))
    return cv2.remap(img, (cx + r * np.cos(ang)).astype(np.float32), (cy + r * np.sin(ang)).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_opencv(source_bgr, tone, seed, stop=None):
    """The dream, painted without TensorFlow. Returns a uint8 BGR image."""
    st = STYLES.get(tone, STYLES["strange"])
    rng = np.random.default_rng(seed)
    h0, w0 = source_bgr.shape[:2]
    scale = MAX_SIDE / max(h0, w0)
    base = cv2.resize(source_bgr, (max(64, int(w0 * scale)), max(64, int(h0 * scale))), interpolation=cv2.INTER_AREA)
    base = base.astype(np.float32) / 255.0
    h, w = base.shape[:2]
    canvas = base.copy()
    phase = rng.uniform(0, 2 * math.pi, 4)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    for f in range(st["frames"]):
        if stop is not None and stop.is_set():
            return None
        # zoom + rotate about a wandering centre: the feedback loop
        centre = (w / 2 + 12 * math.sin(f * 0.3 + phase[0]), h / 2 + 12 * math.cos(f * 0.27 + phase[1]))
        m = cv2.getRotationMatrix2D(centre, st["rot"], st["zoom"])
        canvas = cv2.warpAffine(canvas, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # liquid distortion
        if st["warp"]:
            dx = (st["warp"] * np.sin(ys / 23.0 + f * 0.4 + phase[2])).astype(np.float32)
            dy = (st["warp"] * np.cos(xs / 29.0 + f * 0.35 + phase[3])).astype(np.float32)
            canvas = cv2.remap(canvas, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if st["fold"] and f % 4 == 3:
            canvas = 0.5 * canvas + 0.5 * _kaleido(canvas, st["fold"])
        canvas = _hsv_shift(canvas, st["hue"], st["sat"], st["val"], st["tint"])
        # what she started from keeps leaking back in
        canvas = st["blend"] * canvas + (1 - st["blend"]) * _hsv_shift(base, st["hue"] * f * 0.3, 1.1, 1.0, None)
        if st["edges"]:   # outlines glow
            g = cv2.cvtColor(np.clip(canvas, 0, 1), cv2.COLOR_BGR2GRAY)
            e = cv2.Canny((g * 255).astype(np.uint8), 60, 140).astype(np.float32) / 255.0
            glow = np.array(st["tint"] or (255, 255, 255), np.float32) / 255.0
            canvas = np.clip(canvas + st["edges"] * e[..., None] * glow[None, None, :], 0, 1)
        if st["noise"]:
            canvas = np.clip(canvas + rng.normal(0, st["noise"], canvas.shape).astype(np.float32), 0, 1)

    if st["vignette"]:   # the dream fades at the edges
        vy, vx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((vx - w / 2) / (w / 2)) ** 2 + ((vy - h / 2) / (h / 2)) ** 2)
        canvas *= (1.0 - st["vignette"] * np.clip(d - 0.35, 0, 1) ** 1.5)[..., None]
    return (np.clip(canvas, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------- TensorFlow DeepDream
def _free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0


def tensorflow_side():
    """The longest image side DeepDream can afford right now, or 0 if none."""
    free = _free_ram_gb()
    for need, side in TF_SIDES:
        if free >= need:
            return side
    return 0


def tensorflow_usable():
    """Whether the TensorFlow engine can run right now. The import check runs in a
    subprocess (a broken TF can crash the process) and is remembered for a day."""
    if tensorflow_side() == 0:
        return False
    cached = store.read_json(ENGINE_CACHE, None)
    if cached and time.time() - cached.get("ts", 0) < 86400:
        return bool(cached.get("tensorflow"))
    try:
        ok = subprocess.run([sys.executable, "-c", "import tensorflow as tf; tf.keras.applications.InceptionV3"],
                            capture_output=True, timeout=90).returncode == 0
    except (subprocess.SubprocessError, OSError):
        ok = False
    store.write_json(ENGINE_CACHE, {"ts": time.time(), "tensorflow": ok})
    return ok


def render_tensorflow(source_path, tone, out_path, stop=None):
    """DeepDream via the worker script, in its own process so it can be killed
    the moment she wakes. Returns True on success."""
    worker = Path(__file__).with_name("deepdream_worker.py")
    preset = TF_PRESETS.get(tone, TF_PRESETS["strange"])
    proc = subprocess.Popen([sys.executable, str(worker), str(source_path), str(out_path), json.dumps(preset), str(tensorflow_side() or 224)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + TF_TIMEOUT_S
    while proc.poll() is None:
        if (stop is not None and stop.is_set()) or time.time() > deadline:
            proc.terminate()
            return False
        time.sleep(0.5)
    return proc.returncode == 0 and Path(out_path).exists()


# ---------------------------------------------------------------- the dream world (latent_space.py)
def choose_mode(tone, cycle=0):
    """Later strange dreams alternate spiral and interpolate, so a night isn't one shape."""
    if tone == "strange" and cycle % 2 == 1:
        return "interpolate"
    return TONE_MODE.get(tone, "spiral")


def latent_walk(dream, seed, stop=None):
    """Run latent_space.py's engine for this dream. Returns (BGR frames, mode), or
    (None, None) if it can't run (no PyTorch, say)."""
    if cv2 is None or (stop is not None and stop.is_set()):
        return None, None
    try:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import latent_space as ls
    except Exception:
        return None, None
    mode = choose_mode(dream["tone"], dream.get("cycle", 0))
    ls.DEVICE = "cpu"                                   # small network; no need to touch the GPU
    ls.OUTPUT_DIR = images_dir() / "latent"
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(dream["ts"]))
    try:
        session = ls.dream(mode=mode, frames=LATENT_FRAMES, image_size=LATENT_SIZE, seed=seed,
                           step_size=0.3 if dream["tone"] == "nightmare" else ls.DEFAULT_STEP_SIZE,
                           session_tag=f"{stamp}_{dream.get('cycle', 0)}", verbose=False)
    except Exception as e:
        print(f"[mind] latent walk failed: {e}")
        return None, None
    try:
        frames = [cv2.imread(str(p)) for p in sorted(session.glob("frame_*.png"))]
    finally:
        shutil.rmtree(session, ignore_errors=True)     # keep the GIF and the still, not forty PNGs
    frames = [f for f in frames if f is not None]
    return (frames or None), mode


def grade(frame, tone):
    """Colour the dream world to match its mood. float32 in [0,1] BGR -> same."""
    if tone == "nightmare":
        out = frame * np.array([0.65, 0.7, 1.25], np.float32)[None, None, :] * 0.8      # dark, bleeding red
        return np.clip((out - 0.5) * 1.25 + 0.5, 0, 1)
    if tone == "pleasant":
        return np.clip(frame * 1.05 + np.array([0.02, 0.05, 0.08], np.float32)[None, None, :], 0, 1)   # warm
    return frame


def _save_gif(frames_bgr, path):
    from PIL import Image
    imgs = [Image.fromarray(cv2.cvtColor(cv2.resize(f, (GIF_SIZE, GIF_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            for f in frames_bgr]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=GIF_FRAME_MS, loop=0, optimize=True)


# ---------------------------------------------------------------- painting a dream
def paint(mind, dream, stop=None):
    """Paint `dream`. Returns the still image's file name (inside images_dir()), or
    None. Also sets dream["animation"] (a GIF) when the dream world could run."""
    if cv2 is None:
        return None
    source, described = choose_source(mind, dream)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(dream["ts"]))
    cycle = dream.get("cycle", 0)
    name = f"dream_{stamp}_{cycle}.jpg"
    out = images_dir() / name
    seed = _seed_for(dream["text"])

    frames, mode = latent_walk(dream, seed, stop)
    if frames:
        memory = render_opencv(source, dream["tone"], seed, stop)         # her own memory, painted
        if memory is None or (stop is not None and stop.is_set()):        # she woke up mid-dream
            return None
        size = (frames[0].shape[1], frames[0].shape[0])
        mem = cv2.resize(memory, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        world = [np.clip((1 - MEMORY_BLEED) * grade(f.astype(np.float32) / 255.0, dream["tone"]) + MEMORY_BLEED * mem, 0, 1)
                 for f in frames]
        world = [(w * 255).astype(np.uint8) for w in world]
        still = cv2.resize(world[int(len(world) * 0.7)], (LATENT_SIZE * 2, LATENT_SIZE * 2), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(out), still, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gif = f"dream_{stamp}_{cycle}.gif"
        try:
            _save_gif(world, images_dir() / gif)
            dream["animation"] = gif
        except Exception as e:
            print(f"[mind] couldn't save the dream animation: {e}")
        engine = "latent"
        if tensorflow_usable():                                            # the old pipeline: frames -> DeepDream
            if render_tensorflow(out, dream["tone"], out, stop):
                engine = "latent+tensorflow"
                dreamt = cv2.imread(str(out))            # DeepDream ran at whatever size the RAM allowed
                if dreamt is not None:
                    cv2.imwrite(str(out), cv2.resize(dreamt, (LATENT_SIZE * 2, LATENT_SIZE * 2), interpolation=cv2.INTER_CUBIC),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
        dream.update(image_engine=engine, image_mode=mode, image_source=described)
        return name

    # no PyTorch: the pure OpenCV dream
    engine = "opencv"
    if tensorflow_usable():
        tmp = images_dir() / f".src_{stamp}.jpg"
        try:
            cv2.imwrite(str(tmp), source)
            if render_tensorflow(tmp, dream["tone"], out, stop):
                engine = "tensorflow"
        finally:
            tmp.unlink(missing_ok=True)
    if engine == "opencv":
        img = render_opencv(source, dream["tone"], seed, stop)
        if img is None:
            return None
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    dream["image_engine"], dream["image_source"] = engine, described
    return name
