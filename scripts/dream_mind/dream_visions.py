"""
dream_visions.py - her dreams, seen.

dream_images.py paints a dream from her memories and walks latent space for it,
but neither knows what the dream *says*. This hands the dream's own words to her
image generator (dream_generator.py, next to this file: SDXL-Turbo / SD-Turbo):

  vision   text-to-image from the dream itself -> vision_<stamp>_<cycle>.jpg
  frames   every few frames of the latent walk, re-imagined with the same
           prompt (image-to-image), so the dream world takes on its subject
  video    when she wakes, every frame of the night - the walks and the
           visions, dream by dream - becomes dreams_<stamp>.mp4

All of it lives in the images folder (output/dreams). The generator runs in its
own process with DREAM's Python, so it can be killed the moment she wakes and
its memory goes away with it. Which model: config.json's DREAM.DreamImageModel -
"sdxl-turbo" (best), "sd-turbo" (a quarter the size: 4 GB GPU, 8 GB RAM), any
other Hugging Face id, or "auto": SDXL-Turbo when the PC can take it, else SD-Turbo. On the
GPU that's 12 GB of RAM or more; on the CPU (ImageDevice cpu) 32 GB or more and SDXL-Turbo
already in models/ (download_weights.bat), since it loads as fp32 (~14 GB) and paints a few
times slower - fine while she sleeps, and the CPU has nothing else to do then.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for dream_device
import dream_device

try:
    import cv2
except ImportError:
    cv2 = None

VISUALIZE = True                  # selftest turns this off: it shouldn't need a GPU and a model download
GENERATOR = Path(__file__).with_name("dream_generator.py")   # part of DREAM (was the separate Image-Generator repo)
VISION_SIZE = 512
VISION_FRAMES = 8                 # latent-walk frames re-imagined per dream
STRENGTH = {"pleasant": 0.45, "strange": 0.55, "nightmare": 0.6}   # how far each frame strays from the walk
BLEND = 0.6                       # frames are mixed with the vision, up to this much by the last: the walk turns into the dream
MIN_FREE_RAM_GB = 2.5
MIN_FREE_RAM_SDXL_CPU_GB = 16   # SDXL as fp32 on the CPU: ~14 GB, plus room
CPU_SDXL_RAM_GB = 32            # auto picks SDXL on the CPU only with this much RAM
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


def _local(repo_id):
    """DREAM's own copy of a model, models/<name> (e.g. models/sd-turbo), if it's there."""
    d = store.ROOT / "models" / repo_id.rsplit("/", 1)[-1]
    return str(d) if (d / "model_index.json").exists() else repo_id


def model():
    """The image model: DREAM's own copy in models/ when it has one, else the Hugging Face id
    (downloaded to Hugging Face's cache on the first dream)."""
    return _local(_model_id(_chosen()))


def _chosen():
    try:
        with open(store.ROOT / "config.json", encoding="utf-8") as f:
            return json.load(f)["Config"]["DREAM"].get("DreamImageModel", "auto")
    except (OSError, ValueError, KeyError):
        return "auto"


def _model_id(chosen):
    if chosen and chosen != "auto":
        return chosen if "/" in chosen else f"stabilityai/{chosen}"
    try:
        import psutil
        total = psutil.virtual_memory().total / 1e9
    except Exception:
        total = 0
    if not dream_device.use_gpu("image"):
        # the CPU: SDXL needs ~14 GB as fp32 and is a few times slower, so only on a big-RAM PC
        # that already has it (auto never starts a 7 GB download on its own for the CPU)
        has_sdxl = (store.ROOT / "models" / "sdxl-turbo" / "model_index.json").exists()
        return "stabilityai/sdxl-turbo" if total >= CPU_SDXL_RAM_GB and has_sdxl else "stabilityai/sd-turbo"
    return "stabilityai/sdxl-turbo" if total >= 12 else "stabilityai/sd-turbo"


# the files a diffusers pipeline needs (fp16 weights: half the download; the CPU loads them as fp32)
_PATTERNS = ["*.json", "*.txt", "*fp16.safetensors", "tokenizer/*"]


def _backup_dir():
    """config.json's ModelBackup (a folder of spare copies, e.g. F:/AIModels), or None."""
    try:
        with open(store.ROOT / "config.json", encoding="utf-8") as f:
            d = json.load(f)["Config"]["DREAM"].get("ModelBackup", "")
    except (OSError, ValueError, KeyError):
        d = ""
    return Path(d) if d else None


def download():
    """Put the image model in DREAM's models/ (download_weights.bat runs this): copied from
    ModelBackup if it's there, else downloaded from Hugging Face and then backed up there."""
    import shutil
    repo = _model_id(_chosen())
    name = repo.rsplit("/", 1)[-1]
    local = store.ROOT / "models" / name
    if (local / "model_index.json").exists():
        print(f"  Dream image model ({name}): already there")
        return True
    backup = _backup_dir()
    if backup and (backup / name / "model_index.json").exists():
        print(f"  Dream image model ({name}): copying from {backup / name} ...")
        shutil.copytree(backup / name, local, dirs_exist_ok=True)
        return True
    print(f"  Dream image model ({name}): downloading ...")
    # Hugging Face's Xet transfer hung at 0 MB here (2026-11-04); the plain HTTP download doesn't
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=str(local), allow_patterns=_PATTERNS)
    if backup and backup.exists():
        print(f"  backing it up to {backup / name} ...")
        shutil.copytree(local, backup / name, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    return True


def _free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0


def usable():
    """Whether the generator can run right now. The import check runs once a day
    in a subprocess (a broken torch can take the process down with it)."""
    need = MIN_FREE_RAM_GB
    if "sdxl" in model().lower() and not dream_device.use_gpu("image"):
        need = MIN_FREE_RAM_SDXL_CPU_GB
    if not VISUALIZE or not GENERATOR.exists() or _free_ram_gb() < need:
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
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env=dream_device.subprocess_env("image"))   # ImageDevice in config.json
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
