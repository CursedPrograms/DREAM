"""
dream_world.py - she learns what her life looks like while she's idle.

Her dream world (latent_space.py's DreamDecoder) starts as an untrained network:
pure abstract colour, and the spiral dreams keep it that way on purpose. But
while she's awake with nothing to do, she teaches a second copy of it her own
photos and painted dreams (dream_world_worker.py: a small GAN, in bursts). Once
it has learned enough, her pleasant dreams, nightmares and drifting dreams walk
through that learned world instead - still twisted by the same kaleidoscope.

A burst runs in its own low-priority process and is killed the moment she's
needed: a conversation, a photo to look at, or sleep.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from . import store

ENABLED = True                    # selftest turns this off
BURST_MINUTES = 5
REST_S = 15 * 60                  # between bursts
IDLE_AFTER_S = 180                # quiet this long before she starts
MIN_IMAGES = 12
READY_STEPS = 3000                # before this, the learned world is still mostly noise
WORKER = Path(__file__).with_name("dream_world_worker.py")
CKPT = "dream_world.pt"


def ckpt_path() -> Path:
    return store.mind_dir() / CKPT


def status():
    try:
        return json.loads(ckpt_path().with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"steps": 0, "images": 0}


def learned_checkpoint():
    """The generator latent_space.dream() can load, once it's learned enough; else None."""
    g = ckpt_path().with_name(ckpt_path().stem + "_G.pt")
    return str(g) if g.exists() and status().get("steps", 0) >= READY_STEPS else None


def training_images(mind):
    """Her photos and her painted dreams and visions."""
    paths = []
    try:
        paths += [o["path"] for o in mind.vision.observations(1000)]
    except Exception:
        pass
    d = store.IMAGES_DIR
    if d.exists():
        paths += [str(p) for p in d.glob("dream_*.jpg")] + [str(p) for p in d.glob("vision_*.jpg")]
    return [p for p in dict.fromkeys(paths) if Path(p).exists()]


class WorldTrainer:
    def __init__(self, mind):
        self.mind = mind
        self.proc = None
        self._last_end = 0.0

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def tick(self, now, idle):
        if self.proc is not None:
            if self.proc.poll() is not None:          # the burst finished
                self.proc, self._last_end = None, now
            elif not idle:                            # she's needed
                self.stop(now)
            return
        if not ENABLED or not idle or now - self._last_end < REST_S:
            return
        paths = training_images(self.mind)
        if len(paths) < MIN_IMAGES:
            self._last_end = now                      # check again after a rest
            return
        data = store.mind_dir() / "dream_world_images.json"
        data.write_text(json.dumps(paths), encoding="utf-8")
        store.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [sys.executable, str(WORKER), "--data", str(data), "--ckpt", str(ckpt_path()),
             "--minutes", str(BURST_MINUTES), "--preview", str(store.IMAGES_DIR / "dream_world_learning.jpg")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[mind] learning her dream world from {len(paths)} pictures (step {status().get('steps', 0)})")

    def stop(self, now=None):
        if self.running():
            self.proc.terminate()
        self.proc = None
        self._last_end = now or time.time()
