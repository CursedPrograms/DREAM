#!/usr/bin/env python3
# depth_effect.py
#
# A 2.5D parallax effect for the avatar videos, like After Effects' Displacement Map:
# each clip gets a depth map (MiDaS), and on playback near pixels are shifted more
# than far ones along a slow sway, so the flat video gains a sense of depth.
#
# Running MiDaS live is too slow, so the depth maps are computed once and cached
# as small grayscale videos in videos/depth/ (same name as the clip):
#
#   python scripts/depth_effect.py              build missing / outdated depth videos
#   python scripts/depth_effect.py --large      use MiDaS Large (better, ~20x slower)
#   python scripts/depth_effect.py --force      rebuild all
#
# Turn it on in config.json -> Config.DREAM.DepthEffect.Enabled. Only the looping
# state clips get the effect (one-shot clips play with audio). Clips without a
# cached depth video (e.g. fresh MuseTalk lipsync output) play without the effect.

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DEPTH_DIR = os.path.join(VIDEOS_DIR, "depth")

DEPTH_WIDTH = 480    # cached depth videos are small; they are blurred and scaled up anyway
SMOOTHING = 0.5      # blend with the previous depth frame, to stop MiDaS flickering between frames

# The looping state clips dream.py plays (its VIDEO_POOLS). One-shot clips (intro,
# lipsync, flirt) play with audio and are left flat, so they get no depth video.
LOOP_PREFIXES = ("idle", "listening", "thinking", "talking", "sleeping")

DEFAULTS = {
    "Enabled": False,
    "Strength": 14,   # max shift in pixels (at 1080p) for the nearest parts
    "Speed": 0.12,    # sway cycles per second
    "Scale": 1.0,     # resolution the warp runs at (0.25-1); lower is faster but softer
}


def depth_path(video_path):
    """videos/idle0.mp4 -> videos/depth/idle0.mp4"""
    return os.path.join(DEPTH_DIR, os.path.basename(video_path))


# ==================== PLAYBACK ====================

def _cuda_torch():
    """torch, if it can use the GPU; the warp then runs there (~3 ms plus transfers)."""
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


class DepthStream:
    """Reads a clip's cached depth video in step with the clip and warps its frames.
    Call apply() once per decoded frame and rewind() when the clip loops.

    The warp costs ~20-35 ms per 1080p frame, about as much as decoding it, so it runs
    on a worker thread: apply(frame N) hands N over and returns the warped frame N-1,
    and the warp overlaps with decoding the next frame (OpenCV and CUDA release the GIL).
    The price is one repeated frame when a clip starts."""

    def __init__(self, video_path, settings):
        self.cap = cv2.VideoCapture(depth_path(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open depth video for {os.path.basename(video_path)}")
        self.strength = float(settings.get("Strength", DEFAULTS["Strength"]))
        self.speed = float(settings.get("Speed", DEFAULTS["Speed"]))
        self.scale = min(1.0, max(0.25, float(settings.get("Scale", DEFAULTS["Scale"]))))
        self._torch = _cuda_torch()
        self._grid = None
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    def rewind(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _next_depth(self):
        ok, d = self.cap.read()
        if not ok:
            self.rewind()
            ok, d = self.cap.read()
        return d[:, :, 0] if ok else None

    def apply(self, frame):
        """frame: BGR or RGB uint8 -> a frame displaced by its depth (the previous one, see above)."""
        # The depth frame is read here, not on the worker, so it stays paired with its
        # video frame even when the clip rewinds
        d = self._next_depth()
        if d is None:
            return frame
        job = self._pool.submit(self._warp, frame, d, time.time())
        prev, self._pending = self._pending, job
        return (prev or job).result()

    def _offsets(self, d, h, phase):
        """Per-pixel (dx, dy) in pixels, at the small depth map's size.
        Soft edges stop the warp tearing at depth boundaries (hair against background).
        Depth is centred so mid-depth stays put and near/far move in opposite directions."""
        d = cv2.GaussianBlur(d, (0, 0), 3).astype(np.float32) * (1 / 255.) - 0.5
        amount = self.strength * h / 1080
        return d * (amount * math.cos(phase)), d * (amount * 0.5 * math.sin(phase))

    def _warp(self, frame, d, t):
        # Warping a smaller frame is much cheaper; the player scales it back up to the screen
        if self.scale < 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        dx, dy = self._offsets(d, h, 2 * math.pi * self.speed * t)
        if self._torch is not None:
            return self._warp_gpu(frame, dx, dy)

        # CPU: all the maths runs on the small map; only the final offsets are scaled up
        # (full-res float work was the slow part)
        if self._grid is None or self._grid.shape[:2] != (h, w):
            xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._grid = cv2.merge([xs, ys])
        offset = cv2.resize(cv2.merge([dx, dy]), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(frame, cv2.add(self._grid, offset), None, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def _warp_gpu(self, frame, dx, dy):
        torch = self._torch
        F = torch.nn.functional
        h, w = frame.shape[:2]
        if self._grid is None or tuple(self._grid.shape[1:3]) != (h, w):
            ys, xs = torch.meshgrid(torch.linspace(-1, 1, h, device="cuda"),
                                    torch.linspace(-1, 1, w, device="cuda"), indexing="ij")
            self._grid = torch.stack([xs, ys], -1)[None]
        with torch.no_grad():
            img = torch.from_numpy(frame).cuda().permute(2, 0, 1)[None].float()
            # Pixel offsets -> grid_sample's -1..1 units, scaled up to the frame size on the GPU
            off = torch.from_numpy(np.stack([dx * (2 / (w - 1)), dy * (2 / (h - 1))])).cuda()[None]
            off = F.interpolate(off, size=(h, w), mode="bilinear", align_corners=False)
            out = F.grid_sample(img, self._grid + off.permute(0, 2, 3, 1), mode="bilinear",
                                padding_mode="border", align_corners=True)
            return out[0].permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()

    def release(self):
        self._pool.shutdown(wait=True)
        self.cap.release()


def open_stream(video_path, settings):
    """A DepthStream for the clip, or None if the effect is off or it has no depth video."""
    if not settings.get("Enabled") or not os.path.exists(depth_path(video_path)):
        return None
    try:
        return DepthStream(video_path, settings)
    except Exception:
        return None


def missing_depth(video_paths):
    return [p for p in video_paths if not os.path.exists(depth_path(p))]


# ==================== CACHE BUILDING ====================

def build_depth_video(estimator, video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    iw, ih = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (DEPTH_WIDTH, max(2, round(ih * DEPTH_WIDTH / iw / 2) * 2))

    part = out_path + ".part.mp4"
    writer = cv2.VideoWriter(part, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    prev = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = estimator.estimate(frame, out_size=size)
        prev = d if prev is None else cv2.addWeighted(d, SMOOTHING, prev, 1 - SMOOTHING, 0)
        gray = (prev * 255).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        n += 1
        print(f"\r  {os.path.basename(video_path)}: {n}/{total}", end="", flush=True)
    print()
    cap.release()
    writer.release()

    if n == 0:
        os.remove(part)
        raise RuntimeError("no frames read")
    os.replace(part, out_path)


def build_cache(model="small", force=False):
    from monocular_depth import DepthEstimator, console

    videos = sorted(
        os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR)
        if f.endswith(".mp4") and f.startswith(LOOP_PREFIXES)
    )
    todo = [
        v for v in videos
        if force or not os.path.exists(depth_path(v))
        or os.path.getmtime(depth_path(v)) < os.path.getmtime(v)
    ]
    if not todo:
        console.print("[green]All depth videos are up to date.[/green]")
        return

    os.makedirs(DEPTH_DIR, exist_ok=True)
    estimator = DepthEstimator(model)
    console.print(f"[cyan]Building {len(todo)} depth video(s) with MiDaS {model.capitalize()} on {estimator.device}[/cyan]")

    start = time.time()
    for v in todo:
        try:
            build_depth_video(estimator, v, depth_path(v))
        except Exception as e:
            console.print(f"[red]{os.path.basename(v)} failed:[/red] {e}")
    console.print(f"[green]Done in {time.time() - start:.0f}s:[/green] {DEPTH_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Build the depth videos for the avatar depth effect")
    parser.add_argument("--large", action="store_true", help="use MiDaS Large (better, much slower)")
    parser.add_argument("--force", action="store_true", help="rebuild every depth video")
    args = parser.parse_args()
    build_cache("large" if args.large else "small", args.force)


if __name__ == "__main__":
    main()
