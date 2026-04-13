#!/usr/bin/env python3
"""
latent_space.py — DREAM's dreaming engine.

When DREAM sleeps, she wanders through latent space and saves what she sees
as images. Each run is a dream session: a continuous walk through the void,
rendered frame by frame.

Modes:
  random_walk   — Brownian motion through latent space (default)
  interpolate   — Smooth drift between random anchor points
  spiral        — Spiral outward from origin
  pulse         — Breathe in and out from a fixed point

Output: images saved to output/dreams/
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Platform ───────────────────────────────────────────────────────────────────
IS_WINDOWS = sys.platform == "win32"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "dreams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_LATENT_DIM  = 128     # size of the latent vector
DEFAULT_IMAGE_SIZE  = 256     # output image resolution (square)
DEFAULT_FRAMES      = 60      # how many frames per dream session
DEFAULT_STEP_SIZE   = 0.08    # how far she moves each step (random walk)
DEFAULT_FPS         = 12      # playback fps written to metadata
DEFAULT_MODE        = "interpolate"


class DreamDecoder(nn.Module):
    def __init__(self, latent_dim=128, image_size=256):
        super().__init__()
        # CRITICAL: We must save this so the 'forward' method can see it!
        self.image_size = image_size
        self.latent_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.Hardswish(),
            nn.Linear(2048, 8 * 8 * 256),
            nn.Hardswish(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid() 
        )
        self.apply(self._vivid_glitch_init)

    def _vivid_glitch_init(self, m):
        if isinstance(m, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, mean=0.0, std=0.6)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -3.0, 3.0)

    def forward(self, z):
        # 1. Base image generation
        x = self.fc(z).view(-1, 256, 8, 8)
        img = self.conv(x)
        
        # Resize to the final target size
        img = torch.nn.functional.interpolate(img, size=(self.image_size, self.image_size), mode='bilinear')
        
        # 2. The Kaleidoscopic Math (The "Wow" part)
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size), 
            torch.linspace(-1, 1, self.image_size), 
            indexing='ij'
        )
        grid_x, grid_y = grid_x.to(z.device), grid_y.to(z.device)
        
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        angle = torch.atan2(grid_y, grid_x)
        
        # Use a value from the latent vector to drive the "twirl"
        drift = z[0, 0].item() if z.dim() > 1 else z[0].item()
        warp = torch.sin(radius * 25.0 + angle * 6.0 + drift * 5.0)
        
        # Apply the kaleidoscopic intensity
        return torch.clamp(img * (0.5 + 0.5 * warp), 0, 1)

# ── FIX THIS FUNCTION TOO ───────────────────────────────────────────────────





# ── Latent space walkers ────────────────────────────────────────────────────────

def random_walk(start_z, frames, step_size):
    """Brownian motion — small random nudges each frame."""
    z = start_z.copy()
    trajectory = [z.copy()]
    for _ in range(frames - 1):
        noise = np.random.randn(*z.shape) * step_size
        z = z + noise
        trajectory.append(z.copy())
    return trajectory

def interpolate_walk(latent_dim, frames, num_anchors=6):
    """
    Pick random anchor points and smoothly drift between them.
    Cosine interpolation for dreamlike easing.
    """
    anchors = [np.random.randn(latent_dim) for _ in range(num_anchors)]
    anchors.append(anchors[0])  # loop back to start

    trajectory = []
    frames_per_segment = max(frames // (len(anchors) - 1), 1)

    for i in range(len(anchors) - 1):
        a, b = anchors[i], anchors[i + 1]
        for t in range(frames_per_segment):
            alpha = t / frames_per_segment
            # cosine easing
            alpha = (1 - math.cos(alpha * math.pi)) / 2
            z = (1 - alpha) * a + alpha * b
            trajectory.append(z)
            if len(trajectory) >= frames:
                return trajectory

    return trajectory[:frames]

def spiral_walk(latent_dim, frames, max_radius=3.0):
    """Spiral outward from origin in a random 2D plane within latent space."""
    # Pick two orthogonal basis vectors in latent space
    v1 = np.random.randn(latent_dim)
    v1 /= np.linalg.norm(v1)
    v2 = np.random.randn(latent_dim)
    v2 -= v2.dot(v1) * v1
    v2 /= np.linalg.norm(v2)

    trajectory = []
    for i in range(frames):
        t      = i / frames
        angle  = t * 4 * math.pi          # 2 full rotations
        radius = t * max_radius
        z      = radius * (math.cos(angle) * v1 + math.sin(angle) * v2)
        trajectory.append(z)
    return trajectory

def pulse_walk(latent_dim, frames, min_r=0.5, max_r=3.0):
    """Breathe in and out from a fixed direction — meditative."""
    direction = np.random.randn(latent_dim)
    direction /= np.linalg.norm(direction)

    trajectory = []
    for i in range(frames):
        t      = i / frames
        # sine breathing: 0→1→0→1…
        breath = (math.sin(t * 4 * math.pi) + 1) / 2
        radius = min_r + breath * (max_r - min_r)
        z      = direction * radius
        trajectory.append(z)
    return trajectory


# ── Tensor → image ─────────────────────────────────────────────────────────────

def tensor_to_image(tensor, image_size):
    """Convert decoder output (0 to 1) to uint8 RGB."""
    arr = tensor.squeeze(0).detach().cpu().numpy()   # (3, H, W)
    # REMOVED: (arr + 1) / 2  <-- This was squashing your colors to grey!
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)                     # (H, W, 3)
    return arr

def save_image(arr, path):
    """Save RGB numpy array as image — prefers PIL, falls back to cv2."""
    if PIL_AVAILABLE:
        Image.fromarray(arr).save(path)
    elif CV2_AVAILABLE:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
    else:
        raise RuntimeError("Neither Pillow nor OpenCV is available. Install one: pip install Pillow")


# ── Dream session ──────────────────────────────────────────────────────────────

def dream(
    mode        = DEFAULT_MODE,
    frames      = DEFAULT_FRAMES,
    latent_dim  = DEFAULT_LATENT_DIM,
    image_size  = DEFAULT_IMAGE_SIZE,
    step_size   = DEFAULT_STEP_SIZE,
    checkpoint  = None,
    seed        = None,
    session_tag = None,
    verbose     = True,
):
    """
    Run one dream session. Returns the output directory path.

    Args:
        mode        : 'random_walk' | 'interpolate' | 'spiral' | 'pulse'
        frames      : number of images to generate
        latent_dim  : dimensionality of latent space
        image_size  : output image resolution (square)
        step_size   : step size for random_walk mode
        checkpoint  : path to a saved decoder .pt checkpoint (optional)
        seed        : random seed for reproducibility
        session_tag : name tag for the output folder (auto-generated if None)
        verbose     : print progress
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required: pip install torch")
    if not PIL_AVAILABLE and not CV2_AVAILABLE:
        raise RuntimeError("Install Pillow or OpenCV: pip install Pillow")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ── Session folder ─────────────────────────────────────────────────────────
    tag        = session_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"dream_{tag}_{mode}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n🌙  DREAM is dreaming...")
        print(f"    Mode       : {mode}")
        print(f"    Frames     : {frames}")
        print(f"    Latent dim : {latent_dim}")
        print(f"    Image size : {image_size}x{image_size}")
        print(f"    Device     : {DEVICE}")
        print(f"    Output     : {session_dir}\n")

    # ── Decoder ────────────────────────────────────────────────────────────────
    decoder = DreamDecoder(latent_dim=latent_dim, image_size=image_size).to(DEVICE)
    decoder.eval()

    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=DEVICE)
        decoder.load_state_dict(state)
        if verbose:
            print(f"    Checkpoint : {checkpoint}")
    else:
        if verbose:
            print(f"    Checkpoint : none — using random weights (abstract art mode)\n")

    # ── Trajectory ─────────────────────────────────────────────────────────────
    start_z = np.random.randn(latent_dim)

    if mode == "random_walk":
        trajectory = random_walk(start_z, frames, step_size)
    elif mode == "interpolate":
        trajectory = interpolate_walk(latent_dim, frames)
    elif mode == "spiral":
        trajectory = spiral_walk(latent_dim, frames)
    elif mode == "pulse":
        trajectory = pulse_walk(latent_dim, frames)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: random_walk, interpolate, spiral, pulse")

    # ── Render frames ──────────────────────────────────────────────────────────
    paths = []
    with torch.no_grad():
        for i, z_np in enumerate(trajectory):
            z      = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            output = decoder(z)
            img    = tensor_to_image(output, image_size)
            fname  = session_dir / f"frame_{i:04d}.png"
            save_image(img, fname)
            paths.append(str(fname))

            if verbose:
                bar  = "█" * int((i + 1) / frames * 30)
                rest = "░" * (30 - len(bar))
                print(f"\r    [{bar}{rest}] {i+1}/{frames}", end="", flush=True)

    if verbose:
        print(f"\n\n✨  Dream complete — {len(paths)} frames saved to:\n    {session_dir}\n")

    # ── Session metadata ───────────────────────────────────────────────────────
    meta = {
        "session":    tag,
        "mode":       mode,
        "frames":     frames,
        "latent_dim": latent_dim,
        "image_size": image_size,
        "step_size":  step_size,
        "seed":       seed,
        "device":     DEVICE,
        "checkpoint": checkpoint,
        "fps":        DEFAULT_FPS,
        "timestamp":  datetime.now().isoformat(),
        "files":      paths,
    }
    with open(session_dir / "session.json", "w") as f:
        json.dump(meta, f, indent=2)

    return session_dir


# ── Stitch frames into a video (optional) ─────────────────────────────────────

def frames_to_video(session_dir, fps=DEFAULT_FPS, output_name="dream.mp4"):
    """
    Stitch all PNG frames in a session directory into a video.
    Requires OpenCV.
    """
    if not CV2_AVAILABLE:
        print("OpenCV not available — skipping video stitch. Install: pip install opencv-python")
        return None

    session_dir = Path(session_dir)
    frames_list = sorted(session_dir.glob("frame_*.png"))
    if not frames_list:
        print("No frames found.")
        return None

    sample = cv2.imread(str(frames_list[0]))
    h, w   = sample.shape[:2]

    if IS_WINDOWS:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ext    = ".mp4"
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        ext    = ".avi"

    out_path = session_dir / (output_name.rsplit(".", 1)[0] + ext)
    writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for fp in frames_list:
        frame = cv2.imread(str(fp))
        writer.write(frame)

    writer.release()
    print(f"🎞️  Video saved: {out_path}")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DREAM latent space explorer — generates images while she dreams."
    )
    parser.add_argument("--mode",       default=DEFAULT_MODE,
                        choices=["random_walk", "interpolate", "spiral", "pulse"],
                        help="Walk mode (default: interpolate)")
    parser.add_argument("--frames",     type=int,   default=DEFAULT_FRAMES,    help="Number of frames")
    parser.add_argument("--latent-dim", type=int,   default=DEFAULT_LATENT_DIM, help="Latent vector size")
    parser.add_argument("--image-size", type=int,   default=DEFAULT_IMAGE_SIZE, help="Output image resolution")
    parser.add_argument("--step-size",  type=float, default=DEFAULT_STEP_SIZE,  help="Step size (random_walk)")
    parser.add_argument("--checkpoint", type=str,   default=None,              help="Path to decoder checkpoint .pt")
    parser.add_argument("--seed",       type=int,   default=None,              help="Random seed")
    parser.add_argument("--tag",        type=str,   default=None,              help="Session name tag")
    parser.add_argument("--video",      action="store_true",                   help="Stitch frames into a video after")
    parser.add_argument("--fps",        type=int,   default=DEFAULT_FPS,       help="FPS for video output")
    parser.add_argument("--all-modes",  action="store_true",                   help="Run all 4 modes in sequence")

    args = parser.parse_args()

    if args.all_modes:
        for mode in ["random_walk", "interpolate", "spiral", "pulse"]:
            session_dir = dream(
                mode        = mode,
                frames      = args.frames,
                latent_dim  = args.latent_dim,
                image_size  = args.image_size,
                step_size   = args.step_size,
                checkpoint  = args.checkpoint,
                seed        = args.seed,
                session_tag = args.tag,
            )
            if args.video:
                frames_to_video(session_dir, fps=args.fps)
    else:
        session_dir = dream(
            mode        = args.mode,
            frames      = args.frames,
            latent_dim  = args.latent_dim,
            image_size  = args.image_size,
            step_size   = args.step_size,
            checkpoint  = args.checkpoint,
            seed        = args.seed,
            session_tag = args.tag,
        )
        if args.video:
            frames_to_video(session_dir, fps=args.fps)


if __name__ == "__main__":
    main()
