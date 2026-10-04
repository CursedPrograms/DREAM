"""
dream_world_worker.py - one burst of teaching her dream world what her life looks like.

Run by dream_world.py in its own low-priority process while she's idle; killed
the moment she's needed. Trains latent_space.py's DreamDecoder as the generator
of a small GAN (the idea SynthWomb's trainer had, in PyTorch) on her photos and
painted dreams, and saves every SAVE_EVERY steps so being killed costs little.

    python dream_world_worker.py --data images.json --ckpt dream_world.pt --minutes 5 --preview learning.jpg

The checkpoint holds both networks and their optimisers (to pick up where the
last burst stopped); <ckpt>_G.pt is the generator alone, in the shape
latent_space.dream(checkpoint=...) loads. The kaleidoscope twirl isn't trained:
DreamDecoder adds it on top when she dreams, so the learned shapes still swirl.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/, for latent_space
from latent_space import DreamDecoder  # noqa: E402

SIZE = 64            # DreamDecoder's own resolution before it scales up (8 -> 16 -> 32 -> 64)
LATENT = 128
SAVE_EVERY = 100
MAX_IMAGES = 1500


def lower_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def load_images(paths):
    """Each picture as a few random crops, so a small album goes further."""
    out = []
    for p in paths[:MAX_IMAGES]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for _ in range(3):
            s = int(min(h, w) * random.uniform(0.6, 1.0))
            y, x = random.randint(0, h - s), random.randint(0, w - s)
            crop = cv2.resize(img[y:y + s, x:x + s], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            out.append(crop)
    if not out:
        return None
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0


def augment(x):
    """The same random flips, shifts and colour changes for real and fake, so the
    critic can't just memorise a tiny album (DiffAugment's trick)."""
    if random.random() < 0.5:
        x = x.flip(3)
    dx, dy = random.randint(-6, 6), random.randint(-6, 6)
    x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    b = (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5) * 0.3
    return (x + b).clamp(0, 1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2),      # 32
            sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),    # 16
            sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),   # 8
            sn(nn.Conv2d(256, 1, 8, 1, 0)),
        )

    def forward(self, x):
        return self.net(x * 2 - 1).view(-1)


def generate(G, z):
    """DreamDecoder without the twirl, at its own 64 px."""
    return G.conv(G.fc(z).view(-1, 256, 8, 8))


def fresh_generator():
    G = DreamDecoder(latent_dim=LATENT, image_size=SIZE)
    for m in G.modules():   # its glitch init makes art, not a trainable GAN
        if isinstance(m, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return G


def save(path, G, D, oG, oD, steps, n_images):
    tmp = path.with_suffix(".tmp")
    torch.save({"G": G.state_dict(), "D": D.state_dict(), "oG": oG.state_dict(), "oD": oD.state_dict(), "steps": steps}, tmp)
    os.replace(tmp, path)
    g_path = path.with_name(path.stem + "_G.pt")
    torch.save(G.state_dict(), g_path.with_suffix(".tmp"))
    os.replace(g_path.with_suffix(".tmp"), g_path)
    status = {"steps": steps, "images": n_images, "ts": time.time()}
    path.with_suffix(".json").write_text(json.dumps(status), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--preview")
    args = ap.parse_args()
    lower_priority()

    paths = json.loads(Path(args.data).read_text(encoding="utf-8"))
    data = load_images(paths)
    if data is None:
        print("no usable pictures")
        return 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G, D = fresh_generator().to(dev), Critic().to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    ckpt, steps = Path(args.ckpt), 0
    if ckpt.exists():
        try:
            s = torch.load(ckpt, map_location=dev)
            G.load_state_dict(s["G"]); D.load_state_dict(s["D"])
            oG.load_state_dict(s["oG"]); oD.load_state_dict(s["oD"])
            steps = s.get("steps", 0)
        except Exception as e:   # a different shape after an update: start over
            print(f"starting fresh ({e})")
    G.train(); D.train()
    data = data.to(dev)
    batch = min(32, len(data))
    deadline = time.time() + args.minutes * 60
    print(f"training on {len(paths)} pictures ({len(data)} crops), from step {steps}, on {dev}", flush=True)

    while time.time() < deadline:
        real = augment(data[torch.randint(0, len(data), (batch,), device=dev)])
        fake = generate(G, torch.randn(batch, LATENT, device=dev))
        loss_d = F.softplus(-D(real)).mean() + F.softplus(D(augment(fake.detach()))).mean()
        oD.zero_grad(set_to_none=True); loss_d.backward(); oD.step()
        loss_g = F.softplus(-D(augment(fake))).mean()
        oG.zero_grad(set_to_none=True); loss_g.backward(); oG.step()
        steps += 1
        if steps % SAVE_EVERY == 0:
            save(ckpt, G, D, oG, oD, steps, len(paths))
    save(ckpt, G, D, oG, oD, steps, len(paths))

    if args.preview:   # what she's learned so far: a 4x4 sheet of her dream world
        G.eval()
        with torch.no_grad():
            imgs = generate(G, torch.randn(16, LATENT, device=dev)).cpu().permute(0, 2, 3, 1).numpy()
        rows = [np.concatenate(list(imgs[r * 4:(r + 1) * 4]), axis=1) for r in range(4)]
        sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
        sheet = cv2.resize(sheet, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(args.preview, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"done at step {steps}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
