"""
dream_generator.py - paints her dream visions: SDXL-Turbo / SD-Turbo with diffusers, one job.

    python dream_generator.py --job <job.json> --model stabilityai/sd-turbo

dream_visions.py writes the job and runs this in its own process (so it can be killed the
moment she wakes, and its memory goes away with it). The job:

    prompt     the dream's words in its tone's style
    keyframe   where the vision goes: text-to-image from the prompt
    frames     [[latent-walk frame, where its re-imagined version goes], ...]: image-to-image
               with the same prompt, mixed with the vision more and more (up to `blend`), so
               the walk turns into the dream
    strength   how far each frame strays from the walk (0..1)
    size       picture size (512)
    seed       the same dream always looks the same

Device: the GPU if this process can see one. dream_visions.py hides it when config.json's
ImageDevice is cpu (the default), so dreaming never slows her conversation down.
The model downloads to Hugging Face's cache on the first run.
"""

import argparse
import json
import math
import sys

import numpy as np
import torch
from PIL import Image


def load(model):
    from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    try:   # the fp16 files are half the download; on the CPU they're loaded as fp32
        t2i = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype, variant="fp16")
    except (OSError, ValueError):              # no fp16 weights published for this model
        t2i = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    t2i = t2i.to("cuda" if cuda else "cpu")
    t2i.set_progress_bar_config(disable=True)
    i2i = AutoPipelineForImage2Image.from_pipe(t2i)   # shares the weights: no second copy
    return t2i, i2i, ("cuda" if cuda else "cpu")


def main(job_path, model):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    size, seed, prompt = int(job.get("size", 512)), int(job.get("seed", 0)), job["prompt"]
    t2i, i2i, device = load(model)
    gen = lambda k: torch.Generator(device).manual_seed(seed + k)

    # Turbo models: one step, no guidance (they're distilled for it)
    vision = t2i(prompt=prompt, num_inference_steps=1, guidance_scale=0.0, width=size, height=size,
                 generator=gen(0)).images[0]
    vision.save(job["keyframe"], quality=92)
    print(f"vision: {job['keyframe']}", flush=True)

    frames = job.get("frames", [])
    strength = float(job.get("strength", 0.55))
    steps = max(2, math.ceil(1 / max(strength, 0.05)))   # turbo img2img runs steps*strength steps: at least one
    blend = float(job.get("blend", 0.6))
    v = np.asarray(vision, dtype=np.float32)
    for k, (src, dst) in enumerate(frames):
        start = Image.open(src).convert("RGB").resize((size, size))
        out = i2i(prompt=prompt, image=start, strength=strength, num_inference_steps=steps,
                  guidance_scale=0.0, generator=gen(k + 1)).images[0]
        a = blend * (k + 1) / len(frames)                       # more of the vision each frame
        mixed = (1 - a) * np.asarray(out, dtype=np.float32) + a * v
        Image.fromarray(mixed.clip(0, 255).astype(np.uint8)).save(dst, quality=90)
        print(f"frame {k + 1}/{len(frames)}", flush=True)
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--job", required=True)
    p.add_argument("--model", default="stabilityai/sd-turbo")
    a = p.parse_args()
    sys.exit(main(a.job, a.model))
