#!/usr/bin/env python3
# monocular_depth.py
#
# Real-time depth from a single camera using MiDaS v2.1 (OpenCV DNN).
# Shows the feed next to a depth map (brighter = closer). Press q or Esc to quit.
#
#   python scripts/monocular_depth.py                  webcam
#   python scripts/monocular_depth.py --image a.jpg    one image (saves the depth map)
#   python scripts/monocular_depth.py --video a.mp4    a video file
#   python scripts/monocular_depth.py --small          force the faster, lower quality model
#   python scripts/monocular_depth.py --large          force the better, slower model
#
# By default images use MiDaS Large; live video uses Large on a CUDA build of OpenCV
# and Small on the CPU. The model is downloaded to scripts/models/ on first use.
# Only one program can hold the webcam: don't run this while dream.py's surveillance is on.

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pygame
import requests
from rich.console import Console

console = Console()

IS_WINDOWS = sys.platform == "win32"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAVE_DIR = os.path.join(BASE_DIR, "output", "depth")

MODEL_URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/{}"
# name -> (file, input size)
MODELS = {
    "large": ("model-f6b98070.onnx", 384),  # MiDaS v2.1 Large
    "small": ("model-small.onnx", 256),     # MiDaS v2.1 Small
}
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def download_model(file_name):
    """Download a MiDaS model to scripts/models/ if it is not there yet."""
    path = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(path):
        return path

    os.makedirs(MODEL_DIR, exist_ok=True)
    part = path + ".part"
    console.print(f"[cyan]Downloading {file_name}...[/cyan]")
    try:
        with requests.get(MODEL_URL.format(file_name), stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / total:6.1%}  {done >> 20} / {total >> 20} MB", end="", flush=True)
        print()
        os.replace(part, path)
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        console.print(f"[red]Download failed:[/red] {e}")
        sys.exit(1)

    console.print(f"[green]Saved:[/green] {path}")
    return path


class DepthEstimator:
    """MiDaS v2.1 depth. estimate(frame) -> float32 map in 0..1, brighter = closer."""

    def __init__(self, model="large"):
        file_name, self.input_size = MODELS[model]
        path = download_model(file_name)
        self.net = self.session = None

        # OpenCV on the GPU if it was built with CUDA, else onnxruntime (about 2x
        # faster than OpenCV on the CPU), else OpenCV on the CPU
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.device = "CUDA"
            self.net = cv2.dnn.readNet(path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            return
        try:
            import onnxruntime
            self.session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.device = "CPU, onnxruntime"
        except ImportError:
            self.device = "CPU"
            self.net = cv2.dnn.readNet(path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def estimate(self, frame, out_size=None):
        """out_size (w, h) of the returned map; defaults to the frame size."""
        h, w = frame.shape[:2]
        # MiDaS v2.1: RGB (swapRB converts the BGR frame), (x / 255 - mean) / std with ImageNet mean and std.
        # blobFromImage only does the mean part, so divide by std after; without it the depth comes out wrong.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255., (self.input_size, self.input_size),
                                     (123.675, 116.28, 103.53), True, False)
        blob /= IMAGENET_STD
        if self.session is not None:
            depth = self.session.run(None, {self.input_name: blob})[0][0]
        else:
            self.net.setInput(blob)
            depth = self.net.forward()[0, :, :]
        depth = cv2.resize(depth, out_size or (w, h))
        return cv2.normalize(depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def colorize(depth):
    """0..1 depth map -> BGR image for display or saving."""
    return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


class Viewer:
    """pygame window (the venv has opencv-python-headless, so no cv2.imshow).
    Shows BGR images scaled to fit the screen. Closing it, q or Esc quits."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("DREAM Depth")
        info = pygame.display.Info()
        self.max_w, self.max_h = int(info.current_w * 0.9), int(info.current_h * 0.8)
        self.screen = None

    def show(self, bgr):
        h, w = bgr.shape[:2]
        scale = min(1.0, self.max_w / w, self.max_h / h)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        if self.screen is None or self.screen.get_size() != (w, h):
            self.screen = pygame.display.set_mode((w, h))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.screen.blit(pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB"), (0, 0))
        pygame.display.flip()

    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_q, pygame.K_ESCAPE):
                return True
        return False

    def close(self):
        pygame.quit()


def run_image(estimator, image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        console.print(f"[red]Could not read image:[/red] {image_path}")
        sys.exit(1)

    depth = colorize(estimator.estimate(frame))

    os.makedirs(SAVE_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(SAVE_DIR, f"{name}_depth.png")
    cv2.imwrite(out_path, depth)
    console.print(f"[green]Depth map saved:[/green] {out_path}")

    viewer = Viewer()
    viewer.show(np.hstack([frame, depth]))
    while not viewer.should_quit():
        time.sleep(0.05)
    viewer.close()


def run_stream(estimator, source):
    if source is None:
        # On Windows, CAP_DSHOW avoids a long timeout when no camera is present.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        console.print("[red]Could not open the camera or video[/red] (is dream.py's surveillance using it?)")
        sys.exit(1)

    console.print("[green]Running.[/green] Press q or Esc in the window to quit.")
    viewer = Viewer()
    while not viewer.should_quit():
        ok, frame = cap.read()
        if not ok:
            break

        start = time.time()
        depth = colorize(estimator.estimate(frame))
        fps = 1 / max(time.time() - start, 1e-6)

        cv2.putText(frame, f"{fps:.1f} FPS ({estimator.device})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        viewer.show(np.hstack([frame, depth]))

    cap.release()
    viewer.close()


def main():
    parser = argparse.ArgumentParser(description="Monocular depth estimation (MiDaS v2.1)")
    parser.add_argument("--image", help="estimate depth for one image and save it")
    parser.add_argument("--video", help="use a video file instead of the webcam")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--small", action="store_true", help="use the faster MiDaS Small model")
    size.add_argument("--large", action="store_true", help="use the better MiDaS Large model")
    args = parser.parse_args()

    # Default: Large for single images or on the GPU; Small for live video on the CPU,
    # where Large only manages ~0.4 FPS (Small ~4 FPS)
    if args.small or args.large:
        model = "small" if args.small else "large"
    else:
        model = "large" if args.image or cv2.cuda.getCudaEnabledDeviceCount() > 0 else "small"

    estimator = DepthEstimator(model)
    console.print(f"[cyan]MiDaS {model.capitalize()} on {estimator.device}[/cyan]")

    if args.image:
        run_image(estimator, args.image)
    else:
        run_stream(estimator, args.video)


if __name__ == "__main__":
    main()
