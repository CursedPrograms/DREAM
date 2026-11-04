"""
CPU or GPU for DREAM, from config.json (Config/DREAM). Three settings, each cuda, cpu or auto:

    "Device":      "cuda"   the conversation loop: Whisper (wake word and commands), the
                            LLM (Ollama) and lip-sync - what you wait on when she answers
    "ImageDevice": "cpu"    image work: her looks (the vision model), image generation, her
                            dream world, DeepDream, depth and face recognition - kept off the
                            GPU so it never slows or unloads the loop's models
    "PiperDevice": "cpu"    her voice. cuda measured slower (see piper_args)

cuda without a working GPU falls back to the CPU (and says so once); auto is the GPU when
there is one. Every DREAM (app.py, dream_app.py, scripts/dream.py, webserver.py, the mind,
lip-sync, cpp_dream) asks this module instead of deciding for itself.

    python scripts/dream_device.py      print what config.json picks
"""

import json
import os

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

CHOICES = ["cuda", "cpu", "auto"]
SETTINGS = {           # kind -> (config.json key, default)
    "loop":  ("Device", "cuda"),
    "image": ("ImageDevice", "cpu"),
    "piper": ("PiperDevice", "cpu"),
}

_warned = set()


def choice(kind="loop"):
    """What config.json asks for: cuda, cpu or auto."""
    key, default = SETTINGS[kind]
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            chosen = json.load(f)["Config"]["DREAM"].get(key, default)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return default
    if chosen not in CHOICES:
        print(f"config.json: {key} '{chosen}' isn't one of {', '.join(CHOICES)} - using '{default}'")
        return default
    return chosen


def _gpu_present():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:  # no torch (e.g. a light venv): onnxruntime-gpu can still tell
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


def use_gpu(kind="loop"):
    """True when this kind of work should run on the GPU (asked for, and there is one)."""
    chosen = choice(kind)
    if chosen == "cpu":
        return False
    if _gpu_present():
        return True
    if chosen == "cuda" and kind not in _warned:
        _warned.add(kind)
        print(f"config.json: {SETTINGS[kind][0]} is 'cuda' but no CUDA GPU was found - using the CPU")
    return False


def torch_device(kind="loop"):
    """"cuda" or "cpu", for torch (.to(), device=...)."""
    return "cuda" if use_gpu(kind) else "cpu"


def onnx_providers(kind="loop"):
    """onnxruntime providers, best first."""
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu(kind) else ["CPUExecutionProvider"]


def ollama_options(kind="loop", **options):
    """Ollama "options" with the device applied: on the CPU num_gpu 0; on the GPU Ollama puts
    as many layers on it as fit (all of them for DREAM's 4B models on a 6 GB card)."""
    if not use_gpu(kind):
        options["num_gpu"] = 0
    return options


def piper_args():
    """Piper's flags for PiperDevice. Measured 2026-11-04 on an RTX 2060 with onnxruntime-gpu:
    --cuda took 4.8 s for a reply the CPU voices in 2.8 s (DREAM starts Piper per line and
    CUDA's start-up costs more than the small voice model saves), hence the cpu default.
    cuda only does anything with onnxruntime-gpu installed; without it Piper stays on the CPU."""
    if not use_gpu("piper"):
        return []
    try:
        import onnxruntime
        return ["--cuda"] if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else []
    except ImportError:
        return []


def opencv_dnn(kind="image"):
    """(backend, target) for OpenCV's DNN models (YuNet, SFace): CUDA when wanted and this
    OpenCV was built with CUDA, else the CPU. The pip opencv-python is CPU only."""
    import cv2
    if use_gpu(kind) and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA
    return cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_TARGET_CPU


def subprocess_env(kind="image"):
    """Environment for a helper process (image generator, DeepDream, the dream-world trainer):
    the GPU is hidden from it when this kind of work belongs on the CPU, whatever it does inside."""
    env = dict(os.environ)
    if not use_gpu(kind):
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def describe(kind="loop"):
    return "GPU (CUDA)" if use_gpu(kind) else "CPU"


if __name__ == "__main__":
    for kind, (key, _) in SETTINGS.items():
        print(f"{key:12} {choice(kind):5} -> {describe(kind)}")
