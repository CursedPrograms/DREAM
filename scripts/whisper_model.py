"""
The Whisper model DREAM listens with, and where it runs: Config/DREAM/WhisperModel
and Config/DREAM/WhisperDevice in config.json.

Pick both in options.py, so a GPU upgrade or downgrade is a settings change.
download_weights.bat downloads the chosen model, and every DREAM (dream.py,
app.py, dream_app.py, webserver.py, cpp_dream) loads it.

    python scripts/whisper_model.py              print the chosen model
    python scripts/whisper_model.py --download   download it to Whisper's cache
    python scripts/whisper_model.py --ggml       print whisper.cpp's file name for it

Bigger is more accurate and slower: tiny < base < small < medium < large-v3.
turbo (large-v3-turbo) is close to large-v3 at several times the speed.
The .en models are English only, and a little better at it below medium.

WhisperDevice: cpu keeps the GPU free for the LLM and lip-sync, cuda runs it on
the GPU, auto uses the GPU when there is one.
"""

import json
import os
import sys

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

MODELS = ["tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
          "large-v1", "large-v2", "large-v3", "large", "large-v3-turbo", "turbo"]
DEFAULT = "tiny"
DEVICES = ["cpu", "cuda", "auto"]
DEFAULT_DEVICE = "cpu"

# whisper.cpp names its files after the full model name
_GGML_ALIASES = {"large": "large-v3", "turbo": "large-v3-turbo"}


def _setting(key, allowed, default):
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            chosen = json.load(f)["Config"]["DREAM"].get(key, default)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return default
    if chosen not in allowed:
        print(f"config.json: {key} '{chosen}' isn't one of {', '.join(allowed)} - using '{default}'")
        return default
    return chosen


def name():
    return _setting("WhisperModel", MODELS, DEFAULT)


def device():
    """cpu or cuda. cuda (or auto) without a working GPU falls back to cpu."""
    chosen = _setting("WhisperDevice", DEVICES, DEFAULT_DEVICE)
    if chosen == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if chosen == "cuda":
        print("config.json: WhisperDevice is 'cuda' but no CUDA GPU was found - using the CPU")
    return "cpu"


def load():
    """The chosen model on the chosen device. Returns (model, description)."""
    import whisper
    n, d = name(), device()
    return whisper.load_model(n, device=d), f"Whisper {n} on {d.upper()}"


# The fleet's names as Whisper's hint (initial_prompt), so "tell KIDA zero one to stop" comes out
# as "KIDA01", what dream_fleet.py listens for. Tested 2026-10-14 with small.en on ten Piper-spoken
# fleet lines: this sentence 8/10 understood, a bare list of the names 5/10 (it turned "Whip" into
# "WIP"), no hint 5/10 ("key to 01").
NAMES = ("DREAM talks to the robots Nora, Mila, Whip, Ida, the arm, KIDA00 and KIDA01, "
         "and to Tina, Nina and Rift.")
NOISE = {"", "you", "you.", "thank you.", "thanks for watching!", "bye."}    # what Whisper makes of silence


def transcribe(model, path, cuda=False):
    """What was said in the audio file at path, with the fleet's names as the hint.
    "" for silence: never the hint echoed back, nor Whisper's usual silence phrases."""
    text = model.transcribe(path, language="en", fp16=cuda, initial_prompt=NAMES,
                            condition_on_previous_text=False)["text"].strip()
    if text.lower() in NOISE or text.rstrip(".") in NAMES or NAMES[:30] in text:
        return ""
    return text


def ggml_file():
    n = name()
    return f"ggml-{_GGML_ALIASES.get(n, n)}.bin"


def download():
    import whisper
    n = name()
    root = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
    path = os.path.join(root, os.path.basename(whisper._MODELS[n]))
    if os.path.isfile(path):
        print(f"  Whisper {n}: already there")
        return
    print(f"  Whisper {n}: downloading ...")
    whisper._download(whisper._MODELS[n], root, False)
    print(f"  Whisper {n}: OK")


if __name__ == "__main__":
    if "--download" in sys.argv:
        try:
            download()
        except ImportError:
            print("  Whisper isn't installed in this Python - run install_requirements.bat first")
            sys.exit(1)
        except Exception as e:
            print(f"  Whisper {name()}: FAILED ({e})")
            sys.exit(1)
    elif "--ggml" in sys.argv:
        print(ggml_file())
    else:
        print(name())
