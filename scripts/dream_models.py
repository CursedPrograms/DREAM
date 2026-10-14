"""
The models DREAM uses (talking, looking, speaking), from config.json:

    Config/DREAM/LlmModel          the Ollama model she talks with
    Config/DREAM/Voice             the Piper voice she speaks with (voices/<Voice>.onnx)
    Config/DREAM/VisionModel       the Ollama model she looks with ("what do you see?")
    Config/DREAM/Memory/LlmModel   the Ollama model the memory server thinks with

Pick them in options.py; download_weights.bat downloads them. (Whisper is in
whisper_model.py.)

    python scripts/dream_models.py --llm              print the chat model
    python scripts/dream_models.py --memory-llm       print the memory model, if memory is on
    python scripts/dream_models.py --vision           print the vision model
    python scripts/dream_models.py --voice            print the voice
    python scripts/dream_models.py --download-voice   download the voice into voices/

Thinking models (qwen3:4b, deepseek-r1, gpt-oss) aren't offered for LlmModel: she
speaks the reply as it comes, and they either spend it thinking (an empty reply)
or think out loud. The default is Qwen3 4B's instruct build instead, which never
thinks out loud (TODO.md: "Smarter brain").
"""

import json
import os
import re
import sys
import urllib.request

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
VOICES_DIR  = os.path.join(ROOT_DIR, "voices")

DEFAULT_LLM    = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_MEMORY = "qwen3:4b"
DEFAULT_VISION = "gemma3:4b-it-qat"
DEFAULT_VOICE  = "en_US-amy-medium"
VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


def _dream_cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)["Config"]["DREAM"]
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def llm():
    return str(_dream_cfg().get("LlmModel") or DEFAULT_LLM)


def memory_llm():
    mem = _dream_cfg().get("Memory") or {}
    return str(mem.get("LlmModel") or DEFAULT_MEMORY)


def vision_model():
    return str(_dream_cfg().get("VisionModel") or DEFAULT_VISION)


def memory_enabled():
    return bool((_dream_cfg().get("Memory") or {}).get("Hindsight"))


def voice():
    return str(_dream_cfg().get("Voice") or DEFAULT_VOICE)


def voice_path():
    """voices/<Voice>.onnx, or the first voice in voices/ if that one isn't downloaded."""
    chosen = os.path.join(VOICES_DIR, voice() + ".onnx")
    if os.path.isfile(chosen):
        return chosen
    others = sorted(f for f in os.listdir(VOICES_DIR) if f.endswith(".onnx")) if os.path.isdir(VOICES_DIR) else []
    if others:
        print(f"config.json: voice '{voice()}' isn't in voices/ (run download_weights.bat) - using {others[0]}")
        return os.path.join(VOICES_DIR, others[0])
    return None


def voice_url(name):
    """en_US-amy-medium -> its .onnx on Hugging Face (piper-voices: <family>/<lang>/<speaker>/<quality>/)."""
    m = re.fullmatch(r"(([a-z]{2,3})_[A-Za-z]{2,3})-([^-]+)-(x_low|low|medium|high)", name)
    if not m:
        raise ValueError(f"'{name}' doesn't look like a Piper voice (like en_US-amy-medium)")
    lang, family, speaker, quality = m.group(1), m.group(2), m.group(3), m.group(4)
    return f"{VOICE_BASE_URL}/{family}/{lang}/{speaker}/{quality}/{name}.onnx"


def download_voice():
    name = voice()
    url = voice_url(name)
    os.makedirs(VOICES_DIR, exist_ok=True)
    for src, dest in ((url, name + ".onnx"), (url + ".json", name + ".onnx.json")):
        path = os.path.join(VOICES_DIR, dest)
        if os.path.isfile(path):
            print(f"  {dest}: already there")
            continue
        print(f"  {dest}: downloading ...")
        urllib.request.urlretrieve(src, path + ".part")
        os.replace(path + ".part", path)
        print(f"  {dest}: OK")


if __name__ == "__main__":
    if "--llm" in sys.argv:
        print(llm())
    elif "--memory-llm" in sys.argv:
        if memory_enabled():
            print(memory_llm())
    elif "--vision" in sys.argv:
        print(vision_model())
    elif "--voice" in sys.argv:
        print(voice())
    elif "--download-voice" in sys.argv:
        try:
            download_voice()
        except Exception as e:
            print(f"  Voice {voice()}: FAILED ({e})")
            sys.exit(1)
    else:
        print(f"LLM: {llm()}  Vision: {vision_model()}  Voice: {voice()}  Memory LLM: {memory_llm() if memory_enabled() else 'off'}")
