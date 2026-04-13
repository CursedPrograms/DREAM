#!/usr/bin/env python3

import os
import sys
import time
import tempfile
import subprocess
import requests
import wave
import json
import numpy as np
from rich.console import Console
from rich.panel import Panel

console = Console()

# ==================== FORCE CUDA ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ==================== PATHS ====================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
AUDIO_FILE = os.path.join(AUDIO_DIR, "stt.wav")
VENV_DIR   = os.path.join(BASE_DIR, "venv")
PIPER_BIN  = os.path.join(VENV_DIR, "bin", "piper")

os.makedirs(AUDIO_DIR, exist_ok=True)

# ==================== AUDIO DEVICES ====================
MIC_DEVICE     = "plughw:2,0"   # USB PCM2902  (card 2 after reboot)
SPEAKER_DEVICE = "plughw:1,0"   # ALC897 analog (card 1 after reboot)

# ==================== CONFIG ====================
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 5
RMS_THRESHOLD  = 150
WHISPER_MODEL  = "base"   # base >> tiny for accuracy

# GPU layer offload — GTX 1050 Ti has 4GB VRAM
# 20 layers = safe for 4GB, tune up after checking nvidia-smi
GPU_LAYERS = 20

# Load system prompt from config.json
with open(os.path.join(BASE_DIR, "config.json")) as f:
    cfg = json.load(f)
SYSTEM_PROMPT = cfg["Config"]["DREAM"]["SystemPrompt"]

# ==================== GPU CHECK ====================

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(f"[green]✓[/green] CUDA — {name} ({vram:.1f}GB VRAM)")
            return True
        else:
            console.print("[yellow]⚠[/yellow]  CUDA not available — running on CPU")
            return False
    except ImportError:
        console.print("[yellow]⚠[/yellow]  PyTorch not installed — CPU only")
        return False

# ==================== VOICE MODEL ====================

def find_voice_model():
    if not os.path.isdir(VOICES_DIR):
        return None
    for fname in sorted(os.listdir(VOICES_DIR)):
        if fname.endswith(".onnx"):
            return os.path.join(VOICES_DIR, fname)
    return None

VOICE_MODEL = find_voice_model()

# ==================== BANNER ====================

def startup_banner(cuda_ok):
    piper_ok = os.path.exists(PIPER_BIN)
    voice_ok = VOICE_MODEL is not None
    gpu_line = "[green]CUDA active — GTX 1050 Ti[/green]" if cuda_ok else "[yellow]CPU only[/yellow]"

    console.print(Panel.fit(
        "[bold cyan]ComCentre v2.5[/bold cyan]\n"
        "[dim]DREAM — Local AI Voice Assistant[/dim]\n\n"
        f"[green]LLM:[/green]      {MODEL} ({GPU_LAYERS} layers on GPU)\n"
        f"[green]STT:[/green]      Whisper {WHISPER_MODEL} ({'CUDA' if cuda_ok else 'CPU'})\n"
        f"[green]Piper:[/green]    {'[green]✓[/green]' if piper_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Voice:[/green]    {'[green]' + os.path.basename(VOICE_MODEL) + '[/green]' if voice_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Mic:[/green]      {MIC_DEVICE}\n"
        f"[green]Speaker:[/green]  {SPEAKER_DEVICE}\n"
        f"[green]GPU:[/green]      {gpu_line}",
        border_style="cyan"
    ))

    if not piper_ok:
        console.print(f"[red]✗ Piper not found at {PIPER_BIN}[/red]")
        sys.exit(1)
    if not voice_ok:
        console.print(f"[red]✗ No .onnx voice model in {VOICES_DIR}[/red]")
        sys.exit(1)

# ==================== RECORD ====================

def record_audio():
    console.print("\n[bold yellow]🎙  Speak now...[/bold yellow]")

    for i in range(RECORD_SECONDS):
        console.print(f"[dim]Recording... {i+1}/{RECORD_SECONDS} seconds[/dim]", end="\r")
        time.sleep(1)
    console.print("[dim]Recording... complete!     [/dim]")

    cmd = [
        "arecord", "-q",
        "-D", MIC_DEVICE,
        "-d", str(RECORD_SECONDS),
        "-r", str(SAMPLE_RATE),
        "-f", "S16_LE",
        "-c", str(CHANNELS),
        AUDIO_FILE,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        console.print(f"[red]Recording failed: {result.stderr.decode()}[/red]")
        return False

    size = os.path.getsize(AUDIO_FILE) if os.path.exists(AUDIO_FILE) else 0
    if size > 1000:
        console.print(f"[dim]✓ Saved ({size} bytes)[/dim]")
        return True

    console.print("[red]Recording too small[/red]")
    return False

# ==================== AUDIO LEVEL ====================

def check_audio_levels():
    if not os.path.exists(AUDIO_FILE):
        return False
    try:
        with wave.open(AUDIO_FILE, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio  = np.frombuffer(frames, dtype=np.int16)
        max_amp = int(np.max(np.abs(audio)))
        rms     = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        console.print(f"[dim]Audio — Max: {max_amp}, RMS: {rms:.2f}[/dim]")
        return max_amp > RMS_THRESHOLD
    except Exception as e:
        console.print(f"[red]Audio check error: {e}[/red]")
        return False

# ==================== WHISPER (GPU) ====================

_whisper_model = None

def get_whisper(cuda_ok):
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if cuda_ok else "cpu"
        console.print(f"[dim]Loading Whisper {WHISPER_MODEL} on {device.upper()}...[/dim]")
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL, device=device)
        console.print(f"[green]✓[/green] Whisper {WHISPER_MODEL} on {device.upper()}")
    return _whisper_model

def transcribe(cuda_ok):
    if not os.path.exists(AUDIO_FILE):
        return None
    try:
        model  = get_whisper(cuda_ok)
        result = model.transcribe(
            AUDIO_FILE,
            language="en",
            fp16=cuda_ok,                    # fp16 only on GPU
            condition_on_previous_text=False, # stops hallucination loops
            no_speech_threshold=0.4,
            logprob_threshold=-1.0,
        )
        text = result["text"].strip()
        console.print(f"[dim]Whisper: '{text}'[/dim]")

        # Filter junk hallucinations
        junk = ["thanks for watching", "thank you", "subscribe", "[blank_audio]",
                "[music]", "[silence]", "you", ".", ","]
        if not text or text.lower().strip(".,!? ") in junk:
            return None

        return text

    except Exception as e:
        console.print(f"[red]STT error: {e}[/red]")
        return None

# ==================== LLM (GPU layers) ====================

def ask_llm(prompt, history):
    with console.status("[dim]🧠 Thinking...[/dim]", spinner="dots"):
        context = ""
        for msg in history[-6:]:
            role = "You" if msg["role"] == "assistant" else "Human"
            context += f"{role}: {msg['content']}\n"

        full_prompt = f"System: {SYSTEM_PROMPT}\n\n{context}Human: {prompt}\nYou:"

        payload = {
            "model": MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150,
                "num_gpu": GPU_LAYERS,  # offload layers to GTX 1050 Ti
                "num_thread": 8,
                "low_vram": True,
            }
        }

        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if r.status_code != 200:
                console.print(f"[red]LLM Error {r.status_code}: {r.text[:200]}[/red]")
                return "Sorry, I encountered an error."

            response = r.json().get("response", "").strip()

            if "<think>" in response:
                end = response.find("</think>")
                if end != -1:
                    response = response[end + 8:].strip()

            if response.lower().startswith("dream:"):
                response = response[6:].strip()

            return response if response else "I didn't catch that."

        except requests.exceptions.Timeout:
            return "That took too long. Please try again."
        except Exception as e:
            console.print(f"[red]LLM error: {e}[/red]")
            return f"Error: {e}"

# ==================== SPEAK (GPU piper) ====================

def speak(text):
    if not text:
        return

    console.print(f"\n[bold cyan]DREAM:[/bold cyan] {text}\n")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AUDIO_DIR)
    tmp.close()

    try:
        # --cuda flag uses GPU for piper inference
        proc = subprocess.run(
            [PIPER_BIN, "-m", VOICE_MODEL, "-f", tmp.name, "--cuda"],
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=15,
        )

        # fallback to CPU if --cuda fails
        if proc.returncode != 0:
            proc = subprocess.run(
                [PIPER_BIN, "-m", VOICE_MODEL, "-f", tmp.name],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=15,
            )

        if proc.returncode != 0:
            console.print(f"[red]Piper error: {proc.stderr.decode()}[/red]")
            return

        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 100:
            console.print("[red]Piper produced no audio[/red]")
            return

        play = subprocess.run(
            ["aplay", "-q", "-D", SPEAKER_DEVICE, tmp.name],
            capture_output=True,
            timeout=30,
        )
        if play.returncode != 0:
            console.print(f"[red]aplay error: {play.stderr.decode()}[/red]")

    except Exception as e:
        console.print(f"[red]TTS failed: {e}[/red]")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

# ==================== MAIN ====================

def main():
    cuda_ok = check_gpu()
    startup_banner(cuda_ok)

    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"[green]✓[/green] Ollama — {', '.join(models) or 'no models'}")
        if not any(MODEL in m for m in models):
            console.print(f"[yellow]⚠ '{MODEL}' not pulled. Run: ollama pull {MODEL}[/yellow]")
    except Exception:
        console.print("[red]✗ Ollama not running — start with: ollama serve[/red]")
        sys.exit(1)

    get_whisper(cuda_ok)
    speak("ComCentre online. DREAM is ready.")

    history = []

    while True:
        try:
            if not record_audio():
                continue

            if not check_audio_levels():
                console.print("[dim]No sound — speak louder[/dim]")
                continue

            user_text = transcribe(cuda_ok)
            if not user_text:
                console.print("[dim]Could not transcribe — try again[/dim]")
                continue

            console.print(f"\n[bold green]You:[/bold green] {user_text}")

            if any(w in user_text.lower() for w in ["goodbye", "exit", "quit", "bye", "shut down"]):
                speak("Goodbye.")
                break

            response = ask_llm(user_text, history)
            history.append({"role": "user",      "content": user_text})
            history.append({"role": "assistant",  "content": response})
            if len(history) > 12:
                history = history[-12:]

            speak(response)

        except KeyboardInterrupt:
            speak("Goodbye.")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            continue

if __name__ == "__main__":
    main()
