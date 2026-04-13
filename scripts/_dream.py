#!/usr/bin/env python3

import os
import sys
import time
import tempfile
import subprocess
import requests
import wave
import numpy as np
from rich.console import Console
from rich.panel import Panel
import threading
import json
import math
from rich.markup import escape

import pygame

console = Console()

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
IMG_DIR    = os.path.join(BASE_DIR, "images")
AUDIO_FILE = os.path.join(AUDIO_DIR, "stt.wav")
VENV_DIR   = os.path.join(BASE_DIR, "venv")
PIPER_BIN  = os.path.join(VENV_DIR, "bin", "piper")

os.makedirs(AUDIO_DIR, exist_ok=True)

# ==================== CONFIG ====================
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 16
RMS_THRESHOLD  = 50

with open(os.path.join(BASE_DIR, "config.json")) as f:
    config = json.load(f)

# Access CharName and SystemPrompt
CHAR_NAME = config["Config"]["DREAM"]["CharName"]
SYSTEM_PROMPT = config["Config"]["DREAM"]["SystemPrompt"].format(name=CHAR_NAME)

print("CharName:", CHAR_NAME)
print("SystemPrompt:", SYSTEM_PROMPT)

# ==================== VOICE MODEL ====================

def find_voice_model():
    if not os.path.isdir(VOICES_DIR):
        return None
    for fname in sorted(os.listdir(VOICES_DIR)):
        if fname.endswith(".onnx"):
            return os.path.join(VOICES_DIR, fname)
    return None

VOICE_MODEL = find_voice_model()

# ==================== SHARED STATE ====================
# Voice thread writes state; pygame main thread reads it.
# States: "idle" | "listening" | "thinking" | "talking"

_state = {"value": "idle", "running": True}

def set_state(s):
    _state["value"] = s

# ==================== BANNER ====================

def startup_banner():
    piper_ok = os.path.exists(PIPER_BIN)
    voice_ok = VOICE_MODEL is not None

    console.print(Panel.fit(
        "[bold cyan]ComCentre v2.3[/bold cyan]\n"
        "[dim]DREAM - Local AI Voice Assistant[/dim]\n\n"
        f"[green]LLM:[/green]      {MODEL}\n"
        f"[green]STT:[/green]      Whisper tiny\n"
        f"[green]Piper:[/green]    {'[green]' + PIPER_BIN + '[/green]' if piper_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Voice:[/green]    {'[green]' + os.path.basename(VOICE_MODEL) + '[/green]' if voice_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Images:[/green]   {IMG_DIR}",
        border_style="cyan"
    ))

    if not piper_ok:
        console.print(f"[red]Piper not found at {PIPER_BIN}[/red]")
        sys.exit(1)
    if not voice_ok:
        console.print(f"[red]No .onnx voice in {VOICES_DIR}[/red]")
        sys.exit(1)

    console.print(f"[green]OK[/green] Piper: {PIPER_BIN}")
    console.print(f"[green]OK[/green] Voice: {VOICE_MODEL}")

# ==================== RECORD ====================

def get_mic_samplerate():
    import sounddevice as sd
    dev  = sd.query_devices(kind="input")
    rate = int(dev["default_samplerate"])
    console.print(f"[dim]Mic native rate: {rate} Hz[/dim]")
    return rate

def record_audio():
    import sounddevice as sd
    from scipy.signal import resample_poly

    console.print("\n[bold yellow]Speak now...[/bold yellow]")
    set_state("listening")

    try:
        native_rate = get_mic_samplerate()
        audio = sd.rec(
            int(RECORD_SECONDS * native_rate),
            samplerate=native_rate,
            channels=CHANNELS,
            dtype="int16",
        )
        for i in range(RECORD_SECONDS):
            console.print(f"[dim]Recording... {i+1}/{RECORD_SECONDS}s[/dim]", end="\r")
            time.sleep(1)
        sd.wait()
        console.print("[dim]Recording complete.     [/dim]")

        audio_flat = audio[:, 0] if audio.ndim > 1 else audio.flatten()

        if native_rate != SAMPLE_RATE:
            console.print(f"[dim]Resampling {native_rate} -> {SAMPLE_RATE} Hz[/dim]")
            g          = math.gcd(SAMPLE_RATE, native_rate)
            audio_flat = resample_poly(
                audio_flat, SAMPLE_RATE // g, native_rate // g
            ).astype(np.int16)

        with wave.open(AUDIO_FILE, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_flat.tobytes())

        size = os.path.getsize(AUDIO_FILE)
        if size > 1000:
            console.print(f"[dim]Saved ({size} bytes)[/dim]")
            return True

        console.print("[red]Recording too small[/red]")
        return False

    except Exception as e:
        console.print(f"[red]Recording failed: {e}[/red]")
        return False
    finally:
        set_state("idle")

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
        console.print(f"[dim]Audio max: {max_amp}  rms: {rms:.1f}[/dim]")
        return max_amp > RMS_THRESHOLD
    except Exception as e:
        console.print(f"[red]Audio check error: {e}[/red]")
        return False

# ==================== WHISPER ====================

_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        console.print("[dim]Loading Whisper...[/dim]")
        import whisper
        _whisper_model = whisper.load_model("tiny", device="cpu")
        console.print("[green]OK[/green] Whisper ready")
    return _whisper_model

def transcribe():
    if not os.path.exists(AUDIO_FILE):
        return None
    try:
        model  = get_whisper()
        result = model.transcribe(AUDIO_FILE, language="en", fp16=False)
        text   = result["text"].strip()
        return text if text else None
    except Exception as e:
        console.print(f"[red]STT error: {e}[/red]")
        return None

# ==================== LLM ====================

def ask_llm(prompt, history):
    set_state("thinking")
    with console.status("[dim]Thinking...[/dim]", spinner="dots"):
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
                "num_gpu": 20,
            }
        }

        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            if r.status_code != 200:
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
            return f"Error: {e}"
        finally:
            set_state("idle")

# ==================== SPEAK ====================

def speak(text):
    if not text:
        return

    console.print(f"\n[bold cyan]DREAM:[/bold cyan] {text}\n")
    set_state("talking")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AUDIO_DIR)
    tmp.close()

    try:
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

        pygame.mixer.music.load(tmp.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.unload()

    except Exception as e:
        console.print(f"[red]TTS failed: {e}[/red]")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        set_state("idle")

# ==================== VOICE LOOP (background thread) ====================

def voice_loop():
    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"[green]OK[/green] Ollama: {', '.join(models) or 'no models'}")
        if not any(MODEL in m for m in models):
            console.print(f"[yellow]'{MODEL}' not pulled. Run: ollama pull {MODEL}[/yellow]")
    except Exception:
        console.print("[red]Ollama not running. Start with: ollama serve[/red]")
        _state["running"] = False
        return

    get_whisper()
    speak("ComCentre online. DREAM is ready.")

    history = []

    while _state["running"]:
        try:
            if not record_audio():
                continue

            if not check_audio_levels():
                console.print("[dim]No sound detected -- speak louder[/dim]")
                continue

            user_text = transcribe()
            if not user_text:
                console.print("[dim]Could not transcribe -- try again[/dim]")
                continue

            console.print(f"\n[bold green]You:[/bold green] {user_text}")

            if any(w in user_text.lower() for w in ["goodbye", "exit", "quit", "bye", "shut down"]):
                speak("Goodbye.")
                _state["running"] = False
                break

            response = ask_llm(user_text, history)
            history.append({"role": "user",     "content": user_text})
            history.append({"role": "assistant", "content": response})
            if len(history) > 12:
                history = history[-12:]

            speak(response)

        except Exception as e:
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            set_state("idle")
            continue

# ==================== PYGAME MAIN THREAD ====================

def _load_cover(filename, sw, sh):
    full = os.path.join(IMG_DIR, filename)
    if not os.path.exists(full):
        surf = pygame.Surface((sw, sh))
        surf.fill((10, 12, 20))
        font = pygame.font.SysFont("monospace", 28)
        lbl  = font.render(f"[missing: {filename}]", True, (80, 80, 100))
        surf.blit(lbl, lbl.get_rect(center=(sw // 2, sh // 2)))
        return surf
    raw    = pygame.image.load(full).convert()
    iw, ih = raw.get_size()
    scale  = max(sw / iw, sh / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    scaled = pygame.transform.smoothscale(raw, (nw, nh))
    xo     = (nw - sw) // 2
    yo     = (nh - sh) // 2
    return scaled.subsurface((xo, yo, sw, sh)).copy()

def run_display():
    pygame.init()
    pygame.mixer.init()

    info   = pygame.display.Info()
    SW, SH = info.current_w, info.current_h

    screen = pygame.display.set_mode((SW, SH), pygame.FULLSCREEN | pygame.NOFRAME)
    pygame.display.set_caption("DREAM")
    pygame.mouse.set_visible(False)

    images = {
        "idle":      [_load_cover("dream.jpg",      SW, SH)],
        "listening": [_load_cover("listening0.jpeg",  SW, SH)],
        "thinking":  [_load_cover("thinking0.jpeg",   SW, SH),
                      _load_cover("thinking1.jpeg",   SW, SH)],
        "talking":   [_load_cover("talking0.jpeg",    SW, SH),
                      _load_cover("talking1.jpeg",    SW, SH),
                      _load_cover("talking2.jpeg",    SW, SH)],
    }

    intervals = {
        "idle":      2000,
        "listening": 2000,
        "thinking":   500,
        "talking":    280,
    }

    clock      = pygame.time.Clock()
    frame_idx  = 0
    last_state = None
    last_swap  = pygame.time.get_ticks()

    while _state["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _state["running"] = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _state["running"] = False

        state = _state["value"]
        now   = pygame.time.get_ticks()

        if state != last_state:
            frame_idx  = 0
            last_swap  = now
            last_state = state

        frames   = images.get(state, images["idle"])
        interval = intervals.get(state, 1000)

        if len(frames) > 1 and now - last_swap >= interval:
            frame_idx = (frame_idx + 1) % len(frames)
            last_swap = now

        screen.blit(frames[frame_idx], (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# ==================== MAIN ====================

def main():
    startup_banner()

    # Voice loop runs in background thread
    vt = threading.Thread(target=voice_loop, daemon=True)
    vt.start()

    # Pygame display runs on the main thread
    run_display()

    vt.join(timeout=2)

if __name__ == "__main__":
    main()
