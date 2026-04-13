#!/usr/bin/env python3

# okay so after 15 minutes she should fall asleep, so the wake word would be wake up if she is asleep, she runs deep_dream_batch when sleeping. the video shown is also sleeping.mp4. than for lypsyncing if she talk use musetalk to lipsync after the piper part of the loop use that audio and display that video, also check if cuda is available and use that so the flow now is. 

# Mic → Whisper → Ollama → Piper TTS → MuseTalk → Speaker

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
import socket
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.markup import escape
import random
import surveillance

import pygame
import psutil

try:
    import cv2
    CV2_AVAILABLE = True
except Exception as _cv2_err:
    CV2_AVAILABLE = False
    _cv2_import_error = str(_cv2_err)
else:
    _cv2_import_error = ""

try:
    from scapy.all import ARP, Ether, srp, conf as scapy_conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

console = Console()

# ==================== PLATFORM ====================
IS_WINDOWS = sys.platform == "win32"

# ==================== PATHS ====================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR  = os.path.join(BASE_DIR, "audio")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
IMG_DIR    = os.path.join(BASE_DIR, "images")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
AUDIO_FILE = os.path.join(AUDIO_DIR, "stt.wav")
WAKE_FILE  = os.path.join(AUDIO_DIR, "wake.wav")

if IS_WINDOWS:
    VENV_DIR  = os.path.join(BASE_DIR, "venv311")
    PIPER_BIN = os.path.join(VENV_DIR, "Scripts", "piper.exe")
else:
    VENV_DIR  = os.path.join(BASE_DIR, "venv")
    PIPER_BIN = os.path.join(VENV_DIR, "bin", "piper")

os.makedirs(AUDIO_DIR, exist_ok=True)

# at the top with your other imports
import surveillance

# after startup_banner() or wherever you want it to start
surveillance.start_surveillance()
surveillance.start_motion_detection()

# ==================== CONFIG ====================
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 16
WAKE_SECONDS   = 3
RMS_THRESHOLD  = 200

# How long (seconds) of no wake-word before the flirty attention grab fires
FLIRT_IDLE_TIMEOUT = 600  # 10 minutes

WAKE_WORDS = [
    "hey dream", "hey, dream", "hi dream", "hi, dream",
    "okay dream", "ok dream", "dream",
]

WIFI_TRIGGERS = [
    "check wifi", "check the wifi", "wifi scan", "scan wifi",
    "scan the wifi", "who's on the wifi", "who is on the wifi",
    "check network", "network scan", "check connections",
    "what devices", "list devices", "show devices",
]

STATS_TRIGGERS = [
    "system stats", "cpu usage", "ram usage", "memory usage",
    "disk usage", "how's the system", "system status",
    "how are you doing", "check stats", "check the stats",
    "temperature", "cpu temp", "how hot", "system health",
]

with open(os.path.join(BASE_DIR, "config.json")) as f:
    config = json.load(f)

# Access CharName and SystemPrompt
CHAR_NAME     = config["Config"]["DREAM"]["CharName"]
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

# ==================== PIPER SAMPLE RATE DETECTION ====================

def get_piper_sample_rate():
    """
    Read the sample_rate from the .onnx.json config file that sits next to
    the voice model. Falls back to 22050 if the file is missing or malformed.
    """
    if VOICE_MODEL is None:
        return 22050
    json_path = VOICE_MODEL + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path) as fh:
                cfg = json.load(fh)
            sr = cfg.get("audio", {}).get("sample_rate")
            if sr:
                console.print(f"[dim]Piper voice sample rate: {sr} Hz[/dim]")
                return int(sr)
        except Exception as e:
            console.print(f"[yellow]Could not read piper json config: {e}[/yellow]")
    console.print("[yellow]Piper json config not found — assuming 22050 Hz[/yellow]")
    return 22050

# ==================== SHARED STATE ====================
_state = {
    "value":        "idle",
    "running":      True,
    "wake_active":  False,
    "last_wake_ts": time.time(),
    "flirt_played": False,
    "force_video":  None,
}

def set_state(s):
    _state["value"] = s

def touch_interaction():
    """Call whenever the user actually interacts — resets the flirt timer."""
    _state["last_wake_ts"] = time.time()
    _state["flirt_played"] = False

# ==================== VIDEO POOL HELPERS ====================

def _glob_videos(prefix):
    """Return sorted list of video paths matching videos/<prefix>*.mp4"""
    if not os.path.isdir(VIDEOS_DIR):
        return []
    return sorted(
        os.path.join(VIDEOS_DIR, f)
        for f in os.listdir(VIDEOS_DIR)
        if f.startswith(prefix) and f.endswith(".mp4")
    )

VIDEO_POOLS = {}

def build_video_pools():
    global VIDEO_POOLS
    VIDEO_POOLS = {
        "idle":       _glob_videos("idle"),
        "listening":  _glob_videos("listening"),
        "thinking":   _glob_videos("thinking"),
        "talking":    _glob_videos("talking"),
        "flirtytalk": _glob_videos("flirtytalk"),
    }
    for k, v in VIDEO_POOLS.items():
        console.print(f"[dim]  {k}: {len(v)} video(s)[/dim]")

# ==================== BANNER ====================

def startup_banner():
    piper_ok = os.path.exists(PIPER_BIN)
    voice_ok = VOICE_MODEL is not None
    console.print(Panel.fit(
        "[bold cyan]ComCentre v2.6[/bold cyan]\n"
        "[dim]DREAM - Local AI Voice Assistant[/dim]\n\n"
        f"[green]LLM:[/green]      {MODEL}\n"
        f"[green]STT:[/green]      Whisper tiny\n"
        f"[green]Wake:[/green]     'Hey DREAM'\n"
        f"[green]Piper:[/green]    {'[green]' + PIPER_BIN + '[/green]' if piper_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Voice:[/green]    {'[green]' + os.path.basename(VOICE_MODEL) + '[/green]' if voice_ok else '[red]NOT FOUND[/red]'}\n"
        f"[green]Videos:[/green]   {VIDEOS_DIR}\n"
        f"[green]Platform:[/green] {'Windows' if IS_WINDOWS else 'Linux'}",
        border_style="cyan"
    ))
    if not piper_ok:
        console.print(f"[red]❌  Piper not found at: {PIPER_BIN}[/red]")
        if IS_WINDOWS:
            console.print("[yellow]    Run: pip install piper-tts  (inside your venv)[/yellow]")
        else:
            console.print("[yellow]    Run: pip install piper-tts[/yellow]")
        sys.exit(1)
    if not voice_ok:
        console.print(f"[red]❌  No .onnx voice model found in: {VOICES_DIR}[/red]")
        console.print("[yellow]    Download a voice from https://rhasspy.github.io/piper-samples/[/yellow]")
        sys.exit(1)
    console.print(f"[green]OK[/green] Piper ready")
    console.print(f"[green]OK[/green] Voice: {os.path.basename(VOICE_MODEL)}")
    if CV2_AVAILABLE:
        import cv2 as _cv2
        console.print(f"[green]OK[/green] OpenCV {_cv2.__version__}")
    else:
        console.print(f"[yellow]WARN[/yellow] opencv-python not importable: {_cv2_import_error}")

    build_video_pools()

# ==================== AUDIO ====================

def get_mic_samplerate():
    import sounddevice as sd
    return int(sd.query_devices(kind="input")["default_samplerate"])

def _record_clip(filepath, seconds):
    import sounddevice as sd
    from scipy.signal import resample_poly
    try:
        native_rate = get_mic_samplerate()
        audio = sd.rec(int(seconds * native_rate), samplerate=native_rate,
                       channels=CHANNELS, dtype="int16")
        sd.wait()
        audio_flat = audio[:, 0] if audio.ndim > 1 else audio.flatten()
        if native_rate != SAMPLE_RATE:
            g = math.gcd(SAMPLE_RATE, native_rate)
            audio_flat = resample_poly(audio_flat, SAMPLE_RATE // g,
                                       native_rate // g).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_flat.tobytes())
        return os.path.getsize(filepath) > 500
    except Exception as e:
        console.print(f"[red]Record error: {e}[/red]")
        return False

def check_audio_levels(filepath=None):
    fp = filepath or AUDIO_FILE
    if not os.path.exists(fp):
        return False
    try:
        with wave.open(fp, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return int(np.max(np.abs(audio))) > RMS_THRESHOLD
    except Exception:
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

def transcribe_file(filepath):
    try:
        result = get_whisper().transcribe(filepath, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        console.print(f"[red]STT error: {e}[/red]")
        return ""

# ==================== WAKE WORD LOOP ====================

def listen_for_wake_word():
    console.print("[dim]Listening for wake word 'Hey DREAM'...[/dim]")
    _state["wake_active"] = True
    set_state("idle")

    while _state["running"]:
        if not _record_clip(WAKE_FILE, WAKE_SECONDS):
            time.sleep(0.2)
            continue

        if not check_audio_levels(WAKE_FILE):
            continue

        text = transcribe_file(WAKE_FILE).lower().strip()
        if not text:
            continue

        console.print(f"[dim]Wake check: '{text}'[/dim]")

        if any(w in text for w in WAKE_WORDS):
            console.print("[bold green]Wake word detected![/bold green]")
            _state["wake_active"] = False
            touch_interaction()
            return "__WAKE__"

        if any(t in text for t in WIFI_TRIGGERS):
            _state["wake_active"] = False
            touch_interaction()
            return "__WIFI_SCAN__"

    _state["wake_active"] = False
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
            "options": {"temperature": 0.7, "num_predict": 150, "num_gpu": 20},
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

# ==================== SPEAK ====================

def speak(text):
    if not text:
        return
    console.print(f"\n[bold cyan]DREAM:[/bold cyan] {text}\n")
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AUDIO_DIR)
    tmp.close()
    try:
        proc = subprocess.run(
            [PIPER_BIN, "-m", VOICE_MODEL, "-f", tmp.name],
            input=text.encode("utf-8"), capture_output=True, timeout=15,
        )
        if proc.returncode != 0:
            console.print(f"[red]Piper error: {proc.stderr.decode()}[/red]")
            return
        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) < 100:
            return
        set_state("talking")
        sound   = pygame.mixer.Sound(tmp.name)
        channel = sound.play()
        while channel and channel.get_busy():
            time.sleep(0.05)
    except Exception as e:
        console.print(f"[red]TTS failed: {e}[/red]")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        set_state("idle")

# ==================== SYSTEM STATS ====================

_stats_cache = {"data": {}, "last": 0.0}

def get_system_stats():
    now = time.time()
    if now - _stats_cache["last"] < 2.0:
        return _stats_cache["data"]
    data = {}
    mem = psutil.virtual_memory()
    data["ram_used"]    = mem.used  / 1073741824
    data["ram_total"]   = mem.total / 1073741824
    data["ram_pct"]     = mem.percent
    data["cpu_pct"]     = psutil.cpu_percent(interval=None)
    data["cpu_cores"]   = psutil.cpu_count(logical=False) or 1
    data["cpu_threads"] = psutil.cpu_count(logical=True)  or 1

    # sensors_temperatures() is Linux-only
    data["cpu_temp"] = None
    if not IS_WINDOWS:
        try:
            temps = psutil.sensors_temperatures()
            all_t = []
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                if key in temps:
                    all_t += [t.current for t in temps[key]]
            data["cpu_temp"] = max(all_t) if all_t else None
        except Exception:
            pass

    # Disk root differs per platform
    disk_path = "C:\\" if IS_WINDOWS else "/"
    disk = psutil.disk_usage(disk_path)
    data["disk_used"]  = disk.used  / 1073741824
    data["disk_total"] = disk.total / 1073741824
    data["disk_pct"]   = disk.percent

    _stats_cache["data"] = data
    _stats_cache["last"] = now
    return data

def build_stats_summary():
    s = get_system_stats()
    parts = [
        f"CPU is at {s['cpu_pct']:.0f} percent",
        f"RAM usage is {s['ram_used']:.1f} of {s['ram_total']:.1f} gigabytes, that's {s['ram_pct']:.0f} percent",
        f"disk is {s['disk_used']:.0f} of {s['disk_total']:.0f} gigabytes used",
    ]
    if s.get("cpu_temp"):
        parts.append(f"CPU temperature is {s['cpu_temp']:.0f} degrees Celsius")
    return ". ".join(parts) + "."

# ==================== WIFI SCANNER ====================

def _ping_host(ip):
    """Ping a host — flags differ between Windows and Linux."""
    if IS_WINDOWS:
        cmd = ["ping", "-n", "1", "-w", "1000", str(ip)]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", str(ip)]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(ip) if r.returncode == 0 else None

def _get_hostname(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return ""

def _get_vendor(mac):
    try:
        r = requests.get(f"https://api.macvendors.com/{mac}", timeout=3)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    return "Unknown"

def _is_randomized(mac):
    try:
        return bool(int(mac.split(":")[0], 16) & 0x02)
    except Exception:
        return False

def _guess_type(mac, hostname, vendor):
    if _is_randomized(mac):
        return "phone or tablet with randomized MAC"
    h = (hostname + vendor).lower()
    if any(x in h for x in ["iphone", "apple", "ipad"]):                          return "Apple device"
    if any(x in h for x in ["samsung", "android", "xiaomi", "huawei"]):           return "Android device"
    if any(x in h for x in ["router", "gateway", "dlink", "tp-link", "asus", "netgear"]): return "router"
    if any(x in h for x in ["windows", "intel", "realtek"]):                      return "Windows PC"
    if any(x in h for x in ["ubuntu", "linux", "debian", "raspi"]):               return "Linux device"
    if any(x in h for x in ["tv", "cast", "roku", "echo", "alexa"]):              return "smart TV or IoT device"
    if vendor and vendor != "Unknown":
        return vendor
    return "unknown device"

def _read_arp_cache():
    """Read ARP cache — works on both Windows and Linux."""
    arp_cache = {}
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, stderr=subprocess.DEVNULL)
        if IS_WINDOWS:
            # Windows arp -a output:
            # Interface: 192.168.1.5 --- 0x4
            #   Internet Address      Physical Address      Type
            #   192.168.1.1           aa-bb-cc-dd-ee-ff     dynamic
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    ip  = parts[0].strip()
                    mac = parts[1].strip().replace("-", ":").upper()
                    try:
                        ipaddress.ip_address(ip)   # validate it's actually an IP
                        if mac not in ("FF:FF:FF:FF:FF:FF", ""):
                            arp_cache[ip] = mac
                    except ValueError:
                        pass
        else:
            # Linux ip neigh output:
            # 192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE
            for line in out.splitlines():
                p = line.split()
                if "lladdr" in p and "FAILED" not in line and "INCOMPLETE" not in line:
                    arp_cache[p[0]] = p[p.index("lladdr") + 1]
    except Exception as e:
        console.print(f"[yellow]ARP cache read warning: {e}[/yellow]")
    return arp_cache

def run_wifi_scan():
    set_state("thinking")
    speak("Scanning the network. Give me a moment.")

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        my_ip  = s.getsockname()[0]
        s.close()
        subnet = my_ip.rsplit(".", 1)[0] + ".0/24"

        console.print(f"[dim]Scanning {subnet}...[/dim]")

        network = ipaddress.ip_network(subnet, strict=False)
        with ThreadPoolExecutor(max_workers=80) as ex:
            list(ex.map(_ping_host, network.hosts()))

        arp_direct = {}
        if SCAPY_AVAILABLE:
            scapy_conf.verb = 0
            pkt = Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=subnet)
            answered, _ = srp(pkt, timeout=3, verbose=False, retry=1)
            arp_direct = {r.psrc: r.hwsrc for _, r in answered}

        arp_cache = _read_arp_cache()

        all_ips = set(arp_direct) | set(arp_cache)
        devices = []
        for ip in sorted(all_ips, key=lambda x: list(map(int, x.split(".")))):
            raw_mac  = (arp_direct.get(ip) or arp_cache.get(ip) or "??:??:??:??:??:??")
            mac      = raw_mac.replace("-", ":").upper()
            hostname = _get_hostname(ip)
            vendor   = _get_vendor(mac) if not _is_randomized(mac) else "N/A"
            dtype    = _guess_type(mac, hostname, vendor)
            devices.append({
                "ip": ip, "mac": mac, "hostname": hostname or "",
                "vendor": vendor, "type": dtype, "me": ip == my_ip,
            })

        count  = len(devices)
        phones = sum(1 for d in devices if "phone" in d["type"] or "tablet" in d["type"])
        others = count - phones

        lines = [f"I found {count} device{'s' if count != 1 else ''} on the network."]
        for d in devices:
            if d["me"]:
                lines.append(f"{d['ip']} is this machine.")
            else:
                lines.append(f"{d['ip']}: {d['type']}{', hostname ' + d['hostname'] if d['hostname'] else ''}.")

        summary = " ".join(lines)
        if len(summary) > 600:
            summary = (f"I found {count} devices on your network. "
                       f"{phones} appear to be phones or tablets. "
                       f"The rest include {others} other devices.")

        return summary

    except Exception as e:
        console.print(f"[red]WiFi scan error: {e}[/red]")
        return "Sorry, the network scan failed."
    finally:
        set_state("idle")

# ==================== FLIRT WATCHER (background thread) ====================

def flirt_watcher():
    while _state["running"]:
        time.sleep(5)
        if not _state["running"]:
            break
        if _state["flirt_played"]:
            continue
        if _state["value"] != "idle":
            touch_interaction()
            continue
        elapsed = time.time() - _state["last_wake_ts"]
        if elapsed >= FLIRT_IDLE_TIMEOUT:
            pool = VIDEO_POOLS.get("flirtytalk", [])
            if pool:
                chosen = random.choice(pool)
                console.print(f"[magenta]Flirt timeout — queueing {os.path.basename(chosen)}[/magenta]")
                _state["force_video"] = chosen
                _state["flirt_played"] = True
            else:
                console.print("[yellow]Flirt timeout but no flirtytalk videos found.[/yellow]")
                _state["flirt_played"] = True

# ==================== VIDEO PLAYER ====================

class VideoPlayer:
    def __init__(self, path, sw, sh, loop=True):
        self.sw, self.sh = sw, sh
        self.loop     = loop
        self.finished = False
        self.cap      = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.fps      = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.ms_frame = 1000.0 / self.fps
        self._surface = pygame.Surface((sw, sh))
        self._last_ms = 0.0
        self._read_next()

    def _read_next(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            else:
                self.finished = True
                return
        if ok:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw = rgb.shape[:2]
            scale  = max(self.sw / iw, self.sh / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            rgb    = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            xo, yo = (nw - self.sw) // 2, (nh - self.sh) // 2
            rgb    = rgb[yo:yo + self.sh, xo:xo + self.sw]
            self._surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

    def get_frame(self, now_ms):
        if not self.finished and now_ms - self._last_ms >= self.ms_frame:
            self._read_next()
            self._last_ms = now_ms
        return self._surface

    def release(self):
        self.cap.release()

# ==================== VIDEO STATE MANAGER ====================

class VideoStateManager:
    def __init__(self, sw, sh):
        self.sw = sw
        self.sh = sh
        self._player: VideoPlayer | None       = None
        self._current_path   = None
        self._current_state  = None
        self._prev_idle_path = None
        self._force_player: VideoPlayer | None = None

    def _pick_random(self, pool, avoid=None):
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        choices = [p for p in pool if p != avoid]
        return random.choice(choices) if choices else random.choice(pool)

    def _load(self, path, loop=True):
        if self._player:
            self._player.release()
        try:
            self._player       = VideoPlayer(path, self.sw, self.sh, loop=loop)
            self._current_path = path
        except Exception as e:
            console.print(f"[red]VideoPlayer load error: {e}[/red]")
            self._player       = None
            self._current_path = None

    def get_frame(self, now_ms, state):
        force_path = _state.get("force_video")
        if force_path and self._force_player is None:
            try:
                self._force_player = VideoPlayer(force_path, self.sw, self.sh, loop=False)
                console.print(f"[magenta]Playing force video: {os.path.basename(force_path)}[/magenta]")
            except Exception as e:
                console.print(f"[red]Force video error: {e}[/red]")
                _state["force_video"] = None

        if self._force_player is not None:
            frame = self._force_player.get_frame(now_ms)
            if self._force_player.finished:
                self._force_player.release()
                self._force_player = None
                _state["force_video"] = None
                console.print("[magenta]Force video done.[/magenta]")
            return frame

        pool = VIDEO_POOLS.get(state, [])
        if not pool:
            pool = VIDEO_POOLS.get("idle", [])

        if state != self._current_state:
            avoid = self._prev_idle_path if state == "idle" else None
            path  = self._pick_random(pool, avoid=avoid)
            if state == "idle":
                self._prev_idle_path = path
            self._current_state = state
            if path:
                self._load(path, loop=True)

        if state == "idle" and self._player and self._player.finished:
            path = self._pick_random(pool, avoid=self._current_path)
            self._prev_idle_path = path
            if path:
                self._load(path, loop=True)

        if self._player is None:
            blank = pygame.Surface((self.sw, self.sh))
            blank.fill((10, 12, 20))
            return blank

        return self._player.get_frame(now_ms)

    def release(self):
        if self._player:
            self._player.release()
        if self._force_player:
            self._force_player.release()

# ==================== MAIN VOICE LOOP ====================

def voice_loop():
    try:
        r      = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        console.print(f"[green]OK[/green] Ollama: {', '.join(models) or 'no models'}")
        if not any(MODEL in m for m in models):
            console.print(f"[yellow]'{MODEL}' not pulled.[/yellow]")
    except Exception:
        console.print("[red]Ollama not running.[/red]")
        _state["running"] = False
        return

    get_whisper()
    speak("ComCentre online. DREAM is ready. Say Hey DREAM to wake me.")

    history = []

    while _state["running"]:
        try:
            trigger = listen_for_wake_word()

            if trigger is None or not _state["running"]:
                break

            if trigger == "__WIFI_SCAN__":
                summary = run_wifi_scan()
                speak(summary)
                continue

            speak("Yes?")
            set_state("listening")
            console.print("[bold yellow]Listening for command...[/bold yellow]")

            if not _record_clip(AUDIO_FILE, RECORD_SECONDS):
                speak("I didn't catch that.")
                continue

            if not check_audio_levels():
                speak("I couldn't hear you clearly.")
                continue

            user_text = transcribe_file(AUDIO_FILE)
            if not user_text:
                speak("I didn't catch that.")
                continue

            console.print(f"\n[bold green]You:[/bold green] {user_text}")
            set_state("idle")

            lower = user_text.lower()

            if any(w in lower for w in ["goodbye", "exit", "quit", "bye", "shut down", "shutdown"]):
                speak("Goodbye.")
                _state["running"] = False
                break

            if any(t in lower for t in WIFI_TRIGGERS):
                summary = run_wifi_scan()
                speak(summary)
                continue

            if any(t in lower for t in STATS_TRIGGERS):
                speak(build_stats_summary())
                continue

            response = ask_llm(user_text, history)
            history.append({"role": "user",      "content": user_text})
            history.append({"role": "assistant",  "content": response})
            if len(history) > 12:
                history = history[-12:]
            speak(response)

        except Exception as e:
            console.print(f"[red]Error: {escape(str(e))}[/red]")
            set_state("idle")
            continue

# ==================== DISPLAY (main thread) ====================

def run_display():
    info   = pygame.display.Info()
    SW, SH = info.current_w, info.current_h
    screen = pygame.display.set_mode((SW, SH), pygame.FULLSCREEN | pygame.NOFRAME)
    pygame.display.set_caption("DREAM")
    pygame.mouse.set_visible(False)

    if not CV2_AVAILABLE:
        console.print("[red]opencv-python required for video playback — exiting.[/red]")
        _state["running"] = False
        return

    vsm   = VideoStateManager(SW, SH)
    clock = pygame.time.Clock()

    while _state["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                _state["running"] = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    _state["running"] = False

        state  = _state["value"]
        now_ms = pygame.time.get_ticks()

        frame = vsm.get_frame(now_ms, state)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    vsm.release()
    pygame.quit()

# ==================== MAIN ====================

def main():
    startup_banner()

    piper_sr = get_piper_sample_rate()
    console.print(f"[dim]Mixer init: {piper_sr} Hz, 16-bit mono, buffer=4096[/dim]")
    pygame.mixer.pre_init(frequency=piper_sr, size=-16, channels=1, buffer=4096)
    pygame.init()
    pygame.mixer.init()
    console.print(
        f"[green]OK[/green] Mixer: "
        f"{pygame.mixer.get_init()[0]} Hz / "
        f"{pygame.mixer.get_init()[1]}-bit / "
        f"{pygame.mixer.get_init()[2]} ch"
    )

    ft = threading.Thread(target=flirt_watcher, daemon=True)
    ft.start()

    vt = threading.Thread(target=voice_loop, daemon=True)
    vt.start()
    run_display()
    vt.join(timeout=2)

if __name__ == "__main__":
    main()