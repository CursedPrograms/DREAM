#!/usr/bin/env python3

import os, sys, time, json, math, socket, tempfile, subprocess, wave, re
from collections import deque
import threading, queue, random, ipaddress
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests as _req
import psutil
from flask import Flask, Response, request, jsonify, render_template, send_file, send_from_directory, stream_with_context

# ── Zeroconf peer discovery ────────────────────────────────────────────────────
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    from scipy.signal import resample_poly
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

try:
    import serial as _serial
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False

# ── Platform ───────────────────────────────────────────────────────────────────
IS_WINDOWS = sys.platform == "win32"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(BASE_DIR, "audio");    os.makedirs(AUDIO_DIR, exist_ok=True)
VOICES_DIR  = os.path.join(BASE_DIR, "voices")
VIDEOS_DIR  = os.path.join(BASE_DIR, "static", "videos")
CERTS_DIR   = os.path.join(BASE_DIR, "certs")
CERT_PATH   = os.path.join(CERTS_DIR, "cert.pem")
KEY_PATH    = os.path.join(CERTS_DIR, "key.pem")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
AUDIO_FILE  = os.path.join(AUDIO_DIR, "stt.wav")

if IS_WINDOWS:
    VENV_DIR  = os.path.join(BASE_DIR, "venv311")
    PIPER_BIN = os.path.join(VENV_DIR, "Scripts", "piper.exe")
else:
    VENV_DIR  = os.path.join(BASE_DIR, "venv")
    PIPER_BIN = os.path.join(VENV_DIR, "bin", "piper")

# ── Import scan_wifi from scripts/ ────────────────────────────────────────────
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
try:
    import scan_wifi as _sw
    SCAN_WIFI_AVAILABLE = True
except ImportError:
    SCAN_WIFI_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 16
RMS_THRESHOLD  = 200

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}")

with open(CONFIG_PATH) as f:
    cfg = json.load(f)

dream_cfg     = cfg["Config"]["DREAM"]
comcentre_cfg = cfg["Config"]["ComCentre"]

CHAR_NAME     = dream_cfg["CharName"]
SYSTEM_PROMPT = dream_cfg["SystemPrompt"].format(name=CHAR_NAME)

# ── Zeroconf / ComCentre identity ──────────────────────────────────────────────
ZEROCONF_TYPE = comcentre_cfg.get("ZeroconfType", "_flask-link._tcp.local.")
THIS_NAME     = comcentre_cfg.get("ZeroconfName", "COMCENTRE")
THIS_PORT     = comcentre_cfg.get("Port", 5009)
# Plain-HTTP page on this PC only: localhost counts as a secure context, so the browser
# gives the mic to /dream.html without the self-signed-certificate warning.
LOCAL_HTTP_PORT = comcentre_cfg.get("LocalPort", 5010)

WIFI_TRIGGERS  = [
    "check wifi","wifi scan","scan wifi","who's on the wifi","who is on the wifi",
    "check network","network scan","what devices","list devices","show devices",
]
STATS_TRIGGERS = [
    "system stats","cpu usage","ram usage","memory usage","disk usage",
    "how's the system","system status","check stats","temperature","cpu temp",
    "how hot","system health",
]

# ── Alarm board (dream_sensors.ino) ────────────────────────────────────────────
ALARM_SERIAL_PORT = None   # None = find the board that answers "I am Dream" (see scripts/board_id.py)
ALARM_SERIAL_BAUD = 9600

# ── NORA (fleet robot) ──────────────────────────────────────────────────────────
# ComCentre is the fleet's voice/chat interface, not its network gateway —
# RIFT is the one that joins NORA's WiFi AP and shares internet to the fleet
# (see the NORA-Robot-v00 / RIFT repos). ComCentre just talks to NORA's web
# API over whatever route already gets there.
NORA_HOST           = "192.168.4.1"  # NORA's fixed WiFi AP address
NORA_PORT           = 5002           # NORA's robot web API (drive/UV/music/serial/message)
NORA_FLEET_PORT     = 5000           # NORA's fleet-registry port (distinct from her robot API above)
NORA_CHECK_INTERVAL = 30             # seconds between reachability checks

# ── RIFT (fleet registry) ─────────────────────────────────────────────────────
# ComCentre's own peer discovery above is zeroconf-only, so RIFT and NORA's
# HTTP-polling fleet registry (see RIFT/Fleet/register.py, NORA's
# scripts/esp32/esp32.ino) never see DREAM. Announce the same way NORA's
# fleet-authority heartbeat does, so DREAM shows up in RIFT's dashboard too.
# Configurable since RIFT typically runs on a separate machine from ComCentre.
RIFT_HOST           = comcentre_cfg.get("RiftHost", "localhost")
RIFT_PORT           = comcentre_cfg.get("RiftPort", 5000)
RIFT_HEARTBEAT_SECS = 10
FLEET_CAPABILITIES  = ["voice_chat", "tts", "stt", "llm"]

# Fleet heartbeat transport: "wifi" (default) registers with RIFT over HTTP,
# same as always. "bluetooth" registers directly with NORA instead, over a
# Bluetooth serial pairing (Fleet/bt_link.py) — RIFT itself has no
# Bluetooth radio to pair with, so NORA (the only Bluetooth-capable node in
# the fleet, see her esp32.ino 'H' command) is the Bluetooth fallback
# target rather than a like-for-like swap of RIFT's own address.
_fleet_mode_lock      = threading.Lock()
_fleet_transport_mode = "wifi"
_fleet_bt_port        = None
_bt_fleet_link        = None

def _fleet_heartbeat_wifi():
    try:
        _req.post(
            f"http://{RIFT_HOST}:{RIFT_PORT}/register",
            data={"name": THIS_NAME, "type": "ai_assistant",
                  "capabilities": ",".join(FLEET_CAPABILITIES)},
            timeout=2,
        )
    except _req.RequestException:
        pass

def _fleet_heartbeat_bluetooth(bt_port):
    global _bt_fleet_link
    try:
        if _bt_fleet_link is None:
            from Fleet.bt_link import BtFleetLink
            _bt_fleet_link = BtFleetLink(bt_port)
        _bt_fleet_link.register(THIS_NAME, FLEET_CAPABILITIES)
    except Exception:
        if _bt_fleet_link is not None:
            _bt_fleet_link.close()
        _bt_fleet_link = None

def rift_heartbeat():
    while _state["running"]:
        with _fleet_mode_lock:
            mode, bt_port = _fleet_transport_mode, _fleet_bt_port
        if mode == "bluetooth" and bt_port:
            _fleet_heartbeat_bluetooth(bt_port)
        else:
            _fleet_heartbeat_wifi()
        time.sleep(RIFT_HEARTBEAT_SECS)

def find_voice_model():
    if not os.path.isdir(VOICES_DIR): return None
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith(".onnx"): return os.path.join(VOICES_DIR, f)
    return None

VOICE_MODEL = find_voice_model()

# ── Piper sample-rate detection ────────────────────────────────────────────────
def get_piper_sample_rate():
    if VOICE_MODEL is None: return 22050
    jp = VOICE_MODEL + ".json"
    if os.path.exists(jp):
        try:
            with open(jp) as fh:
                c = json.load(fh)
            sr = c.get("audio", {}).get("sample_rate")
            if sr: return int(sr)
        except Exception:
            pass
    return 22050

PIPER_SR = get_piper_sample_rate()

# ── Shared state ───────────────────────────────────────────────────────────────
_state     = {"value": "idle", "running": True}
_history   = []
_sse_queue = queue.Queue()
_chat_lock = threading.Lock()

def set_state(s):
    _state["value"] = s
    _sse_queue.put({"type": "state", "state": s})

def push_event(evt: dict):
    _sse_queue.put(evt)

# ── Zeroconf peer discovery ────────────────────────────────────────────────────
_peer_nodes: dict[str, str] = {}
_peer_lock  = threading.Lock()

def _get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()

MY_IP = _get_ip()

# ── Self-signed TLS cert ────────────────────────────────────────────────────────
# Browsers (Chrome/Safari included) only allow getUserMedia() on localhost or a
# secure (HTTPS) origin — a plain http://<lan-ip> page silently fails to get mic
# access. This generates a cert good for MY_IP once per machine so /dream.html
# works from a phone on the LAN. The phone still needs to accept the one-time
# self-signed-certificate warning in its browser.
def ensure_self_signed_cert():
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return True
    os.makedirs(CERTS_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
             "-keyout", KEY_PATH, "-out", CERT_PATH, "-days", "825",
             "-subj", "/CN=comcentre.local",
             "-addext", f"subjectAltName=DNS:localhost,DNS:comcentre.local,IP:127.0.0.1,IP:{MY_IP}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30, check=True,
        )
        return True
    except FileNotFoundError:
        print("[ComCentre] openssl not found — can't generate a TLS cert; HTTPS disabled")
    except subprocess.CalledProcessError as e:
        print(f"[ComCentre] Cert generation failed: {e.stderr.decode(errors='ignore')}")
    return False

class _PeerListener:
    def remove_service(self, zc, type_, name):
        short = name.split(".")[0]
        with _peer_lock:
            _peer_nodes.pop(short, None)
        push_event({"type": "nodes", "nodes": _safe_peers()})

    def add_service(self, zc, type_, name):
        self.update_service(zc, type_, name)

    def update_service(self, zc, type_, name):
        info = zc.get_service_info(type_, name)
        if info:
            addrs = [socket.inet_ntoa(a) for a in info.addresses]
            if addrs:
                short = name.split(".")[0]
                if short != THIS_NAME:
                    url = f"http://{addrs[0]}:{info.port}"
                    with _peer_lock:
                        _peer_nodes[short] = url
                    push_event({"type": "nodes", "nodes": _safe_peers()})

def _safe_peers():
    with _peer_lock:
        return dict(_peer_nodes)

def _start_zeroconf():
    zc = Zeroconf()
    info = ServiceInfo(
        ZEROCONF_TYPE,
        f"{THIS_NAME}.{ZEROCONF_TYPE}",
        addresses=[socket.inet_aton(MY_IP)],
        port=THIS_PORT,
        properties={"version": "2.8"},
    )
    zc.register_service(info)
    ServiceBrowser(zc, ZEROCONF_TYPE, _PeerListener())
    return zc, info

_zc_instance, _zc_info = None, None

# ── Audio helpers ──────────────────────────────────────────────────────────────
def _record_clip(filepath, seconds):
    if not SD_AVAILABLE: return False
    try:
        native = int(sd.query_devices(kind="input")["default_samplerate"])
        audio  = sd.rec(int(seconds * native), samplerate=native, channels=CHANNELS, dtype="int16")
        sd.wait()
        flat = audio[:, 0] if audio.ndim > 1 else audio.flatten()
        if native != SAMPLE_RATE:
            g    = math.gcd(SAMPLE_RATE, native)
            flat = resample_poly(flat, SAMPLE_RATE // g, native // g).astype(np.int16)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(flat.tobytes())
        return os.path.getsize(filepath) > 500
    except Exception as e:
        push_event({"type": "error", "msg": str(e)})
        return False

def _convert_to_wav(src: str, dst: str) -> bool:
    """Convert any audio format (webm, ogg, mp4…) to 16-bit mono 16kHz WAV via ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", dst],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        return result.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 500
    except FileNotFoundError:
        push_event({"type": "error", "msg": (
            "ffmpeg not found — install with: "
            "winget install ffmpeg" if IS_WINDOWS else "sudo apt install ffmpeg"
        )})
        return False
    except Exception as e:
        push_event({"type": "error", "msg": f"ffmpeg error: {e}"})
        return False

def check_audio_levels(filepath=None):
    fp = filepath or AUDIO_FILE
    if not os.path.exists(fp): return False
    try:
        with wave.open(fp, "rb") as wf:
            audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return int(np.max(np.abs(audio))) > RMS_THRESHOLD
    except Exception:
        return False

# ── Whisper ────────────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("tiny", device="cpu")
    return _whisper_model

_whisper_lock = threading.Lock()

def transcribe_file(filepath):
    try:
        with _whisper_lock:
            return get_whisper().transcribe(filepath, language="en", fp16=False)["text"].strip()
    except Exception as e:
        push_event({"type": "error", "msg": str(e)})
        return ""

# ── Memories & Milestones ──────────────────────────────────────────────────────
# Shared with scripts/dream.py — see scripts/dream_memory.py.
from dream_memory import MEMORIES_PATH, MILESTONES_PATH, extract_memories, memory_prompt_block
from dream_memory import remember as _remember

def remember(kind: str, text: str):
    """Save a fact; tells the dashboard the first time each kind of fact is learned."""
    if _remember(kind, text) == "milestone":
        push_event({"type": "milestone", "text": text})

# ── LLM ───────────────────────────────────────────────────────────────────────
def ask_llm(prompt, history):
    set_state("thinking")
    ctx = ""
    for m in history[-6:]:
        ctx += ("You" if m["role"] == "assistant" else "Human") + f": {m['content']}\n"
    system = SYSTEM_PROMPT + memory_prompt_block()
    payload = {
        "model": MODEL,
        "prompt": f"System: {system}\n\n{ctx}Human: {prompt}\nYou:",
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 150, "num_gpu": 20},
    }
    try:
        r = _req.post(OLLAMA_URL, json=payload, timeout=120)
        resp = r.json().get("response", "").strip()
        if "<think>" in resp:
            end = resp.find("</think>")
            if end != -1: resp = resp[end + 8:].strip()
        if resp.lower().startswith("dream:"): resp = resp[6:].strip()
        return resp or "I didn't catch that."
    except _req.exceptions.Timeout:
        return "That took too long. Please try again."
    except Exception as e:
        return f"Error: {e}"

# ── TTS ────────────────────────────────────────────────────────────────────────
def speak_text(text):
    """Generate WAV via Piper. Returns temp file path or None."""
    if not text or not os.path.exists(PIPER_BIN) or not VOICE_MODEL:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=AUDIO_DIR)
    tmp.close()
    try:
        proc = subprocess.run(
            [PIPER_BIN, "-m", VOICE_MODEL, "-f", tmp.name],
            input=text.encode("utf-8"), capture_output=True, timeout=15,
        )
        if proc.returncode != 0 or os.path.getsize(tmp.name) < 100:
            os.unlink(tmp.name)
            return None
        return tmp.name
    except Exception:
        if os.path.exists(tmp.name): os.unlink(tmp.name)
        return None

# ── System stats ───────────────────────────────────────────────────────────────
_stats_cache = {"data": {}, "last": 0.0}

def get_system_stats():
    now = time.time()
    if now - _stats_cache["last"] < 2.0:
        return _stats_cache["data"]
    data = {}
    mem = psutil.virtual_memory()
    data.update(
        ram_used=mem.used / 1e9, ram_total=mem.total / 1e9, ram_pct=mem.percent,
        cpu_pct=psutil.cpu_percent(interval=None),
        cpu_cores=psutil.cpu_count(logical=False) or 1,
        cpu_threads=psutil.cpu_count(logical=True) or 1,
    )

    # sensors_temperatures() is Linux-only
    data["cpu_temp"] = None
    if not IS_WINDOWS:
        try:
            temps = psutil.sensors_temperatures()
            all_t = []
            for key in ("coretemp", "k10temp", "cpu_thermal", "acpitz"):
                if key in temps: all_t += [t.current for t in temps[key]]
            data["cpu_temp"] = max(all_t) if all_t else None
        except Exception:
            pass

    # Disk root differs per platform
    disk_path = "C:\\" if IS_WINDOWS else "/"
    disk = psutil.disk_usage(disk_path)
    data.update(disk_used=disk.used / 1e9, disk_total=disk.total / 1e9, disk_pct=disk.percent)

    _stats_cache.update(data=data, last=now)
    return data

def build_stats_summary():
    s = get_system_stats()
    parts = [
        f"CPU is at {s['cpu_pct']:.0f} percent",
        f"RAM {s['ram_used']:.1f} of {s['ram_total']:.1f} GB at {s['ram_pct']:.0f} percent",
        f"Disk {s['disk_used']:.0f} of {s['disk_total']:.0f} GB used",
    ]
    if s.get("cpu_temp"):
        parts.append(f"CPU temperature {s['cpu_temp']:.0f} degrees Celsius")
    return ". ".join(parts) + "."

# ── WiFi scan ──────────────────────────────────────────────────────────────────
def run_wifi_scan():
    if SCAN_WIFI_AVAILABLE:
        try:
            my_ip, subnet = _sw.get_local_ip_and_subnet()
            devices  = _sw.scan(subnet)
            enriched = _sw.enrich(devices, my_ip)
            return [
                {
                    "ip":       d["ip"],
                    "mac":      d["mac"],
                    "hostname": d.get("hostname", ""),
                    "vendor":   d.get("vendor", ""),
                    "type":     d.get("type", "Unknown"),
                    "me":       d.get("label", "") != "",
                }
                for d in enriched
            ]
        except Exception as e:
            push_event({"type": "error", "msg": f"scan_wifi error: {e}"})
            return []
    return _inline_wifi_scan()

def _ping_host(ip):
    """Ping a single host — flags differ between Windows and Linux."""
    if IS_WINDOWS:
        cmd = ["ping", "-n", "1", "-w", "1000", str(ip)]
    else:
        cmd = ["ping", "-c", "1", "-W", "1", str(ip)]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(ip) if r.returncode == 0 else None

def _read_arp_cache():
    """Read ARP cache — works on both Windows and Linux."""
    arp_cache = {}
    try:
        out = subprocess.check_output(["arp", "-a"], text=True, stderr=subprocess.DEVNULL)
        if IS_WINDOWS:
            # Windows: 192.168.1.1    aa-bb-cc-dd-ee-ff    dynamic
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    ip  = parts[0].strip()
                    mac = parts[1].strip().replace("-", ":").upper()
                    try:
                        ipaddress.ip_address(ip)
                        if mac not in ("FF:FF:FF:FF:FF:FF", ""):
                            arp_cache[ip] = mac
                    except ValueError:
                        pass
        else:
            # Linux ip neigh: 192.168.1.1 dev eth0 lladdr aa:bb:cc:dd:ee:ff REACHABLE
            for line in out.splitlines():
                p = line.split()
                if "lladdr" in p and "FAILED" not in line and "INCOMPLETE" not in line:
                    arp_cache[p[0]] = p[p.index("lladdr") + 1]
    except Exception:
        pass
    return arp_cache

def _inline_wifi_scan():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); my_ip = s.getsockname()[0]; s.close()
        subnet = my_ip.rsplit(".", 1)[0] + ".0/24"
        net    = ipaddress.ip_network(subnet, strict=False)

        with ThreadPoolExecutor(max_workers=80) as ex:
            list(ex.map(_ping_host, net.hosts()))

        arp_cache = _read_arp_cache()

        devices = []
        for ip in sorted(arp_cache, key=lambda x: list(map(int, x.split(".")))):
            mac = arp_cache[ip].replace("-", ":").upper()
            try:
                hn = socket.gethostbyaddr(ip)[0]
            except Exception:
                hn = ""
            devices.append({
                "ip": ip, "mac": mac, "hostname": hn,
                "vendor": "", "type": "Unknown device", "me": ip == my_ip,
            })
        return devices
    except Exception:
        return []

# ── Sensor board (serial to dream_sensors.ino) — this app is its only owner ────
# Only one program can hold a COM port, so app.py owns the connection and
# scripts/dream.py goes through it: every line the board prints is re-broadcast
# on /events as {"type": "sensor", "line": ...}, and commands are written back
# via POST /api/sensor/command.
_alarm_state       = {"enabled": True}  # mirrors the board's default (alarmEnabled = true on boot)
_sensor_serial     = None
_sensor_serial_lock = threading.Lock()   # guards writes and the _sensor_serial handle

# What the dashboard's SENSORS panel shows: live status plus a rolling event log.
_sensor_status = {"connected": False, "present": False, "range_m": None, "radar_ok": True, "last_motion": None}
_sensor_log    = deque(maxlen=50)

def _sensor_snapshot():
    return {**_sensor_status, "alarm_enabled": _alarm_state["enabled"]}

def _log_sensor(kind: str, text: str):
    entry = {"ts": time.strftime("%H:%M:%S"), "kind": kind, "text": text}
    _sensor_log.append(entry)
    push_event({"type": "sensor_log", **entry})

def _handle_sensor_line(line: str):
    """Updates status/log from one line the board printed, then tells the dashboard."""
    if line == "PRESENT":
        _sensor_status["present"] = True
        _log_sensor("presence", "Presence detected")
    elif line == "ABSENT":
        _sensor_status.update(present=False, range_m=None)
        _log_sensor("presence", "Presence lost")
    elif line.startswith("RANGE"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                _sensor_status["range_m"] = float(parts[1])
            except ValueError:
                pass
    elif line == "MOTION":
        _sensor_status["last_motion"] = time.strftime("%H:%M:%S")
        _log_sensor("motion", "Motion detected — alarm triggered")
    elif line.startswith("ALARM_STATE "):
        _alarm_state["enabled"] = line.endswith("ON")
        push_event({"type": "alarm", "enabled": _alarm_state["enabled"]})
        _log_sensor("alarm", "Alarm enabled" if _alarm_state["enabled"] else "Alarm disabled")
    elif line.startswith("RADAR_ERROR"):
        _sensor_status["radar_ok"] = False
        _log_sensor("error", "mmWave radar failed to start")
    _dream_on_sensor_line(line)
    push_event({"type": "sensor_status", **_sensor_snapshot()})

def _sensor_port():
    """Explicit ALARM_SERIAL_PORT if set, else whichever port answers "I am Dream"."""
    if ALARM_SERIAL_PORT:
        return ALARM_SERIAL_PORT
    from board_id import find_board
    port = find_board("dream", ALARM_SERIAL_BAUD)
    if not port:
        raise RuntimeError('no serial port answered "I am Dream"')
    return port

def sensor_watcher():
    """Keeps the board connected (reconnecting if it drops) and broadcasts every
    line it prints as a "sensor" event."""
    global _sensor_serial
    if not PYSERIAL_AVAILABLE:
        push_event({"type": "error", "msg": "pyserial not installed — sensor board disabled"})
        return

    warned = False
    while _state["running"]:
        try:
            ser = _serial.Serial(_sensor_port(), ALARM_SERIAL_BAUD, timeout=1)
        except Exception as e:
            if not warned:
                push_event({"type": "error", "msg": f"Sensor board unreachable: {e}"})
                warned = True
            time.sleep(5)
            continue
        warned = False
        time.sleep(2)  # board resets when the port opens; let it finish booting
        with _sensor_serial_lock:
            _sensor_serial = ser
        print(f"[ComCentre] Sensor board connected on {ser.port}")
        _sensor_status.update(connected=True, radar_ok=True)
        _log_sensor("system", f"Sensor board connected on {ser.port}")
        push_event({"type": "sensor_status", **_sensor_snapshot()})
        try:
            with ser:
                while _state["running"]:
                    line = ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    _handle_sensor_line(line)
                    push_event({"type": "sensor", "line": line})
        except Exception as e:
            push_event({"type": "error", "msg": f"Sensor board disconnected: {e}"})
            time.sleep(2)
        finally:
            with _sensor_serial_lock:
                _sensor_serial = None
            _sensor_status.update(connected=False, present=False, range_m=None)
            _log_sensor("system", "Sensor board disconnected")
            push_event({"type": "sensor_status", **_sensor_snapshot()})

def send_sensor_command(cmd: str) -> bool:
    """Writes one line to dream_sensors.ino. Returns True once it's sent."""
    with _sensor_serial_lock:
        ser = _sensor_serial
        if ser is None or not ser.is_open:
            push_event({"type": "error", "msg": "Sensor board not connected"})
            return False
        try:
            ser.write((cmd + "\n").encode())
            ser.flush()
            return True
        except Exception as e:
            push_event({"type": "error", "msg": f"Sensor command failed: {e}"})
            return False

def send_alarm_command(enabled: bool) -> bool:
    """Sends ALARM ON/OFF to dream_sensors.ino. Returns True once it's confirmed sent."""
    if not send_sensor_command("ALARM ON" if enabled else "ALARM OFF"):
        return False
    _alarm_state["enabled"] = enabled
    push_event({"type": "alarm", "enabled": enabled})
    return True

# ── NORA (fleet robot) ──────────────────────────────────────────────────────────
_nora_state = {"reachable": False, "last_check": 0.0, "error": None}
_nora_lock  = threading.Lock()

def _nora_url(path: str) -> str:
    return f"http://{NORA_HOST}:{NORA_PORT}{path}"

def _safe_nora():
    with _nora_lock:
        return dict(_nora_state)

def check_nora_reachable():
    """Polls NORA's /sensors so the dashboard knows whether she's on the network."""
    try:
        r = _req.get(_nora_url("/sensors"), timeout=3)
        reachable = r.status_code == 200
        with _nora_lock:
            _nora_state.update(reachable=reachable, error=None, last_check=time.time())
    except Exception as e:
        with _nora_lock:
            _nora_state.update(reachable=False, error=str(e), last_check=time.time())
    push_event({"type": "nora", **_safe_nora()})

def send_nora_serial(cmd: str) -> bool:
    """Forwards a raw command to NORA's onboard Arduino via her /serial passthrough."""
    try:
        r = _req.get(_nora_url("/serial"), params={"cmd": cmd}, timeout=3)
        return r.status_code == 200
    except Exception as e:
        push_event({"type": "error", "msg": f"NORA serial command failed: {e}"})
        return False

def send_nora_message(text: str) -> bool:
    """Posts a text message to NORA's message board (she has no display/speaker of her own)."""
    try:
        r = _req.post(_nora_url("/message"), data={"text": text}, timeout=3)
        return r.status_code == 200
    except Exception as e:
        push_event({"type": "error", "msg": f"NORA message failed: {e}"})
        return False

def nora_watcher():
    while _state["running"]:
        check_nora_reachable()
        time.sleep(NORA_CHECK_INTERVAL)

# ── Video pools (web avatar) ───────────────────────────────────────────────────
# Same clips scripts/dream.py plays (videos/), grouped by filename prefix, plus the
# sleeping loop and the startup intro. Falls back to static/videos if videos/ is missing.
ROOT_VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
VIDEO_POOL_PREFIXES = ("idle", "listening", "thinking", "talking", "flirtytalk")

def build_video_pools():
    pools = {s: [] for s in VIDEO_POOL_PREFIXES}
    pools["sleeping"] = []
    pools["intro"] = []
    if os.path.isdir(ROOT_VIDEOS_DIR):
        source, url_prefix = ROOT_VIDEOS_DIR, "/videos/"
    else:
        source, url_prefix = VIDEOS_DIR, "/static/videos/"
    if not os.path.isdir(source):
        return pools
    for f in sorted(os.listdir(source)):
        if not f.endswith(".mp4"):
            continue
        if f == "sleeping.mp4":
            pools["sleeping"].append(url_prefix + f)
        elif f == "intro1.mp4":
            pools["intro"].append(url_prefix + f)
        else:
            for prefix in VIDEO_POOL_PREFIXES:
                if f.startswith(prefix):
                    pools[prefix].append(url_prefix + f)
                    break
    return pools

# ── DREAM behaviours (web version of scripts/dream.py) ─────────────────────────
# dream.py keeps these in its own threads; here they live on the server so every
# open page stays in sync: idle timers (flirt, then sleep), waking on presence or
# the wake words, the farewell when someone leaves, and the wake-up greeting.
FLIRT_IDLE_TIMEOUT = 600   # 10 minutes — flirt attention grab
SLEEP_IDLE_TIMEOUT = 900   # 15 minutes — fall asleep
NEAR_DISTANCE_M    = 1.0   # wake-up greeting bands (meters)
FAR_DISTANCE_M     = 3.0
WAKE_SECONDS       = 3     # length of each wake-word listening clip
RECORD_SECONDS     = 16    # longest a spoken command can be
STARTUP_TEXT       = "ComCentre online. DREAM is ready. Say Hey DREAM to wake me."

WAKE_WORDS = ["hey dream", "hey, dream", "hi dream", "hi, dream", "okay dream", "ok dream", "dream"]
SLEEP_WAKE_WORDS = ["wake up", "wake up dream", "wake up, dream"]   # only heard while sleeping

ALARM_OFF_TRIGGERS = ["turn off the alarm", "disable the alarm", "disarm the alarm", "alarm off", "stop the alarm"]
ALARM_ON_TRIGGERS  = ["turn on the alarm", "enable the alarm", "arm the alarm", "alarm on"]
LIGHT_OFF_TRIGGERS = ["lights off", "turn off the lights", "turn the lights off", "lights out"]
LIGHT_RAINBOW_TRIGGERS = ["rainbow lights", "lights rainbow", "make the lights rainbow", "rainbow mode", "party lights"]
LIGHT_COLOR_NAMES = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "cyan", "white"]

_dream = {
    "sleeping":      False,
    "busy":          False,  # a page is recording, thinking or speaking
    "last_wake_ts":  time.time(),
    "flirt_played":  False,
    "presence_seen": False,  # mmWave PRESENT currently active — for the ABSENT farewell edge
    "distance_m":    None,
}

def touch_interaction():
    """Call whenever the user actually interacts — resets all idle timers."""
    _dream["last_wake_ts"] = time.time()
    _dream["flirt_played"] = False

def enter_sleep():
    if not _dream["sleeping"]:
        _dream["sleeping"] = True
        push_event({"type": "sleep", "sleeping": True})

def exit_sleep():
    touch_interaction()
    if _dream["sleeping"]:
        _dream["sleeping"] = False
        push_event({"type": "sleep", "sleeping": False})

def wake_greeting() -> str:
    """The "Yes?" line, chosen by the mmWave sensor's last RANGE reading."""
    d = _dream["distance_m"]
    if d is None:
        return "Yes?"
    if d < NEAR_DISTANCE_M:
        return "Whoa, hi! Yes?"
    if d > FAR_DISTANCE_M:
        return "Yes? I hear you over there."
    return "Yes?"

def _audio_url_for(text):
    wav = speak_text(text)
    return f"/audio/{os.path.basename(wav)}" if wav else None

def _announce(text: str, event_type: str = "speak"):
    """Speak a line on every open page ("speak" plays it; "wake" plays it, then listens)."""
    push_event({"type": event_type, "text": text, "audio_url": _audio_url_for(text)})

def _match_light_color(lower: str):
    """Only matches a color when it's clearly about the lights, so ordinary
    chat ("I like the color blue") doesn't accidentally trigger the RGB board."""
    if not re.search(r"\blight(s)?\b|\bmake it\b|\bturn it\b", lower):
        return None
    for name in LIGHT_COLOR_NAMES:
        if re.search(rf"\b{name}\b", lower):
            return name
    return None

def _sensor_voice_command(lower: str):
    """Alarm / RGB-light voice commands. Returns the spoken reply, or None if
    this isn't one."""
    if any(t in lower for t in ALARM_OFF_TRIGGERS):
        return "Alarm disabled." if send_alarm_command(False) else "I can't reach the alarm board."
    if any(t in lower for t in ALARM_ON_TRIGGERS):
        return "Alarm enabled." if send_alarm_command(True) else "I can't reach the alarm board."
    if any(t in lower for t in LIGHT_OFF_TRIGGERS):
        return "Lights off." if send_sensor_command("RGB OFF") else "I can't reach the lights."
    if any(t in lower for t in LIGHT_RAINBOW_TRIGGERS):
        return "Rainbow mode." if send_sensor_command("RGB RAINBOW") else "I can't reach the lights."
    color = _match_light_color(lower)
    if color:
        return f"Lights {color}." if send_sensor_command(f"RGB {color.upper()}") else "I can't reach the lights."
    return None

def _dream_on_sensor_line(line: str):
    """Presence keeps DREAM awake and wakes her when asleep; leaving says goodbye."""
    if line == "PRESENT":
        _dream["presence_seen"] = True
        if _dream["sleeping"]:
            exit_sleep()
            threading.Thread(target=_announce, args=(wake_greeting(), "wake"), daemon=True).start()
        else:
            touch_interaction()
    elif line == "ABSENT":
        if _dream["presence_seen"]:
            _dream["presence_seen"] = False
            if not _dream["sleeping"] and not _dream["busy"] and _state["value"] == "idle":
                threading.Thread(target=_announce, args=("Bye for now.",), daemon=True).start()
    elif line.startswith("RANGE"):
        parts = line.split()
        if len(parts) >= 2:
            try:
                _dream["distance_m"] = float(parts[1])
            except ValueError:
                return
            if not _dream["sleeping"]:
                touch_interaction()

def dream_watcher():
    """Idle timers: flirt clip at FLIRT_IDLE_TIMEOUT, sleep at SLEEP_IDLE_TIMEOUT."""
    while _state["running"]:
        time.sleep(5)
        if _dream["sleeping"]:
            continue
        if _state["value"] != "idle" or _dream["busy"]:
            touch_interaction()
            continue
        elapsed = time.time() - _dream["last_wake_ts"]
        if elapsed >= SLEEP_IDLE_TIMEOUT:
            enter_sleep()
        elif elapsed >= FLIRT_IDLE_TIMEOUT and not _dream["flirt_played"]:
            _dream["flirt_played"] = True
            pool = build_video_pools()["flirtytalk"]
            if pool:
                push_event({"type": "flirt", "clip": random.choice(pool)})

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── SSE broadcast ──────────────────────────────────────────────────────────────
_sse_clients: list[queue.Queue] = []
_sse_lock = threading.Lock()

def _broadcast(evt):
    data = "data: " + json.dumps(evt) + "\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try: q.put_nowait(data)
            except: dead.append(q)
        for q in dead: _sse_clients.remove(q)

def _sse_dispatcher():
    while True:
        evt = _sse_queue.get()
        _broadcast(evt)

threading.Thread(target=_sse_dispatcher, daemon=True).start()

@app.route("/events")
def events():
    client_q = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_clients.append(client_q)

    def gen():
        yield "data: " + json.dumps({"type": "state", "state": _state["value"]}) + "\n\n"
        yield "data: " + json.dumps({"type": "sleep", "sleeping": _dream["sleeping"]}) + "\n\n"
        yield "data: " + json.dumps({"type": "nodes", "nodes": _safe_peers()}) + "\n\n"
        try:
            while True:
                try:
                    data = client_q.get(timeout=25)
                except queue.Empty:
                    yield "data: " + json.dumps({"type": "ping"}) + "\n\n"
                    continue
                yield data
        except GeneratorExit:
            with _sse_lock:
                if client_q in _sse_clients: _sse_clients.remove(client_q)

    return Response(stream_with_context(gen()),
                    content_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", char_name=CHAR_NAME)

@app.route("/settings.html")
def settings_page():
    return render_template("settings.html", char_name=CHAR_NAME)

@app.route("/dream.html")
def dream_page():
    return render_template("dream.html", char_name=CHAR_NAME)

@app.route("/api/videos")
def api_videos():
    return jsonify(build_video_pools())

@app.route("/videos/<path:name>")
def serve_video(name):
    return send_from_directory(ROOT_VIDEOS_DIR, name)

@app.route("/api/dream/config")
def api_dream_config():
    return jsonify({
        "startup_text":   STARTUP_TEXT,
        "wake_seconds":   WAKE_SECONDS,
        "record_seconds": RECORD_SECONDS,
        "sleeping":       _dream["sleeping"],
    })

@app.route("/api/dream/activity", methods=["POST"])
def api_dream_activity():
    """A page reports whether it's busy (recording/thinking/speaking) so the
    idle timers and the farewell don't fire in the middle of a conversation."""
    busy = bool((request.get_json(silent=True) or {}).get("busy"))
    _dream["busy"] = busy
    if busy:
        touch_interaction()
    return jsonify({"busy": busy})

@app.route("/api/speak", methods=["POST"])
def api_speak():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "missing 'text'"}), 400
    return jsonify({"text": text, "audio_url": _audio_url_for(text)})

@app.route("/api/wake", methods=["POST"])
def api_wake():
    """One short listening clip from a page. Checks it for the wake words the
    way scripts/dream.py's listen_for_wake_word() does: while sleeping only
    "wake up" counts; while awake, "hey dream" (or a wifi-scan phrase) does."""
    f = request.files.get("audio")
    if not f:
        return jsonify({"trigger": None})
    fd, raw = tempfile.mkstemp(suffix=".raw", dir=AUDIO_DIR)
    os.close(fd)
    wav = raw + ".wav"
    try:
        f.save(raw)
        if not _convert_to_wav(raw, wav) or not check_audio_levels(wav):
            return jsonify({"trigger": None})
        text = transcribe_file(wav).lower().strip()
    finally:
        for p in (raw, wav):
            if os.path.exists(p):
                os.unlink(p)
    if not text:
        return jsonify({"trigger": None})

    if _dream["sleeping"]:
        if any(w in text for w in SLEEP_WAKE_WORDS):
            exit_sleep()
            greeting = wake_greeting()
            return jsonify({"trigger": "wake", "text": greeting, "audio_url": _audio_url_for(greeting), "heard": text})
        return jsonify({"trigger": None, "heard": text})

    if any(w in text for w in WAKE_WORDS):
        touch_interaction()
        greeting = wake_greeting()
        return jsonify({"trigger": "wake", "text": greeting, "audio_url": _audio_url_for(greeting), "heard": text})
    if any(t in text for t in WIFI_TRIGGERS):
        touch_interaction()
        return jsonify({"trigger": "wifi", "heard": text})
    return jsonify({"trigger": None, "heard": text})

@app.route("/ping")
def ping_route():
    return f"{THIS_NAME} alive"

@app.route("/api/status")
def api_status():
    ollama_ok = False
    try:
        r = _req.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({
        "state":       _state["value"],
        "sleeping":    _dream["sleeping"],
        "ollama":      ollama_ok,
        "piper":       os.path.exists(PIPER_BIN),
        "voice_model": VOICE_MODEL is not None,
        "sd":          SD_AVAILABLE,
        "platform":    "windows" if IS_WINDOWS else "linux",
        "mic_note":    "Microphone access requires HTTPS or localhost (browser security restriction)",
    })

@app.route("/api/nodes")
def api_nodes():
    return jsonify(_safe_peers())

@app.route("/api/chat", methods=["POST"])
def api_chat():
    acquired = _chat_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "busy", "state": _state["value"]}), 429
    try:
        return _handle_chat()
    finally:
        _chat_lock.release()

def _handle_chat():
    user_text  = ""
    voice_mode = False

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("audio")
        if f:
            raw_path  = AUDIO_FILE + ".raw"
            f.save(raw_path)
            converted = _convert_to_wav(raw_path, AUDIO_FILE)
            if converted and check_audio_levels():
                set_state("listening")
                push_event({"type": "transcript", "role": "system", "text": "Transcribing…"})
                user_text  = transcribe_file(AUDIO_FILE)
                voice_mode = True
            elif not converted:
                push_event({"type": "error", "msg": "Audio conversion failed — is ffmpeg installed?"})
    else:
        data       = request.get_json(silent=True) or {}
        user_text  = data.get("text", "").strip()
        voice_mode = bool(data.get("voice", False))

    if not user_text:
        set_state("idle")
        return jsonify({"error": "no input", "state": "idle"}), 400

    if voice_mode:
        push_event({"type": "transcript", "role": "user", "text": user_text})

    exit_sleep()  # any real interaction wakes her and resets the idle timers
    lower = user_text.lower()

    if any(w in lower for w in ["goodbye", "exit", "quit", "bye", "shut down", "shutdown"]):
        reply = "Goodbye."
        push_event({"type": "transcript", "role": "assistant", "text": reply})
        set_state("idle")
        resp = _make_reply(reply, voice_mode)
        enter_sleep()  # dream.py exits here; the web version goes to sleep (pages wait until she's done speaking)
        return resp

    board_reply = _sensor_voice_command(lower)
    if board_reply:
        push_event({"type": "transcript", "role": "assistant", "text": board_reply})
        set_state("idle")
        return _make_reply(board_reply, voice_mode)

    if any(t in lower for t in WIFI_TRIGGERS):
        set_state("thinking")
        push_event({"type": "transcript", "role": "system", "text": "Scanning network…"})
        devices = run_wifi_scan()
        if not devices:
            reply = "Network scan failed or no devices found."
        else:
            lines = [f"Found {len(devices)} device(s) on the network."]
            for d in devices:
                tag = " ← this machine" if d.get("me") else ""
                lines.append(f"{d['ip']}  {d['type']}{' — ' + d['hostname'] if d['hostname'] else ''}{tag}")
            reply = "\n".join(lines)
        push_event({"type": "transcript", "role": "assistant", "text": reply})
        push_event({"type": "wifi", "devices": devices})
        set_state("idle")
        return _make_reply(reply, voice_mode, extra={"devices": devices})

    if any(t in lower for t in STATS_TRIGGERS):
        stats = get_system_stats()
        reply = build_stats_summary()
        push_event({"type": "transcript", "role": "assistant", "text": reply})
        push_event({"type": "stats", "data": stats})
        set_state("idle")
        return _make_reply(reply, voice_mode, extra={"stats": stats})

    for kind, text in extract_memories(user_text):
        remember(kind, text)

    reply = ask_llm(user_text, _history)
    _history.append({"role": "user",      "content": user_text})
    _history.append({"role": "assistant", "content": reply})
    if len(_history) > 12: del _history[:-12]
    push_event({"type": "transcript", "role": "assistant", "text": reply})

    return _make_reply(reply, voice_mode)

def _make_reply(reply: str, voice_mode: bool, extra: dict | None = None):
    audio_url = None
    if voice_mode:
        wav_path = speak_text(reply)
        if wav_path:
            set_state("talking")
            audio_url = f"/audio/{os.path.basename(wav_path)}"
    set_state("idle")
    payload = {"reply": reply, "audio_url": audio_url, "state": "idle"}
    if extra:
        payload.update(extra)
    return jsonify(payload)

@app.route("/audio/<path:fname>")
def serve_audio(fname):
    safe = os.path.join(AUDIO_DIR, os.path.basename(fname))
    if not os.path.exists(safe):
        return "", 404
    return send_file(safe, mimetype="audio/wav")

@app.route("/api/memories")
def api_memories():
    lines = []
    if os.path.exists(MEMORIES_PATH):
        with open(MEMORIES_PATH, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    return jsonify({"memories": lines})

@app.route("/api/milestones")
def api_milestones():
    lines = []
    if os.path.exists(MILESTONES_PATH):
        with open(MILESTONES_PATH, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    return jsonify({"milestones": lines})

@app.route("/api/stats")
def api_stats():
    return jsonify(get_system_stats())

@app.route("/api/wifi")
def api_wifi():
    set_state("thinking")
    devices = run_wifi_scan()
    push_event({"type": "wifi", "devices": devices})
    set_state("idle")
    return jsonify({"devices": devices})

@app.route("/api/sensors")
def api_sensors():
    return jsonify({"status": _sensor_snapshot(), "log": list(_sensor_log)})

@app.route("/api/alarm")
def api_alarm_get():
    return jsonify(_alarm_state)

@app.route("/api/alarm", methods=["POST"])
def api_alarm_set():
    data    = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    if not send_alarm_command(enabled):
        return jsonify({"error": "alarm board unreachable", **_alarm_state}), 503
    return jsonify(_alarm_state)

# Commands scripts/dream.py (or anything else on this PC) may send to the board.
_SENSOR_COMMANDS = {"BUZZER", "ALARM ON", "ALARM OFF", "RGB OFF", "RGB RAINBOW"}

@app.route("/api/sensor/command", methods=["POST"])
def api_sensor_command():
    data = request.get_json(silent=True) or {}
    cmd  = " ".join(str(data.get("cmd", "")).split())  # one line, no stray newlines
    upper = cmd.upper()
    if upper not in _SENSOR_COMMANDS and not upper.startswith("RGB "):
        return jsonify({"error": "unknown sensor command"}), 400
    if upper in ("ALARM ON", "ALARM OFF"):
        ok = send_alarm_command(upper == "ALARM ON")
    else:
        ok = send_sensor_command(cmd)
    if not ok:
        return jsonify({"error": "sensor board unreachable"}), 503
    return jsonify({"status": "sent", "cmd": cmd})

@app.route("/api/nora")
def api_nora():
    return jsonify(_safe_nora())

@app.route("/api/nora/check", methods=["POST"])
def api_nora_check():
    threading.Thread(target=check_nora_reachable, daemon=True).start()
    return jsonify({"status": "checking"})

@app.route("/api/nora/serial", methods=["POST"])
def api_nora_serial():
    data = request.get_json(silent=True) or {}
    cmd  = data.get("cmd", "").strip()
    if not cmd:
        return jsonify({"error": "missing 'cmd'"}), 400
    if not send_nora_serial(cmd):
        return jsonify({"error": "NORA unreachable"}), 503
    return jsonify({"status": "sent", "cmd": cmd})

@app.route("/api/nora/message", methods=["POST"])
def api_nora_message():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "missing 'text'"}), 400
    if not send_nora_message(text):
        return jsonify({"error": "NORA unreachable"}), 503
    return jsonify({"status": "sent", "text": text})

@app.route("/api/nora/mode")
def api_nora_mode_get():
    with _fleet_mode_lock:
        return jsonify({"mode": _fleet_transport_mode, "bt_port": _fleet_bt_port})

@app.route("/api/nora/mode", methods=["POST"])
def api_nora_mode_set():
    global _fleet_transport_mode, _fleet_bt_port, _bt_fleet_link
    data = request.get_json(silent=True) or {}
    mode = (data.get("mode") or "").strip().lower()
    if mode not in ("wifi", "bluetooth"):
        return jsonify({"error": "mode must be 'wifi' or 'bluetooth'"}), 400

    bt_port = (data.get("bt_port") or "").strip() or None
    if mode == "bluetooth" and not bt_port:
        return jsonify({"error": "bt_port is required for Bluetooth mode"}), 400

    with _fleet_mode_lock:
        if _bt_fleet_link is not None:
            _bt_fleet_link.close()
            _bt_fleet_link = None
        _fleet_transport_mode = mode
        _fleet_bt_port = bt_port
        return jsonify({"mode": _fleet_transport_mode, "bt_port": _fleet_bt_port})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[ComCentre] Platform : {'Windows' if IS_WINDOWS else 'Linux'}")
    print(f"[ComCentre] My IP    : {MY_IP}")
    print(f"[ComCentre] Port     : {THIS_PORT}")
    print(f"[ComCentre] Piper    : {PIPER_BIN}")
    print(f"[ComCentre] Piper SR : {PIPER_SR} Hz")
    print(f"[ComCentre] scan_wifi: {'scripts/scan_wifi.py loaded' if SCAN_WIFI_AVAILABLE else 'NOT FOUND — using inline fallback'}")
    print(f"[ComCentre] DREAM static IP: {dream_cfg.get('StaticIP', 'not set')}")

    _zc_instance, _zc_info = _start_zeroconf()
    print(f"[ComCentre] Zeroconf registered as {THIS_NAME} on port {THIS_PORT}")

    threading.Thread(target=sensor_watcher, daemon=True).start()
    threading.Thread(target=dream_watcher, daemon=True).start()
    threading.Thread(target=nora_watcher, daemon=True).start()
    threading.Thread(target=rift_heartbeat, daemon=True).start()
    print(f"[ComCentre] Announcing to RIFT at {RIFT_HOST}:{RIFT_PORT} every {RIFT_HEARTBEAT_SECS}s")

    https_ok = ensure_self_signed_cert()
    run_kwargs = {"host": "0.0.0.0", "port": THIS_PORT, "debug": False, "threaded": True}
    if https_ok:
        run_kwargs["ssl_context"] = (CERT_PATH, KEY_PATH)
        print(f"[ComCentre] HTTPS enabled — open https://{MY_IP}:{THIS_PORT}/dream.html on your phone")
        print("[ComCentre] (accept the one-time self-signed certificate warning)")
        threading.Thread(
            target=lambda: app.run(host="127.0.0.1", port=LOCAL_HTTP_PORT, debug=False, threaded=True, use_reloader=False),
            daemon=True,
        ).start()
        print(f"[ComCentre] On this PC, no warning: http://localhost:{LOCAL_HTTP_PORT}/dream.html")
    else:
        print(f"[ComCentre] HTTPS unavailable — mic access on /dream.html will only work from localhost")

    try:
        app.run(**run_kwargs)
    finally:
        if _zc_instance and _zc_info:
            _zc_instance.unregister_service(_zc_info)
            _zc_instance.close()
            print("[ComCentre] Zeroconf unregistered.")