#!/usr/bin/env python3

import os, sys, time, json, math, socket, tempfile, subprocess, wave
import threading, queue, random, ipaddress
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests as _req
import psutil
from flask import Flask, Response, request, jsonify, render_template, send_file, stream_with_context

# ── Zeroconf peer discovery ────────────────────────────────────────────────────
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser

# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    from scipy.signal import resample_poly
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

# ── Platform ───────────────────────────────────────────────────────────────────
IS_WINDOWS = sys.platform == "win32"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR   = os.path.join(BASE_DIR, "audio");    os.makedirs(AUDIO_DIR, exist_ok=True)
VOICES_DIR  = os.path.join(BASE_DIR, "voices")
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

WIFI_TRIGGERS  = [
    "check wifi","wifi scan","scan wifi","who's on the wifi","who is on the wifi",
    "check network","network scan","what devices","list devices","show devices",
]
STATS_TRIGGERS = [
    "system stats","cpu usage","ram usage","memory usage","disk usage",
    "how's the system","system status","check stats","temperature","cpu temp",
    "how hot","system health",
]

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

def transcribe_file(filepath):
    try:
        return get_whisper().transcribe(filepath, language="en", fp16=False)["text"].strip()
    except Exception as e:
        push_event({"type": "error", "msg": str(e)})
        return ""

# ── LLM ───────────────────────────────────────────────────────────────────────
def ask_llm(prompt, history):
    set_state("thinking")
    ctx = ""
    for m in history[-6:]:
        ctx += ("You" if m["role"] == "assistant" else "Human") + f": {m['content']}\n"
    payload = {
        "model": MODEL,
        "prompt": f"System: {SYSTEM_PROMPT}\n\n{ctx}Human: {prompt}\nYou:",
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

    lower = user_text.lower()

    if any(w in lower for w in ["goodbye", "exit", "quit", "bye", "shut down", "shutdown"]):
        reply = "Goodbye."
        push_event({"type": "transcript", "role": "assistant", "text": reply})
        set_state("idle")
        return _make_reply(reply, voice_mode)

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

    try:
        app.run(host="0.0.0.0", port=THIS_PORT, debug=False, threaded=True)
    finally:
        if _zc_instance and _zc_info:
            _zc_instance.unregister_service(_zc_info)
            _zc_instance.close()
            print("[ComCentre] Zeroconf unregistered.")