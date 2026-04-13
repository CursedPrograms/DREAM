#!/usr/bin/env python3
"""
ComCentre v2.6 — DREAM Flask Web Interface
Replaces the pygame/OpenCV display + threading model with a Flask server.
The browser becomes the "display thread"; SSE pushes state updates live.
"""

import os, sys, time, json, math, socket, tempfile, subprocess, wave, threading, queue, random, ipaddress
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np
import requests as _req
import psutil
from flask import Flask, Response, request, jsonify, render_template, stream_with_context

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import sounddevice as sd
    from scipy.signal import resample_poly
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

try:
    from scapy.all import ARP, Ether, srp, conf as scapy_conf
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR  = os.path.join(BASE_DIR, "audio");   os.makedirs(AUDIO_DIR, exist_ok=True)
VOICES_DIR = os.path.join(BASE_DIR, "voices")
VENV_DIR   = os.path.join(BASE_DIR, "venv")
PIPER_BIN  = os.path.join(VENV_DIR, "bin", "piper")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
AUDIO_FILE  = os.path.join(AUDIO_DIR, "stt.wav")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "phi3:mini"
SAMPLE_RATE    = 16000
CHANNELS       = 1
RECORD_SECONDS = 16
WAKE_SECONDS   = 3
RMS_THRESHOLD  = 200

CHAR_NAME     = "DREAM"
SYSTEM_PROMPT = (
    "You are {name}, a sharp, witty AI assistant with a dry sense of humour. "
    "Keep answers concise — two or three sentences at most unless asked for detail."
).format(name=CHAR_NAME)

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    CHAR_NAME     = cfg["Config"]["DREAM"]["CharName"]
    SYSTEM_PROMPT = cfg["Config"]["DREAM"]["SystemPrompt"].format(name=CHAR_NAME)

WAKE_WORDS = ["hey dream","hey, dream","hi dream","hi, dream","okay dream","ok dream","dream"]
WIFI_TRIGGERS  = ["check wifi","wifi scan","scan wifi","who's on the wifi","check network","network scan","what devices","list devices","show devices"]
STATS_TRIGGERS = ["system stats","cpu usage","ram usage","memory usage","disk usage","how's the system","system status","check stats","temperature","cpu temp","how hot","system health"]

def find_voice_model():
    if not os.path.isdir(VOICES_DIR): return None
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.endswith(".onnx"): return os.path.join(VOICES_DIR, f)
    return None

VOICE_MODEL = find_voice_model()

# ── Shared state ──────────────────────────────────────────────────────────────
_state = {
    "value":    "idle",   # idle | listening | thinking | talking
    "running":  True,
}
_history   = []           # conversation history
_sse_queue = queue.Queue()  # events pushed to all SSE clients

def set_state(s):
    _state["value"] = s
    _sse_queue.put({"type": "state", "state": s})

def push_event(evt: dict):
    _sse_queue.put(evt)

# ── Audio helpers ─────────────────────────────────────────────────────────────

def _record_clip(filepath, seconds):
    if not SD_AVAILABLE:
        return False
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
        push_event({"type":"error","msg": str(e)})
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

# ── Whisper ───────────────────────────────────────────────────────────────────
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
        push_event({"type":"error","msg": str(e)})
        return ""

# ── LLM ───────────────────────────────────────────────────────────────────────

def ask_llm(prompt, history):
    set_state("thinking")
    ctx = ""
    for m in history[-6:]:
        ctx += ("You" if m["role"]=="assistant" else "Human") + f": {m['content']}\n"
    payload = {
        "model": MODEL,
        "prompt": f"System: {SYSTEM_PROMPT}\n\n{ctx}Human: {prompt}\nYou:",
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 150, "num_gpu": 20},
    }
    try:
        r = _req.post(OLLAMA_URL, json=payload, timeout=120)
        resp = r.json().get("response","").strip()
        if "<think>" in resp:
            end = resp.find("</think>")
            if end != -1: resp = resp[end+8:].strip()
        if resp.lower().startswith("dream:"): resp = resp[6:].strip()
        return resp or "I didn't catch that."
    except _req.exceptions.Timeout:
        return "That took too long. Please try again."
    except Exception as e:
        return f"Error: {e}"

# ── TTS ───────────────────────────────────────────────────────────────────────

def speak_text(text):
    """Returns path to temp wav file (caller responsible for cleanup), or None."""
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

# ── System stats ──────────────────────────────────────────────────────────────
_stats_cache = {"data": {}, "last": 0.0}

def get_system_stats():
    now = time.time()
    if now - _stats_cache["last"] < 2.0:
        return _stats_cache["data"]
    data = {}
    mem = psutil.virtual_memory()
    data.update(ram_used=mem.used/1e9, ram_total=mem.total/1e9, ram_pct=mem.percent,
                cpu_pct=psutil.cpu_percent(interval=None),
                cpu_cores=psutil.cpu_count(logical=False) or 1,
                cpu_threads=psutil.cpu_count(logical=True) or 1)
    try:
        temps = psutil.sensors_temperatures()
        all_t = []
        for key in ("coretemp","k10temp","cpu_thermal","acpitz"):
            if key in temps: all_t += [t.current for t in temps[key]]
        data["cpu_temp"] = max(all_t) if all_t else None
    except Exception:
        data["cpu_temp"] = None
    disk = psutil.disk_usage("/")
    data.update(disk_used=disk.used/1e9, disk_total=disk.total/1e9, disk_pct=disk.percent)
    _stats_cache.update(data=data, last=now)
    return data

def build_stats_summary():
    s = get_system_stats()
    parts = [
        f"CPU is at {s['cpu_pct']:.0f}%",
        f"RAM {s['ram_used']:.1f}/{s['ram_total']:.1f} GB ({s['ram_pct']:.0f}%)",
        f"Disk {s['disk_used']:.0f}/{s['disk_total']:.0f} GB ({s['disk_pct']:.0f}%)",
    ]
    if s.get("cpu_temp"):
        parts.append(f"CPU temp {s['cpu_temp']:.0f}°C")
    return " · ".join(parts)

# ── WiFi scan ─────────────────────────────────────────────────────────────────

def _ping(ip):
    r = subprocess.run(["ping","-c","1","-W","1",str(ip)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return str(ip) if r.returncode==0 else None

def _hostname(ip):
    try: return socket.gethostbyaddr(ip)[0]
    except: return ""

def _vendor(mac):
    try:
        r = _req.get(f"https://api.macvendors.com/{mac}", timeout=3)
        return r.text.strip() if r.status_code==200 else "Unknown"
    except: return "Unknown"

def _randomized(mac):
    try: return bool(int(mac.split(":")[0],16) & 0x02)
    except: return False

def _dtype(mac, hostname, vendor):
    if _randomized(mac): return "phone/tablet (random MAC)"
    h = (hostname+vendor).lower()
    if any(x in h for x in ["iphone","apple","ipad"]): return "Apple device"
    if any(x in h for x in ["samsung","android","xiaomi","huawei"]): return "Android device"
    if any(x in h for x in ["router","gateway","dlink","tp-link","asus","netgear"]): return "Router"
    if any(x in h for x in ["windows","intel","realtek"]): return "Windows PC"
    if any(x in h for x in ["ubuntu","linux","debian","raspi"]): return "Linux device"
    if any(x in h for x in ["tv","cast","roku","echo","alexa"]): return "Smart TV / IoT"
    return vendor if vendor and vendor!="Unknown" else "Unknown device"

def run_wifi_scan():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); my_ip = s.getsockname()[0]; s.close()
        subnet = my_ip.rsplit(".",1)[0]+".0/24"
        net = ipaddress.ip_network(subnet, strict=False)
        with ThreadPoolExecutor(max_workers=80) as ex:
            list(ex.map(_ping, net.hosts()))
        arp_direct = {}
        if SCAPY_AVAILABLE:
            scapy_conf.verb = 0
            pkt = Ether(dst="ff:ff:ff:ff:ff:ff")/ARP(pdst=subnet)
            answered, _ = srp(pkt, timeout=3, verbose=False, retry=1)
            arp_direct = {r.psrc: r.hwsrc for _,r in answered}
        arp_cache = {}
        try:
            out = subprocess.check_output(["ip","neigh"], text=True)
            for line in out.splitlines():
                p = line.split()
                if "lladdr" in p and "FAILED" not in line:
                    arp_cache[p[0]] = p[p.index("lladdr")+1]
        except: pass
        devices = []
        for ip in sorted(set(arp_direct)|set(arp_cache), key=lambda x: list(map(int,x.split(".")))):
            raw_mac = arp_direct.get(ip) or arp_cache.get(ip) or "??:??:??:??:??:??"
            mac     = raw_mac.replace("-",":").upper()
            hn      = _hostname(ip)
            vnd     = _vendor(mac) if not _randomized(mac) else "N/A"
            devices.append({"ip":ip,"mac":mac,"hostname":hn,"vendor":vnd,
                            "type":_dtype(mac,hn,vnd),"me":ip==my_ip})
        return devices
    except Exception as e:
        return []

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── SSE broadcast ─────────────────────────────────────────────────────────────
_sse_clients = []
_sse_lock    = threading.Lock()

def _broadcast(evt):
    data = "data: " + json.dumps(evt) + "\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_clients:
            try: q.put_nowait(data)
            except: dead.append(q)
        for q in dead: _sse_clients.remove(q)

def _sse_dispatcher():
    """Single thread draining the global queue and broadcasting."""
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
        # send current state immediately
        yield "data: " + json.dumps({"type":"state","state":_state["value"]}) + "\n\n"
        try:
            while True:
                try: data = client_q.get(timeout=25)
                except queue.Empty:
                    yield "data: " + json.dumps({"type":"ping"}) + "\n\n"
                    continue
                yield data
        except GeneratorExit:
            with _sse_lock:
                if client_q in _sse_clients: _sse_clients.remove(client_q)

    return Response(stream_with_context(gen()),
                    content_type="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", char_name=CHAR_NAME)

@app.route("/api/status")
def api_status():
    # check ollama
    ollama_ok = False
    try:
        r = _req.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except: pass
    return jsonify({
        "state":       _state["value"],
        "ollama":      ollama_ok,
        "piper":       os.path.exists(PIPER_BIN),
        "voice_model": VOICE_MODEL is not None,
        "sd":          SD_AVAILABLE,
        "scapy":       SCAPY_AVAILABLE,
    })

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Accepts JSON {text: "..."}  (typed input)
    or multipart/form-data with field 'audio' (uploaded WAV from browser mic).
    Returns JSON {reply, audio_url?, state}
    """
    user_text = ""

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("audio")
        if f:
            f.save(AUDIO_FILE)
            if check_audio_levels():
                set_state("listening")
                push_event({"type":"transcript","role":"system","text":"Transcribing…"})
                user_text = transcribe_file(AUDIO_FILE)
    else:
        data = request.get_json(silent=True) or {}
        user_text = data.get("text","").strip()

    if not user_text:
        set_state("idle")
        return jsonify({"error":"no input","state":"idle"}), 400

    push_event({"type":"transcript","role":"user","text":user_text})

    lower = user_text.lower()

    # ── Special commands ───────────────────────────────────────────────────
    if any(w in lower for w in ["goodbye","exit","quit","bye","shut down","shutdown"]):
        reply = "Goodbye."
        push_event({"type":"transcript","role":"assistant","text":reply})
        set_state("idle")
        return jsonify({"reply":reply, "state":"idle"})

    if any(t in lower for t in WIFI_TRIGGERS):
        set_state("thinking")
        push_event({"type":"transcript","role":"system","text":"Scanning network…"})
        devices = run_wifi_scan()
        if not devices:
            reply = "Network scan failed or no devices found."
        else:
            lines = [f"Found {len(devices)} device(s) on the network."]
            for d in devices:
                tag = " ← this machine" if d["me"] else ""
                lines.append(f"• {d['ip']}  {d['type']}{' — '+d['hostname'] if d['hostname'] else ''}{tag}")
            reply = "\n".join(lines)
        push_event({"type":"transcript","role":"assistant","text":reply})
        push_event({"type":"wifi","devices":devices})
        set_state("idle")
        return jsonify({"reply":reply,"devices":devices,"state":"idle"})

    if any(t in lower for t in STATS_TRIGGERS):
        stats = get_system_stats()
        reply = build_stats_summary()
        push_event({"type":"transcript","role":"assistant","text":reply})
        push_event({"type":"stats","data":stats})
        set_state("idle")
        return jsonify({"reply":reply,"stats":stats,"state":"idle"})

    # ── LLM ────────────────────────────────────────────────────────────────
    reply = ask_llm(user_text, _history)
    _history.append({"role":"user","content":user_text})
    _history.append({"role":"assistant","content":reply})
    if len(_history) > 12: del _history[:-12]

    push_event({"type":"transcript","role":"assistant","text":reply})

    # ── TTS ────────────────────────────────────────────────────────────────
    audio_url = None
    wav_path  = speak_text(reply)
    if wav_path:
        set_state("talking")
        # serve the file under /audio/<filename>
        fname = os.path.basename(wav_path)
        audio_url = f"/audio/{fname}"

    set_state("idle")
    return jsonify({"reply":reply,"audio_url":audio_url,"state":"idle"})

@app.route("/audio/<path:fname>")
def serve_audio(fname):
    import flask
    safe = os.path.join(AUDIO_DIR, os.path.basename(fname))
    if not os.path.exists(safe):
        return "", 404
    return flask.send_file(safe, mimetype="audio/wav")

@app.route("/api/stats")
def api_stats():
    return jsonify(get_system_stats())

@app.route("/api/wifi")
def api_wifi():
    set_state("thinking")
    devices = run_wifi_scan()
    set_state("idle")
    return jsonify({"devices": devices})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)#change adress to 5008