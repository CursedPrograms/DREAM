"""
dream_mirror.py - what DREAM is showing and saying right now, for a robot that mirrors her
(RITA, the fleet's handheld remote: her clips from its SD card, her voice on its speaker).

Her page (static/js/dream.js) reports each clip it starts and each line it starts speaking
(POST /api/dream/now); app.py keeps it here. Robots read it from a small plain-HTTP server
of its own (config.json Dashboard.MirrorPort, 5015), because her dashboard is HTTPS with a
self-signed certificate, and an ESP32 pays about a second and 40 KB of RAM for every TLS
handshake. It only answers two read-only questions:

    GET /status          {"state", "sleeping", "clip", "clip_ms", "loop", "speech", "speech_ms"}
                         clip: the clip's name without folder or extension ("idle2"), clip_ms:
                         how far into it she is now; speech: the line being spoken (an id) or
                         null, speech_ms: how far into it
    GET /speech/<id>?from_ms=N
                         that line as raw unsigned 8-bit mono PCM at 16 kHz (what the ESP32's
                         DAC plays), from N ms in, so a robot that arrives late starts in sync

Nothing here can change anything in DREAM.
"""

import json
import os
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

RATE = 16000
_now = {"clip": None, "clip_ts": 0.0, "loop": True, "speech": None, "speech_ts": 0.0}
_lock = threading.Lock()


def _name(path):
    return os.path.splitext(os.path.basename(path or ""))[0] or None


def note(clip=None, loop=True, speech=None, speech_done=False):
    """The page started a clip and/or a spoken line (or finished speaking)."""
    t = time.time()
    with _lock:
        if clip:
            _now.update(clip=_name(clip), clip_ts=t, loop=bool(loop))
        if speech:
            _now.update(speech=_name(speech), speech_ts=t)
        elif speech_done:
            _now.update(speech=None, speech_ts=0.0)


def status(state, sleeping):
    t = time.time()
    with _lock:
        n = dict(_now)
    return {
        "state": state, "sleeping": bool(sleeping),
        "clip": n["clip"], "clip_ms": int((t - n["clip_ts"]) * 1000) if n["clip"] else 0, "loop": n["loop"],
        "speech": n["speech"], "speech_ms": int((t - n["speech_ts"]) * 1000) if n["speech"] else 0,
    }


def pcm8(wav_path, from_ms=0):
    """A WAV as unsigned 8-bit mono at 16 kHz, from `from_ms` in."""
    with wave.open(wav_path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:   # 24/32-bit: rare from Piper, take the top 16 bits
        x = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width)[:, -2:].copy().view("<i2").astype(np.float32).ravel() / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != RATE and len(x):
        t = np.arange(0, len(x) / rate, 1.0 / RATE)
        x = np.interp(t, np.arange(len(x)) / rate, x)
    x = x[int(max(0, from_ms) * RATE / 1000):]
    return (np.clip(x * 0.9, -1, 1) * 127 + 128).astype(np.uint8).tobytes()


def start_server(port, get_state, audio_dir):
    """The robots' read-only window onto her. get_state() -> (state, sleeping)."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/status":
                    state, sleeping = get_state()
                    return self._send(200, json.dumps(status(state, sleeping)).encode(), "application/json")
                m = re.fullmatch(r"/speech/([\w.-]+)", u.path)
                if m:
                    wav = os.path.join(audio_dir, m.group(1) + ".wav")
                    if not os.path.isfile(wav):
                        return self._send(404, b"no such line", "text/plain")
                    from_ms = int((parse_qs(u.query).get("from_ms") or ["0"])[0])
                    return self._send(200, pcm8(wav, from_ms), "application/octet-stream")
                self._send(404, b"not here", "text/plain")
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self._send(500, str(e).encode(), "text/plain")
                except Exception:
                    pass

    try:
        srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    except OSError as e:
        print(f"[DREAM] Mirror for robots not started on :{port}: {e}")
        return None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True, name="dream-mirror").start()
    print(f"[DREAM] Mirror for robots (RITA): http://0.0.0.0:{port}/status")
    return srv
