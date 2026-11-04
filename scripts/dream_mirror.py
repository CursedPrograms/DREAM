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
    GET /video/<clip>.mp4  the clip itself, full quality (RIFT's call window plays it in step)
    GET /speech/<id>.wav   the line as it is, full quality (the call window's sound)
    GET /transcript?after=N  the conversation since line N: what was said to her (her mic, her page,
                         a call) and everything she said, so a call window reads like her dashboard
    GET /camera.mjpeg?w=320
                         what her camera sees, with her detections drawn on (people, objects,
                         faces: surveillance_core.analyse_frame), as MJPEG at up to 3 fps.
                         Only runs while someone watches: the camera opens for it (unless
                         dream.py's surveillance already has it) and closes a few seconds after
                         the last viewer leaves; detection runs on ImageDevice (the CPU by
                         default), so it never slows her conversation. Anyone on the network
                         can watch it: config.json DREAM.MirrorCamera false turns it off.

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


_chat = []          # [{"seq", "role": "user" | "assistant", "text", "ts"}], the last 200 lines
_chat_seq = 0


def note_line(role, text):
    """A line of the conversation (app.py's push_event hands them over)."""
    global _chat_seq
    text = (text or "").strip()
    if role not in ("user", "assistant") or not text:
        return
    with _lock:
        if _chat and _chat[-1]["role"] == role and _chat[-1]["text"] == text and time.time() - _chat[-1]["ts"] < 5:
            return   # the same line twice in a row (her reply and its spoken copy)
        _chat_seq += 1
        _chat.append({"seq": _chat_seq, "role": role, "text": text, "ts": time.time()})
        del _chat[:-200]


def transcript(after=0):
    with _lock:
        return {"seq": _chat_seq, "lines": [l for l in _chat if l["seq"] > after]}


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


# ---- her camera, with what she detects drawn on, while someone is watching ----
CAMERA_FPS = 3
_cam = {"viewers": 0, "last_viewer": 0.0, "jpg": None, "seq": 0, "error": ""}
_cam_cond = threading.Condition()
_cam_thread = None


def camera_allowed():
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            return bool(json.load(f)["Config"]["DREAM"].get("MirrorCamera", True))
    except (OSError, ValueError, KeyError, TypeError):
        return True


def _camera_worker():
    import cv2
    try:
        from dream_mind import frames
    except Exception:
        frames = None
    analyse = None
    try:   # her detector: YOLO + faces, on ImageDevice
        from surveillance_core import analyse_frame as analyse
    except Exception as e:
        print(f"[DREAM] Mirror camera: no detections ({e}) - the plain picture")
    cap = None
    while True:
        with _cam_cond:
            watched = _cam["viewers"] > 0 or time.time() - _cam["last_viewer"] < 5
        if not watched:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.3)
            continue
        t0 = time.time()
        frame = None
        if cap is None and frames is not None:
            frame, _ = frames.latest(1.0)          # dream.py's surveillance has the camera: share it
        if frame is None:
            if cap is None:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    _cam["error"] = "no camera"
                    time.sleep(2)
                    continue
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            if frames is not None:
                frames.publish(frame)              # her mind (faces, looks) sees it too
        try:
            out = analyse(frame, "rita")[0] if analyse else frame
        except Exception as e:
            _cam["error"] = str(e)
            out = frame
        with _cam_cond:
            _cam["jpg"] = out
            _cam["seq"] += 1
            _cam["error"] = ""
            _cam_cond.notify_all()
        time.sleep(max(0.0, 1.0 / CAMERA_FPS - (time.time() - t0)))


def _camera_frames(width):
    """Yields JPEGs of her annotated camera, each new one, while the client stays."""
    import cv2
    global _cam_thread
    with _cam_cond:
        _cam["viewers"] += 1
        if _cam_thread is None:
            _cam_thread = threading.Thread(target=_camera_worker, daemon=True, name="dream-mirror-camera")
            _cam_thread.start()
    seen = -1
    try:
        while True:
            with _cam_cond:
                _cam_cond.wait_for(lambda: _cam["seq"] != seen, timeout=2.0)
                frame, seen = _cam["jpg"], _cam["seq"]
            if frame is None:
                continue
            if width and frame.shape[1] > width:
                frame = cv2.resize(frame, (width, frame.shape[0] * width // frame.shape[1]), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                yield buf.tobytes()
    finally:
        with _cam_cond:
            _cam["viewers"] -= 1
            _cam["last_viewer"] = time.time()


def start_server(port, get_state, audio_dir, videos_dir=None):
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

        def _file(self, path, ctype):
            """A file, with byte ranges: a <video> seeks with them."""
            size = os.path.getsize(path)
            start, end = 0, size - 1
            rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if rng and (rng.group(1) or rng.group(2)):
                if rng.group(1):
                    start = int(rng.group(1))
                    end = int(rng.group(2)) if rng.group(2) else size - 1
                else:
                    start = max(0, size - int(rng.group(2)))
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(65536, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                m = re.fullmatch(r"/video/([\w.-]+)\.mp4", u.path)
                if m and videos_dir:
                    path = os.path.join(videos_dir, m.group(1) + ".mp4")
                    return self._file(path, "video/mp4") if os.path.isfile(path) else self._send(404, b"no such clip", "text/plain")
                m = re.fullmatch(r"/speech/([\w.-]+)\.wav", u.path)
                if m:
                    path = os.path.join(audio_dir, m.group(1) + ".wav")
                    return self._file(path, "audio/wav") if os.path.isfile(path) else self._send(404, b"no such line", "text/plain")
                if u.path == "/status":
                    state, sleeping = get_state()
                    return self._send(200, json.dumps(status(state, sleeping)).encode(), "application/json")
                if u.path == "/transcript":
                    after = int((parse_qs(u.query).get("after") or ["0"])[0])
                    return self._send(200, json.dumps(transcript(after)).encode(), "application/json")
                if u.path == "/camera.mjpeg":
                    if not camera_allowed():
                        return self._send(403, b"MirrorCamera is off in DREAM's config.json", "text/plain")
                    width = int((parse_qs(u.query).get("w") or ["320"])[0])
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    for jpg in _camera_frames(width):
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
                    return
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
