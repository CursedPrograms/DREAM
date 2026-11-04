"""
dream_proxy.py - DREAM's body on another PC: her sensor board, her webcam, and her window.

Two PCs: the AI PC runs DREAM (app.py: Whisper, the language model, her voice, her mind). This one, the
PC with her sensor board and a webcam, runs this. It opens her window here, full screen: her page from
the AI PC (her videos, her voice on this PC's speakers, this PC's mic for "Hey DREAM" and what you say,
this PC's camera as her eyes). The newest open page of hers is her body, so while this window is open
she lives here; close it, or stop this proxy, and within a few seconds she's back on the AI PC.

DREAM runs on one PC; her Arduino sensor board (dream_sensors.ino: the mmWave presence radar, the PIR
alarm, the RGB lights) can be plugged into another. Run this there (run_proxy.bat). It finds the board
the way DREAM does (it answers "I am Dream"), and serves it on the network so DREAM uses it as if it
were plugged into her own PC:

    GET  /ping              "DREAM_PROXY"
    GET  /info              {"board": "COM6" or null, "camera": true/false}
    GET  /sensor/lines      every line the board prints, as it prints it (a stream, one per line)
    POST /sensor/command    a line to the board ("ALARM OFF", "RGB RED")
    GET  /camera.mjpeg?w=480   this PC's webcam (only while someone watches)

It announces itself on the network (zeroconf, "_dream-proxy._tcp"), so DREAM finds it by herself: when
no board is plugged into her own PC she looks for a proxy about once a minute. With neither, she
assumes nothing is plugged in. Port 5016 (DREAM Robotics range).

    python dream_proxy.py [--dream 192.168.0.100] [--port 5016] [--no-camera] [--no-window] [--windowed]

--dream: the AI PC's address (otherwise found on the network: DREAM announces herself).

Needs pyserial, zeroconf and (for the camera) opencv-python: DREAM's requirements.txt has them.
"""

import argparse
import os
import queue
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

PORT = 5016
BAUD = 9600            # dream_sensors.ino
SERVICE = "_dream-proxy._tcp.local."

board = {"port": None, "serial": None}
board_lock = threading.Lock()
listeners = []         # one queue per /sensor/lines client
listeners_lock = threading.Lock()


def broadcast(line):
    with listeners_lock:
        for q in listeners:
            q.put(line)


def board_loop():
    """Keeps the board connected; every line it prints goes to every listener."""
    import serial
    from board_id import find_board
    warned = False
    while True:
        port = find_board("dream", BAUD)
        if not port:
            if not warned:
                print('[proxy] no board answered "I am Dream": looking again every 10 s')
                warned = True
            time.sleep(10)
            continue
        warned = False
        try:
            ser = serial.Serial(port, BAUD, timeout=1)
            time.sleep(2)                       # the board resets when the port opens
            with board_lock:
                board.update(port=port, serial=ser)
            print(f"[proxy] sensor board on {port}")
            broadcast("PROXY_BOARD ON")
            with ser:
                while True:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line:
                        broadcast(line)
        except Exception as e:
            print(f"[proxy] sensor board lost: {e}")
        finally:
            with board_lock:
                board.update(port=None, serial=None)
            broadcast("PROXY_BOARD OFF")
            time.sleep(2)


def board_write(line):
    with board_lock:
        ser = board["serial"]
        if ser is None:
            return False
        try:
            ser.write((line.strip() + "\n").encode())
            ser.flush()
            return True
        except Exception:
            return False


# ---- the webcam, only while someone watches ----
cam = {"viewers": 0, "jpg": None, "seq": 0, "has": None}
cam_cond = threading.Condition()


def camera_loop():
    import cv2
    cap = None
    while True:
        with cam_cond:
            watched = cam["viewers"] > 0
        if not watched:
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.3)
            continue
        if cap is None:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if not cap.isOpened():
                cap.release()
                cap = None
                cam["has"] = False
                time.sleep(3)
                continue
            cam["has"] = True
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        with cam_cond:
            cam["jpg"] = frame
            cam["seq"] += 1
            cam_cond.notify_all()
        time.sleep(0.12)


def camera_present():
    """Whether this PC has a webcam (looked at once, quickly)."""
    if cam["has"] is None:
        try:
            import cv2
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            cam["has"] = bool(cap.isOpened())
            cap.release()
        except Exception:
            cam["has"] = False
    return cam["has"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/plain"):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/ping":
            return self._send(200, "DREAM_PROXY")
        if u.path == "/info":
            import json
            return self._send(200, json.dumps({"board": board["port"], "camera": ARGS.camera and camera_present()}),
                              "application/json")
        if u.path == "/sensor/lines":
            q = queue.Queue()
            with listeners_lock:
                listeners.append(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(f"PROXY_BOARD {'ON' if board['port'] else 'OFF'}\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        line = q.get(timeout=5)
                    except queue.Empty:
                        line = ""                 # a blank line now and then: is DREAM still there?
                    self.wfile.write((line + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with listeners_lock:
                    listeners.remove(q)
            self.close_connection = True
            return
        if u.path == "/camera.mjpeg" and ARGS.camera:
            import cv2
            width = int((parse_qs(u.query).get("w") or ["480"])[0])
            with cam_cond:
                cam["viewers"] += 1
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Connection", "close")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with cam_cond:
                        cam_cond.wait_for(lambda: cam["seq"] != seen, timeout=3)
                        frame, seen = cam["jpg"], cam["seq"]
                    if frame is None:
                        if cam["has"] is False:
                            break
                        continue
                    if frame.shape[1] > width:
                        frame = cv2.resize(frame, (width, frame.shape[0] * width // frame.shape[1]))
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        jpg = buf.tobytes()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with cam_cond:
                    cam["viewers"] -= 1
            self.close_connection = True
            return
        self._send(404, "not here")

    def do_POST(self):
        if urlparse(self.path).path == "/sensor/command":
            line = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode(errors="ignore")
            return self._send(200 if board_write(line) else 503, "OK" if board["port"] else "no board")
        self._send(404, "not here")


def announce(port):
    """zeroconf: so DREAM finds this proxy by herself."""
    try:
        from zeroconf import ServiceInfo, Zeroconf
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        name = f"DREAM proxy on {socket.gethostname()}"
        info = ServiceInfo(SERVICE, f"{name}.{SERVICE}", addresses=[socket.inet_aton(ip)], port=port,
                           properties={"host": socket.gethostname()})
        zc = Zeroconf()
        zc.register_service(info)
        print(f"[proxy] announced as '{name}' at {ip}:{port}")
        return zc
    except Exception as e:
        print(f"[proxy] couldn't announce on the network ({e}): set SensorProxy in DREAM's config.json to this PC's address")
        return None


# ---- her window: her page from the AI PC, in an Edge / Chrome app window ----
DREAM_PORT = 5009
DREAM_ZEROCONF = "_flask-link._tcp.local."


def find_dream(seconds=4.0):
    """The AI PC's address: DREAM announces herself on the network (zeroconf, named DREAM)."""
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return None
    found = []

    class Listener:
        def add_service(self, zc, type_, name):
            if not name.upper().startswith("DREAM"):
                return
            info = zc.get_service_info(type_, name, timeout=2000)
            if info and info.addresses:
                found.append(socket.inet_ntoa(info.addresses[0]))

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, DREAM_ZEROCONF, Listener())
        deadline = time.time() + seconds
        while time.time() < deadline and not found:
            time.sleep(0.1)
    finally:
        zc.close()
    return found[0] if found else None


def browser():
    """Edge or Chrome (an app window: no tabs, no address bar)."""
    import shutil
    for c in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]:
        if os.path.exists(c):
            return c
    for name in ("microsoft-edge", "google-chrome", "chromium", "chromium-browser", "msedge", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    return None


def open_window(dream_ip, windowed=False):
    """Her page from the AI PC, full screen. Its own browser profile: her certificate accepted (it's
    her own, self-signed), the mic and camera allowed, sound without a click. Returns the process."""
    import subprocess
    exe = browser()
    if not exe:
        print("[proxy] no Edge or Chrome here: open her page yourself:", f"https://{dream_ip}:{DREAM_PORT}/dream.html?autostart=1")
        return None
    url = f"https://{dream_ip}:{DREAM_PORT}/dream.html?autostart=1&body={socket.gethostname()}"
    profile = os.path.join(os.path.expanduser("~"), ".dream_proxy_window")
    args = [exe, f"--app={url}", f"--user-data-dir={profile}", "--ignore-certificate-errors",
            "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
            "--no-first-run", "--disable-features=Translate"]
    if not windowed:
        args.append("--start-fullscreen")
    print(f"[proxy] her window: {url}")
    return subprocess.Popen(args)


def main():
    global ARGS
    p = argparse.ArgumentParser(description="DREAM's sensor board (and webcam) on another PC")
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--no-camera", dest="camera", action="store_false", help="don't share this PC's webcam")
    p.add_argument("--dream", default="", help="the AI PC's address (default: found on the network)")
    p.add_argument("--no-window", dest="window", action="store_false", help="just the board and webcam: no window of her here")
    p.add_argument("--windowed", action="store_true", help="her window not full screen")
    ARGS = p.parse_args()
    threading.Thread(target=board_loop, daemon=True).start()
    if ARGS.camera:
        threading.Thread(target=camera_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    srv.daemon_threads = True
    zc = announce(ARGS.port)
    print(f"[proxy] DREAM proxy on :{ARGS.port} (camera {'shared' if ARGS.camera else 'off'}). Ctrl+C stops it.")
    window = None
    if ARGS.window:
        dream_ip = ARGS.dream or find_dream()
        if dream_ip:
            window = open_window(dream_ip, ARGS.windowed)
        else:
            print("[proxy] DREAM isn't on the network yet: start her on the AI PC, or pass --dream <its address>")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if window and window.poll() is None:
            window.terminate()             # her window goes with the proxy: she's back on the AI PC
        if zc:
            zc.close()


ARGS = argparse.Namespace(camera=True, port=PORT, dream="", window=True, windowed=False)

if __name__ == "__main__":
    main()
