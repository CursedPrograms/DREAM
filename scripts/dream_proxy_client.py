"""
dream_proxy_client.py - DREAM's side of dream_proxy.py: her sensor board (and webcam) on another PC.

    proxy = find_proxy()          # (ip, port) of a DREAM proxy on the network, or None. Cached: looks
                                  # again at most once a minute (config.json DREAM.SensorProxy pins one)
    ser = open_board()            # the proxy's board as a serial-port look-alike, or None
    ser.readline() / ser.write(b"ALARM OFF\n") / ser.port / ser.is_open / with ser: ...

app.py's sensor_watcher() uses it when no board is plugged into her own PC, so everything that reads
the board (presence, the alarm, the dashboard's SENSORS panel, her mind) works the same either way.
"""

import json
import os
import socket
import threading
import time
import urllib.request

SERVICE = "_dream-proxy._tcp.local."
LOOK_EVERY_S = 60
DEFAULT_PORT = 5016
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_found = {"addr": None, "at": 0.0}
_lock = threading.Lock()


def _pinned():
    """config.json DREAM.SensorProxy: "192.168.0.57" or "192.168.0.57:5016" (empty: look on the network)."""
    try:
        with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
            v = json.load(f)["Config"]["DREAM"].get("SensorProxy", "").strip()
    except (OSError, ValueError, KeyError):
        v = ""
    if not v:
        return None
    host, _, port = v.partition(":")
    return host, int(port) if port.isdigit() else DEFAULT_PORT


def _alive(addr):
    try:
        with urllib.request.urlopen(f"http://{addr[0]}:{addr[1]}/ping", timeout=2) as r:
            return r.read().decode().strip() == "DREAM_PROXY"
    except Exception:
        return False


def _browse(seconds=3.0):
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return None
    found = []

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=2000)
            if info and info.addresses:
                found.append((socket.inet_ntoa(info.addresses[0]), info.port))

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        ServiceBrowser(zc, SERVICE, Listener())
        deadline = time.time() + seconds
        while time.time() < deadline and not found:
            time.sleep(0.1)
    finally:
        zc.close()
    return found[0] if found else None


def find_proxy(force=False):
    """(ip, port) of a DREAM proxy, or None. Looks again at most once a minute."""
    with _lock:
        addr = _found["addr"]
        if addr and _alive(addr):
            return addr
        if not force and time.time() - _found["at"] < LOOK_EVERY_S:
            return None
        _found["at"] = time.time()
        addr = _pinned() or _browse()
        _found["addr"] = addr if addr and _alive(addr) else None
        if _found["addr"]:
            print(f"[DREAM] found a DREAM proxy at {addr[0]}:{addr[1]}")
        return _found["addr"]


def proxy_info(addr):
    try:
        with urllib.request.urlopen(f"http://{addr[0]}:{addr[1]}/info", timeout=2) as r:
            return json.loads(r.read())
    except Exception:
        return {}


class ProxySerial:
    """The proxy's board, looking like the serial.Serial app.py's sensor_watcher() reads."""

    def __init__(self, addr):
        self.addr = addr
        self.port = f"proxy {addr[0]}:{addr[1]}"
        self._resp = urllib.request.urlopen(f"http://{addr[0]}:{addr[1]}/sensor/lines", timeout=15)
        self.is_open = True

    def readline(self):
        while True:
            line = self._resp.readline()          # the proxy sends a blank line every 5 s at least
            if not line:
                raise ConnectionError("the proxy went quiet")
            s = line.decode(errors="ignore").strip()
            if s == "PROXY_BOARD OFF":
                raise ConnectionError("the proxy's board was unplugged")
            if s.startswith("PROXY_BOARD"):
                continue
            return line

    def write(self, data):
        req = urllib.request.Request(f"http://{self.addr[0]}:{self.addr[1]}/sensor/command", data=data, method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status != 200:
                raise OSError("the proxy has no board")
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.is_open = False
        try:
            self._resp.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def open_board():
    """The proxy's sensor board, or None (no proxy, or one without a board)."""
    addr = find_proxy()
    if not addr or not proxy_info(addr).get("board"):
        return None
    try:
        return ProxySerial(addr)
    except Exception:
        return None
