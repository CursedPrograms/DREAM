"""
web_launcher.py - `python scripts/dream.py --web`: run DREAM in the browser.

Starts app.py (the web server) if it isn't already running, then opens
/dream.html. The page does what dream.py's window does — the avatar videos,
wake word, sleep/wake, sensor reactions — using the browser's microphone and
speakers. Ctrl+C stops the server if this launched it.
"""

import json
import os
import subprocess
import sys
import time
import webbrowser

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_TIMEOUT_S = 90  # first start loads models and can be slow


def _config():
    with open(os.path.join(BASE_DIR, "config.json"), encoding="utf-8") as f:
        return json.load(f)["Config"]["ComCentre"]


def _up(url):
    """True if app.py answers at `url` (its /ping route). The HTTPS one is self-signed."""
    try:
        return requests.get(url + "/ping", timeout=1, verify=False).ok
    except requests.RequestException:
        return False


def _page_url(cfg):
    """A URL for /dream.html that works without a certificate warning if we can:
    app.py's plain-HTTP localhost port, else its main port."""
    local = f"http://localhost:{cfg.get('LocalPort', 5010)}"
    main = f"http://localhost:{cfg.get('Port', 5009)}"
    for base in (local, main, main.replace("http://", "https://")):
        if _up(base):
            if base != local:
                # app.py opens its two ports a moment apart; give the no-warning one a few seconds
                for _ in range(5):
                    if _up(local):
                        return local + "/dream.html"
                    time.sleep(1)
            return base + "/dream.html"
    return None


def run():
    try:
        requests.packages.urllib3.disable_warnings()
    except Exception:
        pass
    cfg = _config()

    proc = None
    url = _page_url(cfg)
    if url:
        print(f"app.py is already running - opening {url}")
    else:
        print("Starting app.py...")
        proc = subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "app.py")], cwd=BASE_DIR)
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline and proc.poll() is None:
            url = _page_url(cfg)
            if url:
                break
            time.sleep(1)
        if not url:
            print("app.py did not come up - check its output above.")
            if proc.poll() is None:
                proc.terminate()
            return

    print(f"Opening {url}")
    webbrowser.open(url)

    if proc is None:
        return
    print("DREAM web is running. Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()
