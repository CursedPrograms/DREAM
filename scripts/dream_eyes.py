"""
dream_eyes.py - where DREAM is: which page of hers is her body, and which camera she sees through.

Her BODY is the newest page of hers that's open (her own window on this PC, the DREAM proxy's window on
another, her site on your phone): it shows her, speaks for her and listens for her. Every other page
waits on standby, still and silent, saying where she is. Close the newest one and within a few seconds
the one before it is her body again (close the proxy: she's back on this PC).

Mid-conversation she stays put: a page opened while she's talking with you (or listening, or
thinking) waits until she's been quiet for QUIET_S, then she moves to it. Talking TO a page (typing,
tap-to-talk, "Hey DREAM") brings her there at once: claim(). "Newest" is by when her
server first heard from the page, not the device's own clock (a phone's can be ahead of the PC's).

Her EYES, best first:
    1. the newest open page that has a camera (it sends frames: POST /api/eyes/frame)
    2. the DREAM proxy's webcam (dream_proxy.py), on the PC with her sensor board
    3. her own PC's webcam, as before: opened only when something needs a picture
Frames go into dream_mind/frames.py, which her mind, her faces, her looks and RITA / RIFT's views read.

    body_beat(page, started_ms, label) -> {"active": bool, "label": where she is}
    page_frame(page, started_ms, jpeg) / page_bye(page) / status()
"""

import threading
import time

FRESH_S = 3.0          # a page that hasn't checked in for this long is closed
QUIET_S = 60.0         # a conversation is over after this long without one (engaged())
PROXY_FPS = 4

_pages = {}            # id -> {"started": ms, "label": str, "beat": time, "cam": time of its last frame}
_lock = threading.Lock()
_body = {"page": None, "busy_until": 0.0}   # where she is, and until when she's mid-conversation there
_proxy = {"addr": None, "thread": None, "stop": False}


def _frames():
    try:
        from dream_mind import frames
        return frames
    except Exception:
        return None


def _newest(with_camera=False):
    now = time.time()
    with _lock:
        for k in [k for k, v in _pages.items() if now - max(v["beat"], v["cam"]) > 60]:
            del _pages[k]
        live = {k: v for k, v in _pages.items()
                if now - (v["cam"] if with_camera else max(v["beat"], v["cam"])) < FRESH_S}
    return max(live, key=lambda k: live[k]["started"]) if live else None


def _current():
    """Her body: the newest page, unless she's mid-conversation on another one that's still open."""
    newest = _newest()
    now = time.time()
    with _lock:
        here = _body["page"]
        still_open = here in _pages and now - max(_pages[here]["beat"], _pages[here]["cam"]) < FRESH_S
        if not (still_open and newest != here and now < _body["busy_until"]):
            _body["page"] = newest
        return _body["page"]


def claim(page, started_ms=0, label=None):
    """You're talking to this page: she comes here now (and stays while you talk)."""
    _touch(page, started_ms, label)
    with _lock:
        _body["page"] = page
        _body["busy_until"] = time.time() + QUIET_S


def body_label():
    """Where she is now ("this PC", "a phone"...)."""
    body = _current()
    with _lock:
        return _pages[body]["label"] if body in _pages else ""


def engaged():
    """She's talking, listening or thinking (app.py): her body stays where it is for QUIET_S more."""
    with _lock:
        _body["busy_until"] = time.time() + QUIET_S


def _touch(page, started_ms, label=None, cam=False):
    with _lock:
        # "started": when her server first heard from it (the device's clock may be off)
        p = _pages.setdefault(page, {"started": time.time() * 1000, "label": label or "a page", "beat": 0.0, "cam": 0.0})
        if label:
            p["label"] = label
        p["cam" if cam else "beat"] = time.time()


def body_beat(page, started_ms, label):
    """A page checking in (every second). Is it her body now, and where is she?"""
    _touch(page, started_ms, label)
    body = _current()
    with _lock:
        where = _pages[body]["label"] if body in _pages else label
    return {"active": body == page, "label": where}


def is_body(page):
    """Is this page her body right now (the newest open one)? Only it lends her its phone's sensors."""
    return bool(page) and _current() == page


def page_frame(page, started_ms, jpeg):
    """A frame from one of her pages' cameras. It becomes what she sees if that page is the newest
    one with a camera."""
    _touch(page, started_ms, cam=True)
    if _newest(with_camera=True) != page:
        return False
    try:
        import cv2
        import numpy as np
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return False
    f = _frames()
    if f is not None and frame is not None:
        f.publish(frame)
        return True
    return False


def page_bye(page):
    """The page closed (sendBeacon): the one before it is her body again at once."""
    with _lock:
        _pages.pop(page, None)


def _proxy_loop():
    """The proxy's webcam while no page's camera is her eyes, a few frames a second."""
    import urllib.request
    import cv2
    import numpy as np
    import dream_proxy_client
    while not _proxy["stop"]:
        if _newest(with_camera=True):
            time.sleep(0.5)
            continue
        addr = dream_proxy_client.find_proxy()
        if not addr or not dream_proxy_client.proxy_info(addr).get("camera"):
            time.sleep(5)
            continue
        _proxy["addr"] = addr
        try:
            with urllib.request.urlopen(f"http://{addr[0]}:{addr[1]}/camera.mjpeg?w=480", timeout=10) as r:
                buf, last = b"", 0.0
                while not _proxy["stop"] and not _newest(with_camera=True):
                    chunk = r.read(16384)
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.rfind(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a) if a >= 0 else -1
                    if a >= 0 and b > a:
                        jpg, buf = buf[a:b + 2], buf[b + 2:]
                        if time.time() - last >= 1.0 / PROXY_FPS:
                            last = time.time()
                            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            f = _frames()
                            if f is not None and frame is not None:
                                f.publish(frame)
                    if len(buf) > 1 << 20:
                        buf = b""
        except Exception:
            time.sleep(3)
        _proxy["addr"] = None


def start():
    """Starts following the proxy's webcam (pages need nothing started: they check in)."""
    if _proxy["thread"] is None:
        _proxy["thread"] = threading.Thread(target=_proxy_loop, daemon=True, name="dream-eyes-proxy")
        _proxy["thread"].start()


def status():
    body, eyes = _current(), _newest(with_camera=True)
    with _lock:
        where = _pages[body]["label"] if body in _pages else None
    source = "page" if eyes else "proxy" if _proxy["addr"] else "local"
    out = {"body": body, "where": where, "eyes": source, "eyes_page": eyes}
    if _proxy["addr"]:
        out["proxy"] = f"{_proxy['addr'][0]}:{_proxy['addr'][1]}"
    return out
