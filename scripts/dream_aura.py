"""dream_aura.py - DREAM asks AURA, the fleet's weather station, for the weather or a robot's readings.

They do it the fleet's way first, in Brainfuck: DREAM beeps "wx?" (utterance 17 of RIFT's
Fleet/talk_bf.json) in her deep voice, AURA beeps the sky back in one word ("rain", "sun", ...), and
then DREAM says AURA's whole answer in words.

    from dream_aura import is_aura_question, ask_aura
    if is_aura_question(text):
        reply = ask_aura(text)

AURA is found in RIFT's fleet list (her "ask:<port>" capability), else on this PC (:5017).
"""
import json
import os
import re
import threading

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
TALK_JSON = os.path.join(HERE, "..", "..", "RIFT", "Fleet", "talk_bf.json")
WX_ASK = 17                         # "wx?"
TONES = {"+": 1568, "-": 784, ">": 1319, "<": 1047, "[": 880, "]": 1175, ".": 2093, ",": 523}
DREAM_PITCH = 0.6                   # her voice is deep and slow (like her sensor board's)
AURA_PORT = 5017

WEATHER_WORDS = ("weather", "outside", "rain", "forecast", "umbrella", "moon", "sunrise", "sunset",
                 "humid", "snow", "storm", "windy", "how cold", "how warm", "sunny", "cloudy", "aura",
                 "barometer", "air pressure", "gust", "stargaz", "see the stars", "thunder", "lightning",
                 "sunscreen", "frost", "fog", "dew point", "muggy", "water the garden", "water the plants",
                 "daylight", "days getting", "uv index")
READING_WORDS = ("temp", "warm", "hot", "cold", "humid", "reading")
ROBOTS = ("NORA", "MILA", "NIKA", "WHIP", "KIDA00", "KIDA01", "KIDA-00", "KIDA-01", "ARIA", "RITA", "IDA", "ARM")


def is_aura_question(text):
    """The weather, the sky, or a robot's temperature / humidity. (Her own CPU temperature stays
    "system stats": "temperature" alone isn't the weather.)"""
    low = (text or "").lower()
    if any(w in low for w in WEATHER_WORDS):
        return True
    flat = re.sub(r"[^a-z0-9]", "", low)
    return any(re.sub(r"[^a-z0-9]", "", r.lower()) in flat for r in ROBOTS) and any(w in low for w in READING_WORDS)


def find_aura(rift_url="http://127.0.0.1:5000"):
    """AURA's base URL: from RIFT's fleet list, else this PC."""
    try:
        for r in requests.get(rift_url.rstrip("/") + "/robots", timeout=2).json().get("robots", []):
            if str(r.get("name", "")).upper() == "AURA" and r.get("ip"):
                port = next((c.split(":", 1)[1] for c in r.get("capabilities", []) if str(c).startswith("ask:")), AURA_PORT)
                return f"http://{r['ip']}:{port}"
    except (requests.RequestException, ValueError):
        pass
    return f"http://127.0.0.1:{AURA_PORT}"


def beep(u, pitch=DREAM_PITCH):
    """Utterance u's Brainfuck program, one tone per symbol (Windows' speaker; elsewhere silent)."""
    try:
        with open(TALK_JSON, encoding="utf-8") as f:
            code = next(x["bf"] for x in json.load(f) if x["index"] == u)
    except (OSError, ValueError, StopIteration, KeyError):
        return False
    if os.name != "nt":
        return False
    import winsound
    for ch in code:
        winsound.Beep(max(37, int(TONES.get(ch, 1000) * pitch)), 80 if ch == "." else 28)
    return True


def ask_aura(question, rift_url="http://127.0.0.1:5000", chirp=True):
    """{"text": what DREAM says, "bf": AURA's Brainfuck word or None, "ok": reached her}."""
    base = find_aura(rift_url)
    if chirp:
        beep(WX_ASK)                                  # "wx?", in Brainfuck, before she asks in words
    try:
        a = requests.get(base + "/ask", params={"q": question, "chirp": "1" if chirp else "0"}, timeout=30).json()
    except (requests.RequestException, ValueError):
        return {"ok": False, "bf": None,
                "text": "I can't reach AURA, the weather station. Is she running?"}
    bf = (a.get("bf") or {}).get("text")
    return {"ok": True, "bf": bf, "text": f"AURA says: {a.get('answer', '')}".strip()}


def ask_aura_async(question, done, **kw):
    """The same, in a thread: done(result) when AURA has answered."""
    threading.Thread(target=lambda: done(ask_aura(question, **kw)), daemon=True).start()
