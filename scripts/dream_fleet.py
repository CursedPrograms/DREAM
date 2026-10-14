"""
dream_fleet.py - DREAM talks to the fleet: tells and asks robots through RIFT,
and hands TINA's kind of work to TINA. Shared by scripts/dream.py (the desktop
app) and app.py (the web version).

    "tell NORA to stop the music"       -> RIFT  POST /talk {"to", "verb": "do", "action": "music_stop"}
    "ask MILA how she is"               -> RIFT  POST /talk {"to", "verb": "ask"}
    "ask TINA why KIDA's Hailo is off"  -> TINA  POST /task, then GET /task/<id> until she's done
    "fix the bug in NORA's sketch"      -> TINA too: code, files, repos and tools are her job

Only the final answer is ever returned for speaking (Reply.say). Everything else -
TINA's reasoning, her tool calls, RIFT's raw answers - goes to on_thought(text),
which the app shows on screen when "Show thoughts" is on. It never goes to Piper.

Anything that can move a robot needs a person's yes: RIFT answers 403 "needs
confirmed", DREAM asks, and the next thing you say ("yes" / "no") decides
(Reply.awaiting: listen again without the wake word). TINA's own questions
("this moves NORA, go ahead?") are passed on the same way. RIFT never relays
anything that unlocks a robot, whatever anyone says.

    python scripts/dream_fleet.py "tell NORA to stop the music"     (try one line)
"""

import json
import os
import re
import time

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

KNOWN_ROBOTS = ["NORA", "MILA", "WHIP", "IDA", "ARM", "KIDA00", "KIDA01", "NINA"]
PENDING_S = 60          # a question waits this long for "yes" or "no"
TINA_TIMEOUT_S = 600    # the longest DREAM waits for TINA's answer
SAY_MAX = 600           # TINA's answer is cut here for speaking; the rest is on her page

YES = re.compile(r"\b(yes|yeah|yep|yup|sure|go ahead|do it|okay|ok|please do|affirmative)\b", re.I)
NO = re.compile(r"\b(no|nope|don'?t|do not|cancel|stop|never ?mind|negative)\b", re.I)


def _cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            c = json.load(f)["Config"]
    except (OSError, ValueError, KeyError):
        return {}, {}
    return c.get("DREAM", {}), c.get("Dashboard", {})


_dream_cfg, _dash_cfg = _cfg()
RIFT = f"http://{_dash_cfg.get('RiftHost', 'localhost')}:{_dash_cfg.get('RiftPort', 5000)}"
TINA = str(_dream_cfg.get("TinaUrl", "http://127.0.0.1:5013")).rstrip("/")
HAND_TO_TINA = bool(_dream_cfg.get("HandToTina", True))


class Reply:
    """say: what DREAM says out loud. awaiting: she asked a yes/no question."""
    def __init__(self, say, awaiting=False):
        self.say = say
        self.awaiting = awaiting


_pending = None     # {"kind": "robot" | "tina", "t", ...} - a question waiting for yes/no


def _noop(_text):
    pass


# ------------------------------------------------------------------ understanding
# action -> phrases. Order matters: the music ones come before a bare "stop".
ACTIONS = [
    ("unlock",      r"\bunlock\b|\bpassword\b|\bpin\b"),          # RIFT refuses it; DREAM says so
    ("music_stop",  r"\b(stop|pause|kill|end|turn off|switch off|cut)\b.*\b(music|song|tune|track|playing)\b"
                    r"|\b(music|song)\b.*\b(off|stop)\b|\bbe quiet\b|\bsilence\b|\bshut up\b"),
    ("music_next",  r"\bnext (song|track|tune)\b|\bskip\b"),
    ("music_prev",  r"\b(previous|last|prior) (song|track|tune)\b|\bgo back a song\b"),
    ("music_play",  r"\b(play|start|put on)\b.*\b(music|song|tune|track|something)\b|\bplay\b"),
    ("uv_blink",    r"\bblink"),
    ("uv_off",      r"\b(uv|ultraviolet)\b.*\boff\b|\boff\b.*\b(uv|ultraviolet)\b"),
    ("uv_on",       r"\b(uv|ultraviolet)\b"),
    ("lights_off",  r"\blights?\b.*\boff\b|\boff\b.*\blights?\b"),
    ("lights_on",   r"\blights?\b"),
    ("lock_wheels", r"\block\b.*\bwheels?\b|\bwheels?\b.*\block"),
    ("scan",        r"\bscan\b|\blook around\b"),
    ("turn_left",   r"\b(turn|spin|rotate)\b.*\bleft\b"),
    ("turn_right",  r"\b(turn|spin|rotate)\b.*\bright\b"),
    ("left",        r"\bleft\b"),
    ("right",       r"\bright\b"),
    ("back",        r"\b(back|backward|backwards|reverse)\b"),
    ("forward",     r"\b(forward|forwards|ahead|straight)\b"),
    ("stop",        r"\b(stop|halt|freeze|stand still|brake)\b"),
]
ALTERNATIVES = {"lights_on": ["uv_on"], "lights_off": ["uv_off"], "uv_on": ["lights_on"], "uv_off": ["lights_off"]}
DONE = {
    "music_stop": "stopped the music", "music_play": "is playing music", "music_next": "skipped to the next song",
    "music_prev": "went back a song", "uv_on": "switched her UV light on", "uv_off": "switched her UV light off",
    "uv_blink": "is blinking her UV light", "lights_on": "switched her lights on", "lights_off": "switched her lights off",
    "lock_wheels": "locked her wheels", "stop": "stopped", "scan": "had a look around",
    "forward": "drove forward", "back": "backed up", "left": "went left", "right": "went right",
    "turn_left": "turned left", "turn_right": "turned right",
}
NUMBERS = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "half a": 0.5, "a half": 0.5}

TELL_RE = re.compile(r"\b(?:tell|ask|have|get|make|order|command)\s+(?P<who>[a-z][a-z0-9 -]{1,14}?)\s+to\s+(?P<what>.+)", re.I)
ASK_RE = re.compile(r"\b(?:ask\s+(?P<who1>[a-z][a-z0-9 -]{1,14}?)\s+(?:how|what|whether|if|about)\b"
                    r"|how(?:'s| is| are)\s+(?P<who2>[a-z][a-z0-9 -]{1,14}?)(?:\s+(?:doing|feeling|going|today|now)\b|$)"
                    r"|(?:what(?:'s| is)|check)\s+(?P<who3>[a-z][a-z0-9 -]{1,14}?)(?:'s|s)?\s+(?:status|state)\b)", re.I)
TINA_RE = re.compile(r"^\s*(?:hey\s+)?tina\b[,:]?|\b(?:ask|tell|get|have)\s+tina\b(?:\s+to)?|\bhand (?:it|this) to tina\b", re.I)
TASK_RE = re.compile(
    r"\b(fix|debug|check|look at|look into|read|search|find|grep|compile|flash|upload|edit|change|update|write|"
    r"review|diagnose|investigate|explain)\b.*\b(code|repo|repository|sketch|firmware|file|files|script|bug|"
    r"error|function|readme|todo|config|log|logs|\.py|\.ino|\.js|arduino|esp32|github)\b"
    r"|\bwhy (?:isn'?t|is not|won'?t|doesn'?t|does not|can'?t)\b.*\b(working|responding|answering|connecting|moving|"
    r"compiling|starting|running)\b"
    r"|\bwhy (?:does|did|do|is|are|was|were)\b.*\b(crash|crashes|crashing|crashed|exit|exits|exited|fail|fails|failing|"
    r"failed|hang|hangs|hanging|freeze|freezes|froze|reboot|reboots|rebooting|rebooted|disconnect|disconnects)\b", re.I)


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()).replace("zero", "0").replace("one", "1")


_roster = {"t": 0.0, "names": []}


def roster():
    """Robot names RIFT knows right now (cached 30 s), plus the fleet's usual ones."""
    if time.time() - _roster["t"] > 30:
        names = []
        try:
            r = requests.get(f"{RIFT}/fleet", timeout=2)
            names = [rb.get("name") for rb in r.json().get("robots", []) if rb.get("name")]
        except (requests.RequestException, ValueError, AttributeError):
            pass
        _roster.update(t=time.time(), names=names)
    return _roster["names"] + [n for n in KNOWN_ROBOTS if n not in _roster["names"]]


def robot_named(words):
    """'nora' -> 'NORA', 'kida zero one' -> 'KIDA01', 'kida' -> the first KIDA. None if it isn't a robot."""
    if not words:
        return None
    key = _norm(words)
    key = {"kida1": "kida01", "kida0": "kida00", "norah": "nora", "mila": "mila", "milla": "mila"}.get(key, key)
    names = roster()
    for n in names:
        if _norm(n) == key:
            return n
    if key == "kida":
        return next((n for n in names if n.upper().startswith("KIDA")), None)
    return None


def action_for(what):
    for action, pattern in ACTIONS:
        if re.search(pattern, what, re.I):
            return action
    return None


def seconds_in(what):
    m = re.search(r"\b(\d+(?:\.\d+)?|one|two|three|a half|half a|an?)\s+(?:sec|secs|second|seconds)\b", what, re.I)
    if not m:
        return None
    v = m.group(1).lower()
    return float(v) if v[0].isdigit() else NUMBERS.get(v)


def understand(text):
    """What a line means for the fleet: {"verb": "do"|"ask", "to", ...}, {"verb": "tina", "task"}, or None."""
    t = text.strip().rstrip(".!?")
    if TINA_RE.search(t):
        task = TINA_RE.sub("", t, count=1).strip(" ,:") or t
        return {"verb": "tina", "task": task}
    m = TELL_RE.search(t)
    if m:
        who = robot_named(m.group("who"))
        if who:
            what = m.group("what")
            action = action_for(what)
            if action:
                return {"verb": "do", "to": who, "action": action, "seconds": seconds_in(what), "said": what}
            if re.search(r"\b(how|what|status|state|doing)\b", what, re.I) and t.lower().startswith("ask"):
                return {"verb": "ask", "to": who}
            return {"verb": "tina", "task": t}     # a robot job DREAM can't map: TINA has tell_robot and the code
    m = ASK_RE.search(t)
    if m:
        who = robot_named(m.group("who1") or m.group("who2") or m.group("who3"))
        if who:
            return {"verb": "ask", "to": who}
    if HAND_TO_TINA and TASK_RE.search(t):
        return {"verb": "tina", "task": t}
    return None


# ------------------------------------------------------------------ RIFT
def talk(body):
    """POST RIFT's /talk. (status code, answer dict); code 0 = RIFT isn't answering."""
    try:
        r = requests.post(f"{RIFT}/talk", json=dict(body, **{"from": "DREAM"}), timeout=15)
    except requests.RequestException as e:
        return 0, {"ok": False, "error": f"{type(e).__name__}"}
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"ok": False, "error": r.text[:200]}


MODES = ["manual", "auto", "line following", "remote"]


def describe_status(name, status):
    """A robot's /status as one or two spoken sentences (the full answer goes to thoughts)."""
    if not isinstance(status, dict):
        return f"{name} answered: {str(status)[:120]}"
    bits = []
    mode = status.get("mode")
    if mode is not None:
        bits.append(f"in {MODES[mode] if isinstance(mode, int) and 0 <= mode < len(MODES) else mode} mode")
    if status.get("moving") is not None:
        bits.append("moving" if status["moving"] else "standing still")
    if status.get("lock"):
        bits.append("wheels locked")
    if status.get("uv"):
        bits.append("UV light on")
    if status.get("battery") is not None:
        bits.append(f"battery {status['battery']}")
    if isinstance(status.get("temp"), (int, float)) and status["temp"] > -50:
        bits.append(f"{status['temp']:.0f} degrees")
    sides = [(s, status.get(k)) for s, k in (("front", "F"), ("left", "L"), ("back", "B"), ("right", "R"))]
    seen = [f"{s} {v:.0f}" for s, v in sides if isinstance(v, (int, float)) and v >= 0]
    if seen:
        bits.append("distances " + ", ".join(seen) + " centimetres")
    if not bits:   # a robot DREAM doesn't know yet: its first few plain values
        bits = [f"{k} {v}" for k, v in status.items() if isinstance(v, (str, int, float, bool))][:4]
    return f"{name} is online" + (": " + ", ".join(bits) if bits else "") + "."


def _human(action):
    return action.replace("_", " ")


def robot(cmd, on_thought=_noop, confirmed=False):
    """Carries out {"verb": "do"|"ask", "to", ...} through RIFT. Returns a Reply."""
    global _pending
    name = cmd["to"]
    if cmd["verb"] == "ask":
        on_thought(f"Asking {name} through RIFT.")
        code, out = talk({"to": name, "verb": "ask"})
        on_thought(f"RIFT ({code}): {json.dumps(out)[:600]}")
        if code == 0:
            return Reply("I can't reach RIFT, so I can't talk to the robots right now.")
        if not out.get("ok"):
            return Reply(f"{name} isn't online right now." if code == 404 else f"{name} didn't answer.")
        return Reply(describe_status(name, out.get("status")))

    action = cmd["action"]
    body = {"to": name, "verb": "do", "action": action}
    if cmd.get("seconds"):
        body["seconds"] = cmd["seconds"]
    if confirmed:
        body["confirmed"] = True
    on_thought(f"Telling {name} to {_human(action)} through RIFT" + (" (you said yes)." if confirmed else "."))
    code, out = talk(body)
    on_thought(f"RIFT ({code}): {json.dumps(out)[:600]}")
    if code == 400 and out.get("does"):     # "lights" may be her UV light, and the other way round
        alt = next((a for a in ALTERNATIVES.get(action, []) if a in out["does"]), None)
        if alt:
            return robot(dict(cmd, action=alt), on_thought, confirmed)
    if code == 0:
        return Reply("I can't reach RIFT, so I can't talk to the robots right now.")
    if code == 403 and out.get("needs") == "confirmed":
        _pending = {"kind": "robot", "t": time.time(), "cmd": cmd}
        s = cmd.get("seconds")
        how_long = f" for {s:g} second{'' if s == 1 else 's'}" if s else ""
        return Reply(f"That would move {name}: {_human(action)}{how_long}. Should I? Say yes or no.", awaiting=True)
    if code == 403:
        return Reply(f"No. Nobody can tell {name} to {_human(action)} through RIFT. That's for a person standing at the robot.")
    if code == 404:
        return Reply(f"{name} isn't online right now.")
    if code == 400 and "does" in out:
        can = ", ".join(_human(a) for a in out["does"]) or "nothing yet"
        return Reply(f"{name} can't {_human(action)}. She can: {can}.")
    if not out.get("ok"):
        return Reply(f"{name} didn't do it. " + (f"RIFT says: {out.get('error')}" if out.get("error") else "She didn't answer."))
    return Reply(f"Done. {name} {DONE.get(action, 'did it: ' + _human(action))}.")


# ------------------------------------------------------------------ TINA
def _speakable(text):
    """TINA's answer as something to say: no code, no markdown marks, not too long."""
    text = re.sub(r"```.*?```", " (the code is on TINA's page) ", text or "", flags=re.S)
    text = re.sub(r"[`*_#>|]", "", text)
    text = " ".join(text.split())
    if len(text) > SAY_MAX:
        cut = text[:SAY_MAX]
        text = cut[:cut.rfind(". ") + 1 or SAY_MAX] + " The rest is on TINA's page."
    return text


def _thought_from(event):
    kind, data = event.get("kind"), event.get("data")
    if kind == "think":
        return f"TINA thinks: {data}"
    if kind == "say":
        return f"TINA: {data}"
    if kind == "tool":
        args = json.dumps(data.get("args", {}))[:200] if isinstance(data, dict) else ""
        return f"TINA uses {data.get('name') if isinstance(data, dict) else data} {args}"
    if kind == "result":
        return f"{data.get('name', 'result')}: {str(data.get('text', ''))[:300]}" if isinstance(data, dict) else str(data)[:300]
    if kind == "error":
        return f"TINA hit an error: {data}"
    return None


def tina(task, on_thought=_noop):
    """Hands a task to TINA and waits for her final answer. Returns a Reply.
    "from": DREAM keeps DREAM's own conversation with her, so "Hey TINA, and the other one?"
    follows on from what DREAM asked her last (and not from what someone typed on TINA's page)."""
    on_thought(f"Handing this to TINA: {task}")
    try:
        r = requests.post(f"{TINA}/task", json={"task": task, "voice": False, "from": "DREAM"}, timeout=5)
        tid = r.json()["id"]
    except (requests.RequestException, ValueError, KeyError) as e:
        on_thought(f"TINA isn't answering ({type(e).__name__}).")
        return Reply("TINA isn't running, so I can't hand that to her. Start TINA and ask me again.")
    return _follow(tid, 0, on_thought, time.time() + TINA_TIMEOUT_S)


def _follow(tid, since, on_thought, deadline, answered=None):
    global _pending
    grace = time.time() + 5     # the question just answered may still show for a moment
    while time.time() < deadline:
        try:
            d = requests.get(f"{TINA}/task/{tid}", params={"since": since}, timeout=5).json()
        except (requests.RequestException, ValueError):
            time.sleep(1)
            continue
        for e in d.get("events", []):
            text = _thought_from(e)
            if text:
                on_thought(text)
        since = d.get("next", since)
        if d.get("status") == "done":
            answer = (d.get("result") or {}).get("answer", "")
            on_thought(f"TINA's answer: {answer}")
            return Reply(_speakable(answer) or "TINA finished, but she didn't say anything.")
        if d.get("status") == "failed":
            return Reply("TINA ran into a problem with that one. It's on her page.")
        if d.get("question") and not (d["question"] == answered and time.time() < grace):
            _pending = {"kind": "tina", "t": time.time(), "id": tid, "since": since, "deadline": deadline,
                        "question": d["question"]}
            return Reply(f"TINA asks: {_speakable(d['question'])} Yes or no?", awaiting=True)
        time.sleep(1)
    return Reply("TINA is still working on it. Her answer will be on her page.")


# ------------------------------------------------------------------ one line in
def _answer_tina(p, yes, on_thought):
    """Gives TINA your yes/no and waits for her answer (her question takes a moment to clear)."""
    try:
        requests.post(f"{TINA}/task/{p['id']}/answer", json={"yes": yes}, timeout=5)
    except requests.RequestException:
        return Reply("I lost TINA, so I couldn't give her your answer.")
    on_thought(f"Told TINA: {'yes' if yes else 'no'}.")
    return _follow(p["id"], p["since"], on_thought, p["deadline"], answered=p["question"])


def handle(text, on_thought=_noop):
    """DREAM's fleet turn: a Reply if this line was for the fleet (or answers her
    question), None if it's ordinary conversation for her own model."""
    global _pending
    p, _pending = _pending, None
    cmd = understand(text)
    if p and time.time() - p["t"] < PENDING_S:
        yes, no = bool(YES.search(text)), bool(NO.search(text))
        if cmd is None and yes and not no:
            if p["kind"] == "robot":
                return robot(p["cmd"], on_thought, confirmed=True)
            return _answer_tina(p, True, on_thought)
        if cmd is None and no:
            return _answer_tina(p, False, on_thought) if p["kind"] == "tina" else Reply("Okay, I won't.")
        if p["kind"] == "tina":     # something else entirely: TINA would wait forever, so it's a no
            try:
                requests.post(f"{TINA}/task/{p['id']}/answer", json={"yes": False}, timeout=5)
            except requests.RequestException:
                pass
            on_thought("You moved on, so I told TINA no.")
    if cmd is None:
        return None
    if cmd["verb"] == "tina":
        return tina(cmd["task"], on_thought)
    return robot(cmd, on_thought)

if __name__ == "__main__":
    import sys
    line = " ".join(sys.argv[1:]) or "tell NORA to stop the music"
    print("understood:", understand(line))
    reply = handle(line, on_thought=lambda t: print("  (thought)", t))
    print("DREAM says:", reply.say if reply else "(not for the fleet: her own model answers)")
