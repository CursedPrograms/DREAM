"""scripts/dream_fleet.py: what DREAM understands, telling and asking robots through RIFT, and
handing work to TINA (only her final answer is said; her reasoning is a thought).
Offline: a fake RIFT (with RIFT's safety rules) and a fake TINA on localhost; no robots.
Run from the DREAM folder:  venv311\\Scripts\\python -m pytest tests"""
import os
import sys
import threading
import time

import pytest
from flask import Flask, jsonify, request
from werkzeug.serving import make_server

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import dream_fleet as f  # noqa: E402

HARMLESS = {"stop", "music_stop", "music_play", "music_next", "uv_on", "uv_off", "uv_blink", "lights_on", "lights_off",
            "lock_wheels", "scan"}           # as RIFT's relay.HARMLESS
NORA_DOES = ["back", "forward", "left", "lock_wheels", "music_next", "music_play", "music_prev", "music_stop",
             "right", "stop", "turn_left", "turn_right", "uv_blink", "uv_off", "uv_on"]   # RIFT's NORA mapping: no "lights"


class World:
    def __init__(self):
        self.talk, self.tasks, self.online = [], {}, {"NORA"}


@pytest.fixture(scope="module")
def world():
    w = World()
    app = Flask("fake_rift_tina")

    @app.route("/fleet")
    def fleet():
        return jsonify({"robots": [{"name": n} for n in sorted(w.online)]})

    @app.route("/talk", methods=["POST"])
    def talk():
        d = request.get_json()
        w.talk.append(d)
        if d["to"] not in w.online:
            return jsonify({"ok": False, "error": f"{d['to']} isn't online (not on RIFT's roster)"}), 404
        if d["verb"] == "ask":
            return jsonify({"ok": True, "status": {"F": 34, "L": 120, "B": -1, "R": 80, "mode": 0, "uv": 1, "temp": 23.4}})
        a = d["action"]
        if "unlock" in a:
            return jsonify({"ok": False, "tier": "never", "error": "never relayed"}), 403
        if a not in HARMLESS and d.get("confirmed") is not True:
            return jsonify({"ok": False, "needs": "confirmed"}), 403
        if a not in NORA_DOES:
            return jsonify({"ok": False, "does": NORA_DOES, "error": f"NORA can't {a}"}), 400
        return jsonify({"ok": True, "action": a})

    @app.route("/task", methods=["POST"])
    def task():
        tid = str(len(w.tasks) + 1)
        t = w.tasks[tid] = {"text": request.get_json()["task"], "from": request.get_json().get("from"),
                            "events": [], "status": "running", "question": None, "answer": None}

        def work():
            t["events"] += [{"kind": "think", "data": "I should search the repo first."},
                            {"kind": "tool", "data": {"name": "search_files", "args": {"pattern": "TODO"}}}]
            if "ask me" in t["text"]:
                t["question"] = "This moves NORA forward. Go ahead?"
                while t["answer"] is None:
                    time.sleep(0.02)
                time.sleep(0.2)                      # like TINA: the question clears a moment later
                t["question"] = None
            t["result"] = {"answer": "NORA's TODO has **three** items:\n```\ncode\n```\nThe first is the wheel lock."}
            t["status"] = "done"
        threading.Thread(target=work, daemon=True).start()
        return jsonify({"id": tid})

    @app.route("/task/<tid>")
    def get_task(tid):
        t, since = w.tasks[tid], int(request.args.get("since", 0))
        return jsonify({"status": t["status"], "events": t["events"][since:], "next": len(t["events"]),
                        "question": t["question"], "result": t.get("result")})

    @app.route("/task/<tid>/answer", methods=["POST"])
    def answer(tid):
        w.tasks[tid]["answer"] = bool(request.get_json().get("yes"))
        return jsonify({"ok": True})

    srv = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    w.url = f"http://127.0.0.1:{srv.server_port}"
    yield w
    srv.shutdown()


@pytest.fixture
def dream(world, monkeypatch):
    monkeypatch.setattr(f, "RIFT", world.url)
    monkeypatch.setattr(f, "TINA", world.url)
    monkeypatch.setattr(f, "_pending", None)
    monkeypatch.setattr(f, "_roster", {"t": 0.0, "names": []})
    world.talk.clear()
    world.online = {"NORA"}
    thoughts = []

    def say(line):
        thoughts.clear()
        return f.handle(line, on_thought=thoughts.append)
    say.thoughts = thoughts
    return say


# ---------------------------------------------------------------- understanding
@pytest.mark.parametrize("line,want", [
    ("Tell NORA to stop the music.", ("do", "NORA", "music_stop")),
    ("tell nora to play some music", ("do", "NORA", "music_play")),
    ("tell nora to skip", ("do", "NORA", "music_next")),
    ("tell kida zero one to turn left for two seconds", ("do", "KIDA01", "turn_left")),
    ("tell NORA to turn the lights on", ("do", "NORA", "lights_on")),
    ("tell NORA to unlock her wheels", ("do", "NORA", "unlock")),
    ("ask MILA how she is", ("ask", "MILA", None)),
    ("How is NORA doing?", ("ask", "NORA", None)),
    ("what's NORA's status", ("ask", "NORA", None)),
    ("Hey TINA, what's in NORA's TODO?", ("tina", None, None)),
    ("tell NORA to dance", ("tina", None, None)),
    ("why does KIDA's script exit", ("tina", None, None)),
    ("fix the bug in NORA's sketch", ("tina", None, None)),
    ("tell me a story", None), ("how are you doing", None), ("I like the music", None),
])
def test_understand(line, want, monkeypatch):
    monkeypatch.setattr(f, "_roster", {"t": 1e18, "names": ["NORA", "KIDA01", "MILA"]})
    got = f.understand(line)
    if want is None:
        assert got is None
    else:
        assert (got["verb"], got.get("to"), got.get("action")) == want


def test_seconds():
    assert f.understand("tell NORA to go forward for 2 seconds")["seconds"] == 2.0
    assert f.seconds_in("go forward for two seconds") == 2 and f.seconds_in("half a second") == 0.5


# ---------------------------------------------------------------- telling and asking
def test_tell(dream, world):
    r = dream("Tell NORA to stop the music.")
    assert r.say == "Done. NORA stopped the music." and world.talk[-1]["from"] == "DREAM"


def test_ask(dream):
    r = dream("ask NORA how she is")
    assert "NORA is online" in r.say and "front 34" in r.say and "UV light on" in r.say


def test_lights_fall_back_to_uv(dream, world):
    r = dream("tell NORA to turn the lights on")
    assert [t["action"] for t in world.talk] == ["lights_on", "uv_on"] and "UV light on" in r.say


def test_moving_needs_a_yes(dream, world):
    r = dream("tell NORA to go back for one second")
    assert r.awaiting and "back for 1 second" in r.say and len(world.talk) == 1
    r = dream("yes")
    assert r.say == "Done. NORA backed up." and world.talk[-1]["confirmed"] is True


def test_no_means_no(dream, world):
    dream("tell NORA to go forward")
    assert dream("no").say == "Okay, I won't." and len(world.talk) == 1


def test_new_command_isnt_a_no(dream, world):
    dream("tell NORA to go forward")
    r = dream("tell NORA to stop the music")          # has "stop" in it, but it's a new command
    assert r.say.startswith("Done") and world.talk[-1]["action"] == "music_stop"
    assert not any(t.get("confirmed") for t in world.talk)


def test_never_and_offline(dream):
    assert dream("tell NORA to unlock her wheels").say.startswith("No.")
    assert dream("tell MILA to stop the music").say == "MILA isn't online right now."


def test_ordinary_chat_is_left_alone(dream, world):
    assert dream("tell me a story") is None and world.talk == []


def test_rift_down(dream, monkeypatch):
    monkeypatch.setattr(f, "RIFT", "http://127.0.0.1:9")
    assert "can't reach RIFT" in dream("tell NORA to stop the music").say


# ---------------------------------------------------------------- TINA
def test_tina_final_answer_only(dream, world):
    r = dream("Hey TINA, what's in NORA's TODO?")
    assert r.say.startswith("NORA's TODO has three items") and "**" not in r.say and "code is on TINA's page" in r.say
    assert "search the repo" not in r.say
    assert any(t.startswith("TINA thinks") for t in dream.thoughts)
    assert world.tasks[str(len(world.tasks))]["from"] == "DREAM"     # her own conversation with TINA


def test_tina_question_passed_on(dream, world):
    r = dream("ask TINA to drive NORA forward and ask me first")
    assert r.awaiting and r.say.startswith("TINA asks")
    r = dream("yes")
    assert r.say.startswith("NORA's TODO") and world.tasks[str(len(world.tasks))]["answer"] is True


def test_tina_down(dream, monkeypatch):
    monkeypatch.setattr(f, "TINA", "http://127.0.0.1:9")
    assert "TINA isn't running" in dream("ask TINA to check the fleet").say
