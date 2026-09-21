"""
dream_py_test.py - run the REAL dream.py (its real main() and voice_loop()) with only the
hardware stubbed: microphone, speakers, Whisper and the screen. Everything else - the mind
hooks, ask_llm, the sensor handler, the sleep wiring - is the code that ships.

    cd scripts
    ..\\venv311\\Scripts\\python dream_mind\\dream_py_test.py

Needs Ollama running (voice_loop checks for it). Uses a scratch memory folder and a fake
language model, so it touches none of your memories. Takes a minute or two.
"""
import os, sys, tempfile, threading, time, wave
from pathlib import Path
os.environ["SDL_VIDEODRIVER"] = "dummy"; os.environ["SDL_AUDIODRIVER"] = "dummy"
SCRIPTS = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, SCRIPTS); os.chdir(SCRIPTS)
import numpy as np

tmp = tempfile.mkdtemp(prefix="dreampy_")
from dream_mind import store
store.set_dir(tmp)                                   # her mind lives in a scratch folder
import dream_memory                                  # ...and so do the older fact files
dream_memory.MEMORIES_PATH = os.path.join(tmp, "facts.txt"); dream_memory.MILESTONES_PATH = os.path.join(tmp, "miles.txt")

t0 = time.time()
import dream as D
print(f"[harness] real dream.py imported in {time.time()-t0:.0f}s | MIND loaded: {D.MIND is not None}", flush=True)
assert D.MIND is not None

results, spoken, prompts = {}, [], []

def fake_generate(prompt, **kw):
    prompts.append((prompt, kw)); return "Well, hello Sam. A grandfather who teaches guitar sounds like a keeper."
D.mind_llm.generate = fake_generate

D.speak = lambda text: spoken.append(text)
D.LIPSYNC_ENABLED = False
D._init_lipsync_cache = lambda: None
D.get_whisper = lambda: None
D.check_audio_levels = lambda fp=None: True

def fake_record(path, seconds):                      # a "recording" with a real voice-like signal, so the cue code runs
    sr = 16000; t = np.arange(sr * 2) / sr
    x = (np.sin(2 * np.pi * 180 * t * (1 + 0.08 * np.sin(2 * np.pi * 3 * t))) * 0.25 * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
    return True
D._record_clip = fake_record

script = iter([
    "you're so stupid and useless",
    "my name is Sam and I love my old guitar because my grandfather taught me",
    "how are you feeling?",
    "turn off the alarm",
    "goodbye",
])
D.transcribe_file = lambda fp: next(script)

wakes = {"n": 0}
def fake_wake():
    wakes["n"] += 1
    if wakes["n"] > 5:
        D._state["running"] = False; return None
    return "__WAKE__"
D.listen_for_wake_word = fake_wake

# stand in for the display: it clears the intro clip when it has "played", then waits for the conversation to finish
def fake_display():
    deadline = time.time() + 240
    while D._state["running"] and time.time() < deadline:
        if D._state["force_video"]: D._state["force_video"] = None
        time.sleep(0.2)
    results["conversation_done"] = not D._state["running"]
    # sleep wiring: the mind should notice the host going to sleep, and waking
    D.enter_sleep()
    D._state["running"] = True                        # keep the mind running for this part
    end = time.time() + 12
    while time.time() < end and D.MIND._sleep_thread is None: time.sleep(0.3)
    results["mind_began_sleep"] = D.MIND._sleep_thread is not None
    D.exit_sleep()
    end = time.time() + 12
    while time.time() < end and D.MIND._sleep_thread is not None: time.sleep(0.3)
    results["mind_woke"] = D.MIND._sleep_thread is None
    D._state["running"] = False
D.run_display = fake_display

D._handle_sensor_line("PRESENT"); results["sensor_present"] = D.MIND._present
D.MIND.mood.valence = 0.0   # a known start

print("[harness] running the real main() ...", flush=True)
D.main()

print("\n=== spoken by DREAM ===")
for i, s in enumerate(spoken): print(f"  {i}: {s[:110]}")
print("\n=== checks ===")
def check(ok, what): print(("PASS  " if ok else "FAIL  ") + what, flush=True); return ok
ok = True
ok &= check(results.get("sensor_present") is True, "_handle_sensor_line('PRESENT') reaches the mind")
ok &= check(any("stung" in s or "hurt" in s for s in spoken), "insult -> her own boundary reply (no language model)")
gen_prompts = [p for p, _ in prompts if "Sam" in p or "grandfather" in p]
ok &= check(len(gen_prompts) == 1, f"language model called once, only for the real conversation (calls: {len(prompts)})")
ok &= check(bool(gen_prompts) and "You feel" in gen_prompts[0] and "colour your tone" in gen_prompts[0], "...and its prompt carried her inner state")
ok &= check(bool(prompts) and prompts[-1][1].get("num_predict") == D.MIND.token_budget(150), f"...with the tiredness-scaled length ({prompts[-1][1].get('num_predict') if prompts else None})")
ok &= check(any("Sam" in s and "keeper" in s for s in spoken), "her reply was spoken")
ok &= check(any(s.startswith("I feel") for s in spoken), "'how are you feeling?' answered from her own state")
ok &= check(any("alarm board" in s for s in spoken), "the alarm command still takes the original hub path (board unreachable here)")
ok &= check("Goodbye." in spoken, "goodbye still ends the session as before")
ok &= check(results.get("conversation_done") is True, "the voice loop ran to completion")
eps = [e["user"] for e in D.MIND.memory.active()]
ok &= check(len(eps) == 1 and "guitar" in eps[0], f"only the real conversation became a memory ({len(eps)} stored)")
facts = open(dream_memory.MEMORIES_PATH).read() if os.path.exists(dream_memory.MEMORIES_PATH) else ""
ok &= check("Sam" in facts, "the older fact system still learned the name")
ok &= check(D.MIND.conversations == 3, f"interactions counted: the insult, the story and the feelings question, not the alarm or goodbye commands ({D.MIND.conversations})")
ok &= check(results.get("mind_began_sleep") is True, "dream.py's enter_sleep() makes the mind start sleeping")
ok &= check(results.get("mind_woke") is True, "...and exit_sleep() wakes her")
ok &= check(os.path.exists(os.path.join(tmp, "mind_state.json")), "the mind saved its state on shutdown")
print("\nRESULT:", "ALL PASSED" if ok else "SOME FAILED", flush=True)
os._exit(0)
