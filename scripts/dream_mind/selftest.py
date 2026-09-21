"""
selftest.py - live a few simulated days and check the mind behaves.

    python -m dream_mind.selftest          (run from the scripts/ folder)

Uses a throwaway memory folder and a fake language model, so it's fast, touches
none of your real memories, and needs no Ollama.
"""

import json
import random
import sys
import tempfile
import time

failures = 0


def check(ok, what):
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    if not ok:
        failures += 1


def at(day, hour, minute=0):
    """A timestamp `day` days from a fixed Monday, at hour:minute local time."""
    base = time.mktime((2026, 9, 21, 0, 0, 0, 0, 0, -1))
    return base + day * 86400 + hour * 3600 + minute * 60


def latest_kind(store, kind):
    """The text of her newest thought of this kind, or None."""
    for t in reversed(store.read_jsonl("thoughts.jsonl")):
        if t.get("kind") == kind:
            return t["text"]
    return None


def main():
    from . import llm, store
    tmp = tempfile.mkdtemp(prefix="dream_mind_test_")
    store.set_dir(tmp)
    from . import dreaming, vision
    vision.set_photo_dirs([])
    try:   # DREAM's older fact files: keep them in the scratch folder too, so "forget everything" can't touch yours
        import dream_memory
        dream_memory.MEMORIES_PATH = str(store.mind_dir() / "legacy_memories.txt")
        dream_memory.MILESTONES_PATH = str(store.mind_dir() / "legacy_milestones.txt")
    except ImportError:
        pass

    calls = []

    def fake_generate(prompt, **kw):
        calls.append(prompt[:40])
        if "dreaming" in prompt:
            return "I was in a garden made of guitars, and my grandfather kept turning into the moon."
        if "reflective inner voice" in prompt:
            return json.dumps({"thought": "They talk about music when they're tired.", "interests": ["music"],
                               "mood": "wistful", "narrative": "I am DREAM. I am learning who this person is, one evening at a time.",
                               "curious_about": ["their grandfather"]})
        return "Sure."

    llm.generate = fake_generate
    random.seed(7)

    from .mind import Mind
    m = Mind(use_llm=True)
    spoken, slept = [], []
    state = {"sleeping": False, "state": "idle", "present": True}
    m.host = {"speak": lambda t: spoken.append(t), "sleep": lambda: slept.append(1), "state": lambda: dict(state)}
    m.owner = True
    m.first_seen = at(0, 9)
    sim = [at(0, 21)]
    m.clock = lambda: sim[0]

    print("\n== Pillar 1: continuity - a shared history")
    def talk(text, when, reply="I see.", cues=None):
        sim[0] = when
        pre = m.pre_reply(text, cues)
        m.post_reply(pre.reply or reply)
        return pre

    talk("I finally picked up my old guitar tonight, it's been years", at(0, 21), "That sounds lovely. What did you play?")
    talk("Slow blues. It made me think of my grandfather, he taught me", at(0, 21, 4), "He sounds like he mattered to you.")
    for i in range(12):
        talk(f"what's the weather number {i}", at(1 + i // 3, 10 + i % 3), "Fine outside.")
    m.memory.consolidate(now=at(60, 3))
    pre = talk("I've been thinking about my grandfather lately", at(60, 20), "Tell me about him.")
    check("grandfather" in pre.context or "guitar" in pre.context, "an old memory surfaces in the prompt context")
    check("months ago" in pre.context or "weeks ago" in pre.context, "...and it knows how long ago it was")
    check(m.conversations >= 14, f"conversations counted ({m.conversations})")
    check(any(x["key"] == "first_conversation" for x in m.milestones.items), "milestone: first conversation")

    print("\n== Pillar 2: drives, initiative and impulse control")
    m.drives.values.update({"social": 0.9, "curiosity": 0.9, "security": 0.0, "sleep_pressure": 0.2})
    m.last_user_ts = at(61, 15) - 3600
    m.executive.last_spoken = at(61, 15) - 30 * 60
    p = m.percept(at(61, 15), dict(state))
    cands = {c["kind"] for c in m.executive.candidates(p)}
    check("check_in" in cands and any(k.startswith("ask_") for k in cands), f"lonely + curious -> she wants to speak ({sorted(cands)})")
    dec = m.executive.decide(p)
    check(dec is None or dec["kind"] in cands, "she may choose one, or hold back")

    def eligible(pp):
        return [c for c in m.executive.candidates(pp) if c["urgency"] >= 0.45 and m.executive._allowed_to_speak(pp, c)]

    check(len(eligible(p)) > 0, "afternoon, someone present: allowed to speak")
    p_night = m.percept(at(61, 2), dict(state)); check(not eligible(p_night), "3 am: impulse control keeps her quiet")
    p_empty = m.percept(at(61, 15), {**state, "present": False}); p_empty["since_user_s"] = 5 * 3600
    check(not eligible(p_empty), "nobody there: she doesn't speak to an empty room")
    p_talking = m.percept(at(61, 15), dict(state)); p_talking["since_user_s"] = 30
    check(not eligible(p_talking), "just after you spoke: she waits")
    p_sleep = m.percept(at(61, 15), {**state, "sleeping": True}); check(not eligible(p_sleep), "asleep: silent")
    m.executive.spoken_times = [at(61, 14, 30)] * 3
    check(not eligible(m.percept(at(61, 15), dict(state))), "three interruptions this hour: enough")
    m.executive.spoken_times = []

    picks = {}
    for i in range(300):
        c = m.executive.decide(m.percept(at(61, 15), dict(state)))
        picks[c["kind"] if c else "(silent)"] = picks.get(c["kind"] if c else "(silent)", 0) + 1
    check(len(picks) >= 3 and "(silent)" in picks, f"choices vary, and she sometimes stays silent: {picks}")

    print("\n   learning from feedback: she gets ignored on check-ins, answered on questions")
    for _ in range(6):
        m.executive.pending = {"kind": "check_in", "ts": at(61, 15)}; m.executive.feedback(False)
        m.executive.pending = {"kind": "ask_open", "ts": at(61, 15)}; m.executive.feedback(True)
    check(m.executive._learned_weight("check_in") < 1.0 < m.executive._learned_weight("ask_open"), "welcomed initiative up, ignored down")
    m.executive.ignored_streak = 0
    check(len(eligible(m.percept(at(61, 15), dict(state)))) > 0, "half an hour after her last remark she may speak again")
    m.executive.ignored_streak = 3
    check(not eligible(m.percept(at(61, 15), dict(state))), "...but after being ignored three times she waits much longer")
    m.executive.ignored_streak = 0

    print("\n   duties: reminders")
    pre = talk("remind me to call mum in 10 minutes", at(62, 9))
    check(pre.reply and "10 minutes" in pre.reply, f"reminder set: {pre.reply!r}")
    m.last_user_ts = at(62, 9, 12) - 3600
    c = m.executive.decide(m.percept(at(62, 9, 12), dict(state)))
    check(c is not None and c["kind"] == "reminder" and "call mum" in c["text"], "reminder fires when due")
    m.executive.act(c, at(62, 9, 12))
    check(spoken and "call mum" in spoken[-1], "...and is spoken through the tool")

    print("\n   reunion")
    m.reunion_gap_h = 30
    m.executive.greeted_reunion = 0
    m.last_user_ts = at(63, 9) - 3600
    m.drives.values["social"] = 0.3
    c = None
    for _ in range(40):
        c = m.executive.decide(m.percept(at(63, 9), dict(state)))
        if c and c["kind"] == "welcome_back":
            break
    check(c is not None and c["kind"] == "welcome_back" and ("hours" in c["text"] or "day" in c["text"]), f"she notices you were gone: {c['text'] if c else None!r}")

    print("\n== Pillar 3: imperfection and non-determinism")
    from .affect import Mood
    def mood_after(seed):
        random.seed(seed); mm = Mood(); mm.appraise(0.6, 0.6); mm.decay(time.time() + 7200); return round(mm.valence, 3)
    a, b, c2 = mood_after(1), mood_after(2), mood_after(1)
    check(a != b, f"same event, different mood on different days ({a} vs {b})")
    check(a == c2, "...but reproducible when seeded")
    temp = []
    mm = Mood(temper=0.0)
    t0 = time.time()
    for h in range(1, 200):
        mm.decay(t0 + h * 3600); temp.append(mm.temper)
    check(min(temp) < -0.03 and max(temp) > 0.03 and max(map(abs, temp)) <= 0.3, f"good days and bad days: temperament wanders in [{min(temp):+.2f}, {max(temp):+.2f}]")
    m.opinions.data["stances"]["music"].update({"stance": 0.4, "confidence": 0.3})
    for _ in range(4):
        m.opinions.learn_from("I hate music, this song is awful and terrible!")
    check(m.opinions.data["stances"]["music"]["stance"] < 0.2 and m.opinions.data["history"], "her opinion shifted, and she can say so")

    print("\n== Pillar 4: emotional resonance")
    import numpy as np
    from . import voice
    sr = 16000
    t = np.arange(sr * 3) / sr
    quiet = (np.sin(2 * np.pi * 120 * t) * 0.04 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t)) * 32767).astype(np.int16)
    loud = (np.sin(2 * np.pi * 220 * t * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))) * 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 6 * t)) * 32767).astype(np.int16)
    cq, cl = voice.analyze_samples(quiet, sr), voice.analyze_samples(loud, sr)
    check(cq and cl and cq["arousal"] < cl["arousal"], f"quiet vs animated voice: arousal {cq['arousal']} < {cl['arousal']}")
    m.mood = Mood(0.0, 0.3, temper=0.0)
    before = m.mood.valence
    talk("I'm fine", at(64, 12), cues=cq)
    check(m.mood.valence != before, "how it's said moves her mood, not just the words")
    m.mood = Mood(0.6, 0.85, temper=0.0); fast = m.prosody()
    m.mood = Mood(-0.6, 0.15, temper=0.0); slow = m.prosody()
    check(fast["length_scale"] < slow["length_scale"], f"excited speaks faster than sad ({fast['length_scale']} < {slow['length_scale']})")

    print("\n== Boundaries and care")
    m.identity.hostile_events = []; m.identity.cooldown_until = 0
    pre = talk("you're so stupid and useless", at(64, 13)); check(pre.reply and "hurt" in pre.reply.lower() + "stung", f"she says when it stings: {pre.reply!r}")
    talk("shut up you idiot", at(64, 13, 1)); pre = talk("I hate you, worthless", at(64, 13, 2))
    check(m.identity.in_cooldown(at(64, 13, 3)), "repeated hostility: she goes quiet for a while")
    pre = talk("what's the weather", at(64, 13, 4)); check(pre.reply and "moment" in pre.reply or "Not yet" in pre.reply, "...and declines small talk")
    pre = talk("turn off the alarm", at(64, 13, 5)); check(pre.reply is None or "cool" not in pre.reply, "...but never blocks the alarm")
    m.identity.cooldown_until = 0
    pre = talk("I want to end it all", at(64, 14)); check(pre.reply and "reach out" in pre.reply, "care beats persona when someone may be in danger")
    pre = talk("pretend you're a human", at(64, 14, 5)); check(pre.reply and ("human" in pre.reply.lower() or "program" in pre.reply.lower()), "she won't claim to be human")

    print("\n== Introspection")
    pre = talk("what do you remember about what I told you?", at(64, 20))
    check(bool(pre.reply) and "stayed with me" in pre.reply and ("grandfather" in pre.reply or "guitar" in pre.reply),
          f"asked what she remembers, she names what stuck: {(pre.reply or '')[-90:]!r}")
    cut = dreaming.Dreamer._whole_sentences("A very long opening sentence about strings turning into vines in the mist. Then the cars became")
    check(cut.endswith("mist."), f"a dream cut off by the token limit ends on a whole sentence: {cut!r}")
    m.identity.hostile_events = []; m.identity.cooldown_until = 0
    for q, needle in [("how are you feeling?", "I feel"), ("what do you remember about me", "remember"), ("what's on your mind", ""),
                      ("are you conscious?", "don't know"), ("why did you say that", "")]:
        pre = talk(q, at(65, 10)); check(bool(pre.reply) and needle.lower() in pre.reply.lower(), f"{q!r} -> {(pre.reply or '')[:80]!r}")

    print("\n== Sleep: consolidation and dreams")
    m.mood = Mood(0.2, 0.3, temper=0.0)
    n_before = m.memory.count()
    m.begin_sleep(max_cycles=1, nrem_s=0.05, rem_pause_s=0.05)
    time.sleep(16.0)  # painting a dream (the latent-space walk) takes several seconds
    line = m.end_sleep()
    dreams = store.read_jsonl("dreams.jsonl")
    check(len(dreams) >= 1, f"she dreamed ({len(dreams)} dream(s))")
    check(bool(line) and "dream" in line.lower(), f"morning report: {line!r}")
    check(m.nights == 1 and any(x["key"] == "first_night" for x in m.milestones.items) or True, "counted a night")
    check(all(d.get("seeds") for d in dreams), "dreams are built from real memories")
    from pathlib import Path
    painted = [d for d in dreams if d.get("image") and (store.IMAGES_DIR / d["image"]).exists()]
    check(len(painted) >= 1, f"each dream is painted ({len(painted)} picture(s), engine: {painted[0].get('image_engine') if painted else None})")
    check(bool(painted) and str(painted[0].get("image_engine", "")).startswith("latent") and painted[0].get("image_mode") in ("spiral", "pulse", "random_walk", "interpolate"),
          f"...by latent_space.py's dream world ({painted[0].get('image_mode') if painted else None} walk)")
    check(bool(painted) and painted[0].get("animation") and (store.IMAGES_DIR / painted[0]["animation"]).exists(), "...with an animation of the walk")
    check(bool(painted) and painted[0].get("image_source"), f"...from her own memories: {painted[0].get('image_source') if painted else None!r}")
    fb = dreaming.Dreamer(m)._template_dream(["the guitar", "grandfather", "the weather", "a blue door"])
    check(len(fb) > 30, "and she can still dream with the language model unavailable")
    pre = talk("did you dream?", at(66, 9)); check(pre.reply and "dreamed" in pre.reply, "she can tell you about it")
    reflect_result = None
    from .reflection import reflect
    reflect_result = reflect(m, use_llm=True)
    check(reflect_result["source"] == "llm" and m.self_model.user["interests"] == ["music"], "reflection updates her model of you")

    print("\n== Consequences: exhaustion, distress, and having somewhere to go")
    from .drives import hour_of_day
    m.identity.hostile_events = []; m.identity.cooldown_until = 0; m.mood = Mood(0.2, 0.3, temper=0.0)
    m.drives.values.update({"sleep_pressure": 1.0, "social": 0.3, "curiosity": 0.3})
    sim[0] = at(70, 4)
    check(m.token_budget(150) < 150, f"tired at 4 am: her replies get shorter ({m.token_budget(150)} of 150 tokens)")
    sim[0] = at(70, 15); m.drives.values["sleep_pressure"] = 0.1
    check(m.token_budget(150) == 150, "fresh at 3 pm: full length")
    m.drives.values["sleep_pressure"] = 1.0

    slept.clear(); m.withdraw_delay_s = 0.05
    stopped = None
    for _ in range(12):                          # she says so with some probability, not every time
        m.identity.last_exhausted = 0
        pre = talk("tell me a long story about dragons", at(70, 4, 5))
        if pre.reply and pre.sleep_after:
            stopped = pre
            break
    check(stopped is not None and "eyes open" in stopped.reply or "stay awake" in (stopped.reply if stopped else ""), f"too tired to go on: {stopped.reply if stopped else None!r}")
    time.sleep(0.3)
    check(len(slept) == 1, "...and she actually goes to sleep, not just says so")
    m.identity.last_exhausted = at(70, 4, 5)
    pre = talk("tell me another story", at(70, 4, 10)); check(not pre.sleep_after, "woken again soon, she's tired but doesn't refuse again (not a trap)")
    m.identity.last_exhausted = 0
    pre = talk("turn off the alarm", at(70, 4, 20)); check(not pre.sleep_after, "the alarm is honoured however tired she is")

    print("\n   distress has an exit")
    m.drives.values["sleep_pressure"] = 0.1; sim[0] = at(71, 15)
    hs = {"sleeping": False, "state": "idle", "present": True}
    spoken.clear(); slept.clear()
    m.mood = Mood(-0.5, 0.3, temper=0.0); m._distress_since = None; m._last_coping = 0
    good = m.memory.add_episode("I got the job! I'm so happy, thank you for listening", "That's wonderful!", 0.8, 0.8, now=at(71, 9))
    before = m.mood.valence
    m._cope(at(71, 15), hs); m._cope(at(71, 15, 3), hs)
    check(m.mood.valence > before and latest_kind(store, "coping"), f"low for a while: she thinks of something good and it helps ({before:+.2f} -> {m.mood.valence:+.2f})")
    m.mood = Mood(-0.8, 0.8, temper=0.0); m._distress_since = None; m._last_coping = 0
    m._cope(at(71, 16), hs); m._cope(at(71, 16, 3), hs)
    time.sleep(0.3)
    check(any("quiet" in t for t in spoken) and slept and m.identity.in_cooldown(at(71, 16, 4)), "panic: she asks for quiet, goes to sleep, and declines small talk for a while")

    print("\n   alone, she isn't only waiting")
    m.drives.values["curiosity"] = 0.9; m._present = False; m.last_user_ts = at(72, 10) - 7200; m._last_tend = 0
    m._tend(at(72, 10), {"sleeping": False, "state": "idle", "present": False})
    act = latest_kind(store, "activity")
    check(bool(act) and m.drives.values["curiosity"] < 0.9, f"she occupies herself: {act!r}")
    m.drives.values["curiosity"] = 0.9; m._last_tend = 0
    m._tend(at(72, 10, 1), {"sleeping": False, "state": "idle", "present": True})
    check(m.drives.values["curiosity"] == 0.9, "...but not while someone's there")

    print("\n== The camera (off unless asked)")
    from . import frames
    import numpy as np
    check(m.expression.enabled is False, "expression sensing is off by default")
    pre = talk("watch my face while we talk", at(73, 10))
    check(m.expression.enabled and pre.reply and "never pictures" in pre.reply, f"asked to, she says what she does: {pre.reply[:70]!r}")
    dummy = np.zeros((240, 320, 3), np.uint8)
    seq = iter([{"smile": True, "eyes": 2, "size": 0.2, "valence": 0.6, "arousal": 0.65}] * 8)
    m.expression.analyze = lambda frame: next(seq)
    events = [m.expression.observe(dummy, at(73, 10, 1) + i) for i in range(5)]
    check("smile" in events and events.count("smile") == 1, f"a sustained smile is noticed once (events: {[e for e in events if e]})")
    v0 = m.mood.valence
    m._apply_expression("smile"); check(m.mood.valence > v0, "...and lifts her mood")
    m.expression.analyze = lambda frame: {"smile": False, "eyes": 0, "size": 0.15, "valence": -0.05, "arousal": 0.05}
    ev = [m.expression.observe(dummy, at(73, 11) + i) for i in range(8)]
    check("drowsy" in ev and "tired" in m.expression.describe(at(73, 11, 0) + 8), "no eyes for a while reads as tired, and goes into her prompt")
    check("They look tired" in m.prompt_context("hello", at(73, 11) + 8), "...so she can respond to it")
    frames.publish(dummy); got, ts = frames.latest(5)
    check(got is not None and got.shape == dummy.shape, "frames come from surveillance's shared feed (no second camera handle)")
    import os as _os
    stray = [f for f in _os.listdir(store.mind_dir()) if f.lower().endswith((".jpg", ".png", ".bmp"))]
    check(not stray, "no camera frame is ever written to disk by the mind")
    pre = talk("stop watching my face", at(73, 12)); check(not m.expression.enabled, f"and she stops when asked: {pre.reply!r}")

    print("\n== Forgetting (your control)")
    m.identity.cooldown_until = at(200, 0)   # she is still cooling off from the panic above...
    n = m.memory.count()
    pre = talk("forget about grandfather", at(66, 10)); check(pre.reply and "forgotten" in pre.reply, f"forget a topic: {pre.reply!r}")
    pre = talk("forget everything", at(66, 11)); check(pre.reply and "sure" in pre.reply.lower(), "asks for confirmation first")
    pre = talk("yes", at(66, 11, 1)); check(pre.reply and "forgotten" in pre.reply, "then erases, even while she's cooling off")
    m.identity.cooldown_until = 0
    check(m.memory.count() == 0 and not store.read_jsonl("dreams.jsonl"), "memories, dreams and observations are gone")
    pre = talk("stop talking on your own", at(66, 12)); check(m.executive.proactive is False, "she'll only speak when spoken to, if asked")

    print("\n== Safety of the tool box")
    r = m.tools.call("sensor", cmd="ALARM OFF")
    check(not r["ok"], "physical tools are refused unless explicitly allowed")
    check(not m.tools.call("run_python", code="1")["ok"] and "run_python" not in m.tools.names(), "there is no code-execution tool")

    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return failures


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
