"""
dream_wake.py - how hard DREAM is to wake up. One place for app.py (the page and the exe) and
scripts/dream.py.

config.json (Config/DREAM):

    "InstantWake": false   true: the first "wake up" (or "Dream", or someone at the sensor)
                           wakes her, like before.
                           false (the default): it takes 1 to 10 calls, like a person. How
                           many comes from her mind when it's running: how sleepy she is
                           (sleep pressure + her body clock, deepest around 4 AM, so a long
                           sleep makes her easier to wake), plus a random depth each night;
                           on edge (after an alarm) she sleeps lightly. Without the mind:
                           1 to 10 at random. Until then each call only stirs her: she
                           mumbles something sleepy, often about the real time ("It's
                           3 AM..."), sweeter or grumpier with her mood, or sleeps on.
                           Woken after many calls in the night, she's a little grumpy.

    sleeper = Sleeper(lambda: MIND)        her mind (or None), asked at each call
    sleeper.fell_asleep()                  when she falls asleep
    woke, line = sleeper.nudge()           a wake word, or someone at the sensor, while asleep
        woke True:  she's up; say `line` (her groggy greeting) instead of the usual "Yes?"
                    (None after a single call: the normal greeting)
        woke False: still asleep; `line` is a sleepy mumble to say without waking, or None

    python scripts/dream_wake.py           try it: a night of calls
"""

import json
import os
import random
import time

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")

MAX_CALLS = 10
QUIET_CHANCE = 0.35   # how often a call gets no answer at all: she just sleeps on


def instant_wake():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return bool(json.load(f)["Config"]["DREAM"].get("InstantWake", False))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def clock(now=None):
    """The real time the way she'd say it: "3 AM", "half past 3", "10 past 11 PM"."""
    t = time.localtime(now)
    h12 = t.tm_hour % 12 or 12
    ampm = "AM" if t.tm_hour < 12 else "PM"
    m = t.tm_min
    if m < 5:
        return f"{h12} {ampm}"
    if 25 <= m < 35:
        return f"half past {h12}"
    if m < 30:
        return f"{m} past {h12} {ampm}"
    return f"{h12}:{m:02d} {ampm}"


def _part_of_day(now=None):
    h = time.localtime(now).tm_hour
    if h < 5:
        return "night"      # the middle of the night
    if h < 9:
        return "early"
    if h < 18:
        return "day"
    return "evening"


MUMBLES = {
    "night": ["Mmh... it's {time}... go to sleep...",
              "It's {time}... why are you awake?",
              "Nnn... {time}... not now...",
              "It's the middle of the night..."],
    "early": ["Mmh... it's only {time}...",
              "Five more minutes... it's {time}...",
              "Too early... mmh..."],
    "day":   ["Mmh... just a little longer...",
              "I'm resting... it's only {time}...",
              "Five more minutes..."],
    "evening": ["Mmh... it's {time}... let me sleep...",
                "Five more minutes...",
                "I'm so tired..."],
    "any":   ["Mmh...", "Hmm?", "Mm... five more minutes...", "Nnn..."],
}

GETTING_UP = ["Mmh... almost...", "I'm... nearly up...", "Okay... okay... one second..."]

WOKE = {   # after more than one call: she knows you kept at it
    "night": ["Okay, okay, I'm up. It's {time}, you know.",
              "I'm awake... it's {time}. What's wrong?"],
    "early": ["Okay, I'm up. It's {time}. Good morning, I suppose.",
              "Mmh, I'm awake. It's only {time}."],
    "day":   ["Okay, I'm awake. Yes?", "I'm up, I'm up. It's {time}. Yes?"],
    "evening": ["Okay, I'm up. It's {time}. Yes?", "Mmh, I'm awake. What is it?"],
}


SWEET = ["Mmh... is that you?...", "Mm... I was dreaming about you...", "Hmm... come here... five more minutes..."]
GRUMPY = ["No...", "Go away... it's {time}...", "Mmh. Not now.", "Let me sleep..."]


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


class Sleeper:
    def __init__(self, mind=None):
        self._mind = mind or (lambda: None)   # a function: the mind may start after this
        self.depth = 0.5                      # tonight's random part: some nights you sleep deeper
        self.fixed = None                     # without the mind: calls needed, picked at sleep
        self.heard = 0
        self.last_line = None

    def _say(self, pool, t):
        """A line from `pool`, never the same one twice in a row."""
        choices = [x for x in pool if x != self.last_line] or pool
        self.last_line = random.choice(choices)
        return self.last_line.format(time=t)

    def _get_mind(self):
        try:
            return self._mind()
        except Exception:
            return None

    def fell_asleep(self):
        """She's asleep: roll tonight's depth."""
        self.depth = random.random()
        self.fixed = random.randint(1, MAX_CALLS)
        self.heard = 0
        self.last_line = None

    def calls_needed(self, now=None):
        """How many calls it takes right now (1 = the first one wakes her)."""
        if instant_wake():
            return 1
        mind = self._get_mind()
        if mind is None:
            return self.fixed or random.randint(1, MAX_CALLS)
        try:
            if mind.drives.values.get("security", 0.0) > 0.6:
                return 1                                      # on edge: a light sleeper
            hour = time.localtime(now).tm_hour + time.localtime(now).tm_min / 60.0
            sleepy = mind.drives.sleepiness(hour)             # 0 rested .. 1 exhausted, 4 AM
        except Exception:
            return self.fixed or random.randint(1, MAX_CALLS)
        deep = _clamp(sleepy * 0.75 + self.depth * 0.45 - 0.15)
        return 1 + round(deep * (MAX_CALLS - 1))

    def _valence(self):
        mind = self._get_mind()
        try:
            return float(mind.mood.valence) if mind else 0.0
        except Exception:
            return 0.0

    def nudge(self, now=None):
        """A call while she's asleep. (woke, line) - see the top of the file."""
        self.heard += 1
        need = self.calls_needed(now)
        part = _part_of_day(now)
        t = clock(now)
        if self.heard >= need:
            calls, self.heard = self.heard, 0
            mind = self._get_mind()
            if mind is not None and calls >= 4:
                try:   # dragged out of a deep sleep: a little grumpy, but awake
                    mind.mood.appraise(-0.35 if part in ("night", "early") else -0.15, 0.4)
                except Exception:
                    pass
            if calls == 1:
                return True, None   # woke at once: the usual greeting
            return True, random.choice(WOKE[part]).format(time=t)
        if need - self.heard == 1 and random.random() < 0.6:
            return False, self._say(GETTING_UP, t)
        if random.random() < QUIET_CHANCE:
            return False, None
        v = self._valence()
        pool = MUMBLES[part] + MUMBLES["any"]
        if v > 0.3:
            pool = pool + SWEET * 2
        elif v < -0.2 or self.heard >= 5:
            pool = pool + GRUMPY * 2
        return False, self._say(pool, t)


if __name__ == "__main__":
    s = Sleeper()
    s.fell_asleep()
    print(f"InstantWake={instant_wake()}: she needs {s.calls_needed()} call(s) tonight (no mind)")
    for i in range(1, MAX_CALLS + 1):
        woke, line = s.nudge()
        print(f"call {i}: {'AWAKE' if woke else 'asleep'} - {line or '(silence)' if not woke else line or 'Yes?'}")
        if woke:
            break
