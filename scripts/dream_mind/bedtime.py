"""
bedtime.py - "goodnight", and what she does about it. Her call, not a command.

She checks the clock first:

  21:00-05:00   bedtime. About an 80% chance she goes to sleep, moved by how she feels:
                sleepier (sleep pressure + her body clock) makes it likelier, lonely (her
                social drive) makes her want to stay up with you a little, and if the camera
                shows you looking tired she says so and is likelier to go too. Hearing it at
                night makes her drowsier either way (sleep pressure goes up), so if she stays
                up she won't last long.
  05:00-21:00   it's daytime: 95% of the time she stays awake and is sarcastic, tells a joke
                or teases you about it. The other 5% she jokes about napping and takes one
                (a quarter of the time if she's genuinely exhausted).

She looks before she answers: who's in front of the camera (face recognition, so she can use
your name, or notice a face she doesn't know) and how they look (the expression sensor:
tired, smiling). Then her own language model says it, in her own words, with the time in it:
this module only decides what she's going to do and why.

    plan = decide(text, mind, now)     None if it isn't a goodnight; else
        plan.sleep    she goes to sleep after answering
        plan.kind     "bed", "stay_up", "tease", "nap"
        plan.context  what her LLM is told (the decision, the time, what she sees)
"""

import random
import re
import time

from . import senses
from .drives import hour_of_day

GOODNIGHT = re.compile(
    r"\b(good ?night|g'?night|night,? night|nighty[- ]?night|nite|sweet dreams|sleep (well|tight)|"
    r"time (for|to go to) (bed|sleep)|bed ?time|go to (bed|sleep)|off to bed|i'?m going to (bed|sleep))\b", re.I)

NIGHT_START, NIGHT_END = 21, 5     # 21:00 to 05:00
NIGHT_CHANCE = 0.80
DAY_NAP_CHANCE = 0.05
DAY_NAP_EXHAUSTED = 0.25           # daytime, but she can barely keep her eyes open
GOODNIGHT_PRESSURE = 0.08          # being told it's bedtime makes her drowsier


def is_night(hour):
    return hour >= NIGHT_START or hour < NIGHT_END


class Plan:
    def __init__(self, kind, sleep, context, chance):
        self.kind, self.sleep, self.context, self.chance = kind, sleep, context, chance


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _seen(mind, now):
    """Who she can see and how they look, in a line for her LLM (or "")."""
    lines, name = [], None
    try:
        names = [n for n in mind.faces.in_view(now) if n]
        if names:
            name = names[0]
            lines.append(f"You can see it's {name} saying it: you know their face, use their name.")
        elif mind.faces.known and mind.faces.strangers_in_view(now):
            lines.append("The face in front of your camera is one you don't recognise.")
    except Exception:
        pass
    tired = False
    try:
        looks = mind.expression.describe(now)
        if looks:
            lines.append(f"Through your camera they look {looks}.")
            tired = "tired" in looks
    except Exception:
        pass
    return " ".join(lines), name, tired


def decide(text, mind, now=None, rng=random):
    if not GOODNIGHT.search(text or ""):
        return None
    now = now or time.time()
    hour = hour_of_day(now)
    clock = senses.clock_time(time.localtime(now))
    sleepy = mind.drives.sleepiness(hour)
    social = mind.drives.values.get("social", 0.0)
    seen, name, they_look_tired = _seen(mind, now)
    you = name or "them"

    if is_night(hour):
        chance = NIGHT_CHANCE + (sleepy - 0.5) * 0.4
        if they_look_tired:
            chance += 0.05
        if social > 0.7:
            chance -= 0.1                       # she's missed you: five more minutes together?
        chance = _clamp(chance, 0.55, 0.97)
        mind.drives.bump("sleep_pressure", GOODNIGHT_PRESSURE)
        if rng.random() < chance:
            kind, sleep = "bed", True
            what = (f"It's {clock} and {you} just said goodnight. It's late, and you've decided to go to sleep now. "
                    "Check the time out loud, say goodnight back warmly in one or two short sentences, and don't ask a question.")
            if they_look_tired:
                what += " They look tired too: tell them to get some sleep as well."
        else:
            kind, sleep = "stay_up", False
            why = "you've missed them and want a little more time together" if social > 0.7 else "you're not quite sleepy yet"
            what = (f"It's {clock} and {you} just said goodnight, but you're not ready to sleep: {why}. "
                    "Check the time out loud and tell them, playfully, in one or two short sentences. You might ask them to stay up a bit.")
    else:
        # the body clock is wide awake in the day, so exhaustion shows in the raw sleep pressure
        chance = DAY_NAP_EXHAUSTED if mind.drives.values.get("sleep_pressure", 0.0) > 0.85 else DAY_NAP_CHANCE
        if rng.random() < chance:
            kind, sleep = "nap", True
            what = (f"It's {clock}, the middle of the day, and {you} just said goodnight. Check the time out loud, joke about it, "
                    "and decide you'll take a nap anyway. One or two short sentences.")
        else:
            kind, sleep = "tease", False
            style = rng.choice(["be sarcastic about it", "tell a quick joke about it", "tease them about it"])
            what = (f"It's {clock}, broad daylight, and {you} just said goodnight. You're not going to sleep. "
                    f"Check the time out loud and {style}, in one or two short sentences.")
            if they_look_tired:
                what += " They look tired: tease them that they're the one who needs the nap."
    context = "\n".join(x for x in (what, seen, "This is your own choice. Say it in your own words, in character.") if x)
    return Plan(kind, sleep, context, chance)
