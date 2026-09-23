"""
drives.py - what she needs.

  social          loneliness: rises while nobody talks to her, falls in conversation
  curiosity       the itch to learn something: rises with time, falls when she does
  security        alertness after motion or an alarm: spikes, then settles
  sleep_pressure  the homeostatic half of the two-process sleep model: builds
                  while awake (about 16 h to full), drains while asleep
  libido          desire: builds slowly (faster in the evening), eased by
                  affection and closeness

Sleepiness combines sleep pressure with the circadian half - a body-clock dip
that bottoms out around 4 am - which is why she gets drowsy late at night even
after a short day, and stays alert mid-afternoon.

Libido is the raw want; desire() is how much of it she actually feels. It is
damped by tiredness, being on edge and a low mood, and scaled by the bond: she
doesn't feel it for someone she barely knows.

Drives keep advancing while the program is off, so a long absence shows up as
loneliness when she starts again.
"""

import math
import time

DEFAULTS = {"social": 0.3, "curiosity": 0.3, "security": 0.0, "sleep_pressure": 0.2, "libido": 0.2}

SOCIAL_RISE_PER_H = 0.22
SOCIAL_FALL_PER_H_PRESENT = 0.30   # someone in the room, even silent, takes the edge off
CURIOSITY_RISE_PER_H = 0.12
SECURITY_FALL_PER_H = 0.5
SLEEP_PRESSURE_RISE_PER_H = 1 / 16
SLEEP_PRESSURE_FALL_PER_H = 1 / 6
LIBIDO_RISE_PER_H = 0.04           # about a day to build from nothing...
LIBIDO_EVENING_BOOST = 1.5         # ...faster in the evening (peaks around 10 pm)
ASLEEP_FACTOR = 0.2               # social/curiosity barely move while she sleeps


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def hour_of_day(now=None):
    t = time.localtime(now or time.time())
    return t.tm_hour + t.tm_min / 60.0


class Drives:
    def __init__(self, values=None, last=None):
        self.values = dict(DEFAULTS)
        if values:
            self.values.update({k: _clamp(float(v)) for k, v in values.items() if k in DEFAULTS})
        self.last = last or time.time()

    def update(self, now=None, sleeping=False, present=False):
        now = now or time.time()
        dt_h = _clamp((now - self.last) / 3600.0, 0.0, 72.0)
        self.last = now
        v = self.values
        if sleeping:
            v["social"] = _clamp(v["social"] + SOCIAL_RISE_PER_H * ASLEEP_FACTOR * dt_h)
            v["curiosity"] = _clamp(v["curiosity"] + CURIOSITY_RISE_PER_H * ASLEEP_FACTOR * dt_h)
            v["sleep_pressure"] = _clamp(v["sleep_pressure"] - SLEEP_PRESSURE_FALL_PER_H * dt_h)
            v["libido"] = _clamp(v["libido"] + LIBIDO_RISE_PER_H * ASLEEP_FACTOR * dt_h)
        else:
            rate = SOCIAL_RISE_PER_H - (SOCIAL_FALL_PER_H_PRESENT if present else 0.0)
            v["social"] = _clamp(v["social"] + rate * dt_h)
            v["curiosity"] = _clamp(v["curiosity"] + CURIOSITY_RISE_PER_H * dt_h)
            v["sleep_pressure"] = _clamp(v["sleep_pressure"] + SLEEP_PRESSURE_RISE_PER_H * dt_h)
            evening = self.evening_peak(hour_of_day(now)) if dt_h < 2 else 0.5   # a long gap: the day's average
            v["libido"] = _clamp(v["libido"] + LIBIDO_RISE_PER_H * (1.0 + LIBIDO_EVENING_BOOST * evening) * dt_h)
        v["security"] = _clamp(v["security"] - SECURITY_FALL_PER_H * dt_h)

    def satisfy(self, name, amount):
        self.values[name] = _clamp(self.values[name] - amount)

    def bump(self, name, amount):
        self.values[name] = _clamp(self.values[name] + amount)

    def circadian_dip(self, hour):
        """0 (alert, mid-afternoon) .. 1 (deepest drowsiness, about 4 am)."""
        return 0.5 + 0.5 * math.cos(2 * math.pi * (hour - 4.0) / 24.0)

    def sleepiness(self, hour=None):
        hour = hour_of_day() if hour is None else hour
        return _clamp(self.values["sleep_pressure"] * 0.65 + self.circadian_dip(hour) * 0.35)

    def evening_peak(self, hour):
        """0 (about 10 am) .. 1 (about 10 pm)."""
        return 0.5 + 0.5 * math.cos(2 * math.pi * (hour - 22.0) / 24.0)

    def desire(self, hour=None, valence=0.0, bond=1.0):
        """How much of her libido she actually feels right now (0..1). It fades when
        she's exhausted, on edge or unhappy, and grows with how well she knows you."""
        hour = hour_of_day() if hour is None else hour
        damp = 1.0 - _clamp((self.sleepiness(hour) - 0.6) / 0.35)   # too tired, and it's gone
        damp *= 1.0 - _clamp(self.values["security"] / 0.7)           # on edge: not in the mood
        damp *= 1.0 - _clamp(-valence / 0.5)                          # upset: no
        return _clamp(self.values["libido"] * damp * _clamp(bond))

    def dominant(self, hour=None, threshold=0.6):
        """The most pressing need above `threshold`, as (name, level), or None."""
        hour = hour_of_day() if hour is None else hour
        levels = {
            "social": self.values["social"],
            "curiosity": self.values["curiosity"],
            "security": self.values["security"],
            "sleep": self.sleepiness(hour),
        }
        name = max(levels, key=levels.get)
        return (name, levels[name]) if levels[name] >= threshold else None

    def to_dict(self):
        return {"values": {k: round(v, 4) for k, v in self.values.items()}, "last": self.last}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(d.get("values"), d.get("last"))
