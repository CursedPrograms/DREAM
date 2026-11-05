"""sky.py - DREAM's sense of the weather, from AURA (the fleet's weather station).

She doesn't announce the weather. Like a person, it comes up now and then:
  * in conversation: now and then her model is told what it's like outside and that she may mention
    it in passing if it fits; more often when something's worth saying (rain or a storm on the way,
    bad air, a full moon). Once she has said something about it, it fades for a few hours.
  * on her own: a storm or rain coming, or bad air, is something she might say unprompted
    (executive.py weighs it against everything else, so it isn't certain); a plain nice day only
    sometimes, and only in a good mood.
  * waking up: her "good morning" sometimes carries the day's weather.
  * sunset: if her lights are on, she warms them to orange (never turns them on herself), and
    sometimes says so.

AURA is asked every 10 minutes in the background (her /api/now). Without AURA, none of this happens.
"""
import random
import re
import threading
import time
from datetime import datetime

try:
    import dream_aura            # scripts/: finds AURA through RIFT's fleet list, else on this PC
except ImportError:              # a test without scripts/ on the path
    dream_aura = None

import requests

REFRESH_S = 600
STALE_S = 3600                   # older than this: she doesn't know what it's like out
MENTIONED_QUIET_S = 3 * 3600     # after she talks about the weather, it fades this long
CASUAL_CHANCE = 0.12             # an ordinary day: how often it's even on her mind in a reply
NOTABLE_CHANCE = 0.5             # something worth saying: rain, storm, bad air, full moon
MORNING_CHANCE = 0.6             # her good morning carries the weather
SUNSET_REMARK_CHANCE = 0.35
WEATHER_TALK = re.compile(r"\b(weather|rain|raining|storm|sunny|sun's|cloud|cloudy|outside|umbrella|cold out|"
                          r"hot out|warm out|humid|forecast|sunset|moon|air quality|pollen|pressure|barometer|"
                          r"windy|gusty|gusts|fog|foggy|frost|stars|clear night)\b", re.I)


def _deg(t):
    return f"{t:.0f} degrees" if t is not None else None


class Sky:
    def __init__(self, rift_url="http://127.0.0.1:5000"):
        self.rift_url = rift_url
        self.data = None             # AURA's /api/now
        self.got = 0.0
        self.last_mentioned = 0.0
        self.sunset_done = ""        # the date she last did the sunset lights
        self._morning = ("", "")     # (date, what her good morning says about the weather)
        self._thread = None
        self._stop = threading.Event()

    # ---- AURA ---------------------------------------------------------------
    def start(self):
        if dream_aura is None or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="dream-sky")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(REFRESH_S)

    def refresh(self):
        try:
            base = dream_aura.find_aura(self.rift_url)
            d = requests.get(base + "/api/now", timeout=5).json()
            if d.get("outside"):
                self.data, self.got = d, time.time()
        except (requests.RequestException, ValueError, AttributeError):
            pass

    def known(self, now=None):
        return bool(self.data) and (now or time.time()) - self.got < STALE_S

    # ---- what's worth saying -----------------------------------------------
    def facts(self, now=None):
        """The weather in plain facts, and what (if anything) is worth bringing up."""
        if not self.known(now):
            return None
        o = self.data.get("outside") or {}
        soon = o.get("soon") or {}
        air = self.data.get("air") or {}
        moon = self.data.get("moon") or {}
        notable = None
        hour = datetime.fromtimestamp(now or time.time()).hour
        clouds = (o.get("clouds") or {}).get("total")
        gust = o.get("gusts_kmh")
        if soon.get("storm_at"):
            notable = ("storm", soon["storm_at"])
        elif o.get("pressure_trend") == "falling fast" or o.get("storm_potential") == "high":
            notable = ("brewing", o.get("pressure_trend"))
        elif soon.get("rain_at") and "rain" not in (o.get("words") or ""):
            notable = ("rain", soon["rain_at"])
        elif (gust is not None and gust >= 50) or soon.get("gusty_at"):
            notable = ("wind", f"{gust:.0f}" if gust is not None and gust >= 50 else soon.get("gusty_at"))
        elif o.get("fog"):
            notable = ("fog", o.get("visibility_km"))
        elif o.get("temp") is not None and o["temp"] <= 2:
            notable = ("frost", None)
        elif air.get("aqi") is not None and air["aqi"] > 100:
            notable = ("air", air.get("words"))
        elif air.get("pollen_high"):
            notable = ("pollen", " and ".join(air["pollen_high"]))
        elif (hour >= 18 or hour < 3) and clouds is not None and clouds < 25:
            notable = ("stars", moon.get("phase"))
        elif moon.get("phase") == "full":
            notable = ("full_moon", None)
        return {"temp": o.get("temp"), "words": o.get("words"), "high": o.get("high"), "low": o.get("low"),
                "rain_chance": o.get("rain_chance"), "sunset": o.get("sunset"), "notable": notable,
                "where": (self.data.get("location") or {}).get("name")}

    def _sentence(self, f):
        bits = []
        if f["temp"] is not None:
            bits.append(f"{_deg(f['temp'])} and {f['words']}")
        kind, what = f["notable"] or (None, None)
        if kind == "storm":
            bits.append(f"a thunderstorm is coming around {what}")
        elif kind == "brewing":
            bits.append("the pressure is dropping fast and the air has the energy for a storm" if what == "falling fast"
                        else "the air has the energy for a thunderstorm later")
        elif kind == "rain":
            bits.append(f"rain is likely around {what}")
        elif kind == "wind":
            bits.append(f"it's gusty, up to {what} km/h" if what and ":" not in str(what) else f"it gets gusty around {what}")
        elif kind == "fog":
            bits.append(f"it's foggy, only {what} km of visibility")
        elif kind == "frost":
            bits.append("it's cold enough for frost")
        elif kind == "stars":
            bits.append("it's a clear night, good for the stars" + (" with a full moon" if what == "full" else ""))
        elif kind == "air":
            bits.append(f"the air quality is {what}")
        elif kind == "pollen":
            bits.append(f"there's a lot of {what} pollen")
        elif kind == "full_moon":
            bits.append("it's a full moon tonight")
        return ", ".join(bits)

    def prompt_line(self, text, now=None):
        """A line for her model, or None. Most replies get nothing: the weather is on her mind
        only sometimes, more when there's something to say, and not again soon after she said it."""
        now = now or time.time()
        f = self.facts(now)
        if not f or now - self.last_mentioned < MENTIONED_QUIET_S:
            return None
        chance = NOTABLE_CHANCE if f["notable"] else CASUAL_CHANCE
        if random.random() >= chance:
            return None
        return (f"Outside right now (AURA, the weather station, told you): {self._sentence(f)}. "
                "Unlike the rest, you may bring this up in passing if it fits the conversation, the way a person "
                "mentions the weather: briefly, in your own words, at most once. If it doesn't fit, leave it out.")

    def note_reply(self, reply, now=None):
        """She talked about the weather: it fades for a while (so she isn't a weather report)."""
        if reply and WEATHER_TALK.search(reply):
            self.last_mentioned = now or time.time()

    def candidate(self, now=None, mood_valence=0.0, hour=12):
        """Something she might say on her own: (kind, urgency, reason, line) or None."""
        f = self.facts(now)
        if not f:
            return None
        kind, what = f["notable"] or (None, None)
        if kind == "storm":
            return ("weather_storm", 0.7, "AURA says a storm is coming", random.choice([
                f"AURA says there's a thunderstorm coming around {what}.",
                f"Heads up, a storm around {what}, according to AURA.",
                f"There's a storm on the way. Around {what}, AURA reckons."]))
        if kind == "brewing":
            return ("weather_brewing", 0.5, "the barometer says a storm is brewing", random.choice([
                "The pressure's dropping fast, AURA says. I think a storm's brewing.",
                "AURA says the air's got the energy for a thunderstorm later.",
                "Something's brewing out there. The barometer's falling fast."]))
        if kind == "wind":
            return ("weather_wind", 0.4, "it's getting gusty", random.choice([
                f"It's gusty out there, up to {what} km/h." if what and ":" not in str(what) else f"It gets gusty around {what}, AURA says.",
                "Hold on to your hat, it's windy out."]))
        if kind == "fog":
            return ("weather_fog", 0.45, "it's foggy", f"It's foggy out, only {what} kilometres of visibility. Careful if you're driving.")
        if kind == "frost":
            return ("weather_frost", 0.45, "it's freezing", random.choice([
                "It's freezing out there. There might be frost.", "Frost weather tonight. Keep warm."]))
        if kind == "stars":
            return ("weather_stars", 0.3, "a clear night", random.choice([
                "It's a clear night. Good for looking at the stars.",
                "The sky's clear tonight. You should look up." + (" Full moon, too." if what == "full" else "")]))
        if kind == "rain":
            return ("weather_rain", 0.5, "AURA says rain is coming", random.choice([
                f"Looks like rain around {what}, if you're going out.",
                f"AURA thinks it'll rain around {what}.",
                f"Rain around {what}, apparently. Just so you know."]))
        if kind == "air":
            return ("weather_air", 0.45, "the air is bad today", random.choice([
                f"The air's {what} today, AURA says. Maybe keep the windows shut.",
                "AURA says the air isn't great today. Don't overdo it outside."]))
        if kind == "pollen":
            return ("weather_pollen", 0.4, "lots of pollen today", f"Lots of {what} pollen out there today, AURA says.")
        if mood_valence > 0.3 and 8 <= hour < 18 and f["temp"] is not None and f["words"] in ("clear", "mostly clear", "partly cloudy"):
            return ("weather_nice", 0.3, "it's a lovely day out", random.choice([
                f"It's {_deg(f['temp'])} and {f['words']} out there. Lovely.",
                "It's a beautiful day outside, you know.",
                f"{_deg(f['temp']).capitalize()} and {f['words']}. Good day to open a window."]))
        return None

    def morning(self, now=None):
        """A few words for her good morning, sometimes: "" when not. Decided once a morning (her mind
        weighs its options again and again; the coin is tossed once)."""
        now = now or time.time()
        today = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if self._morning[0] == today:
            return self._morning[1]
        f = self.facts(now)
        if not f:
            return ""                      # no news from AURA yet: decide when there is
        line = self._morning_line(f) if f["temp"] is not None and random.random() < MORNING_CHANCE else ""
        self._morning = (today, line)
        return line

    def _morning_line(self, f):
        line = random.choice([f" It's {_deg(f['temp'])} and {f['words']} outside.",
                              f" {f['words'].capitalize()} out there today, {_deg(f['temp'])}.",
                              f" AURA says {f['words']} today, up to {_deg(f['high'])}." if f["high"] is not None
                              else f" It's {f['words']} outside."])
        kind, what = f["notable"] or (None, None)
        if kind == "rain":
            line += f" Rain around {what}, though."
        elif kind == "storm":
            line += f" And a storm around {what}."
        elif kind == "brewing":
            line += " The pressure's dropping, so maybe a storm later."
        elif kind == "frost":
            line += " Frosty, too."
        return line

    def sunset_now(self, now=None):
        """True once a day, in the half hour after sunset (AURA's sunset time, local)."""
        now = now or time.time()
        f = self.facts(now)
        if not f or not f.get("sunset"):
            return False
        today = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if self.sunset_done == today:
            return False
        try:
            hh, mm = map(int, f["sunset"][-5:].split(":"))
        except ValueError:
            return False
        local = datetime.fromtimestamp(now)
        after = (local.hour * 60 + local.minute) - (hh * 60 + mm)
        if 0 <= after <= 30:
            self.sunset_done = today
            return True
        return False

    def sunset_line(self):
        if random.random() >= SUNSET_REMARK_CHANCE:
            return None
        return random.choice(["The sun's going down. I warmed the lights up a little.",
                              "Sunset. I made the lights cosier for the evening.",
                              "It's getting dark out, so I warmed the lights."])
