"""DREAM's sense of the weather (dream_mind/sky.py): now and then, not a weather report."""
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from dream_mind.sky import Sky  # noqa: E402


def sky_with(outside, air=None, moon=None):
    s = Sky()
    s.data = {"outside": outside, "air": air or {}, "moon": moon or {"phase": "waning crescent"},
              "location": {"name": "Johannesburg"}}
    s.got = time.time()
    return s


CLEAR = {"temp": 27.0, "words": "mostly clear", "high": 29, "low": 16, "sunset": "2026-11-05T18:27", "soon": {}}
RAIN = dict(CLEAR, soon={"rain_at": "16:00", "rain_chance": 70, "storm_at": None})


def test_mostly_quiet_in_conversation():
    random.seed(1)
    s = sky_with(CLEAR)
    lines = [s.prompt_line("how was your day") for _ in range(400)]
    said = [l for l in lines if l]
    assert 20 < len(said) < 90                                # an ordinary day: about one reply in eight
    assert "27 degrees and mostly clear" in said[0] and "may bring this up in passing" in said[0]


def test_more_when_something_is_coming_then_fades():
    random.seed(2)
    s = sky_with(RAIN)
    said = [l for l in (s.prompt_line("hi") for _ in range(200)) if l]
    assert 70 < len(said) < 130 and "rain is likely around 16:00" in said[0]
    s.note_reply("Oh, and it might rain later, take an umbrella.")
    assert all(s.prompt_line("hi") is None for _ in range(100))   # she said it: quiet for a few hours
    s.note_reply("I love talking to you.")                        # not about the weather: nothing changes


def test_her_own_initiative():
    assert sky_with(RAIN).candidate()[0] == "weather_rain"
    storm = dict(CLEAR, soon={"rain_at": "15:00", "storm_at": "17:00"})
    assert sky_with(storm).candidate()[0] == "weather_storm"
    assert sky_with(CLEAR, air={"aqi": 160, "words": "unhealthy"}).candidate()[0] == "weather_air"
    assert sky_with(CLEAR).candidate(mood_valence=0.5, hour=10)[0] == "weather_nice"
    assert sky_with(CLEAR).candidate(mood_valence=-0.2, hour=10) is None   # grumpy: a nice day isn't worth saying
    assert sky_with(CLEAR).candidate(mood_valence=0.5, hour=21) is None


def test_morning_decided_once():
    random.seed(3)
    s = sky_with(RAIN)
    first = s.morning()
    assert all(s.morning() == first for _ in range(20))
    days = [sky_with(RAIN).morning() for _ in range(200)]
    with_weather = [d for d in days if d]
    assert 90 < len(with_weather) < 150 and "Rain around 16:00" in with_weather[0]


def test_sunset_once_in_the_half_hour():
    s = sky_with(CLEAR)
    day = datetime.now().replace(hour=18, minute=27, second=0, microsecond=0)
    before, after, late = (day.timestamp() + m * 60 for m in (-5, 10, 45))

    def at(sky, t):                       # AURA's reading as fresh as that moment
        sky.got = t
        return sky.sunset_now(t)
    assert not at(s, before)
    assert at(s, after) and not at(s, after + 60)                  # once a day
    assert not at(sky_with(CLEAR), late)


def test_nothing_without_aura():
    s = Sky()
    assert s.prompt_line("hi") is None and s.candidate() is None and s.morning() == "" and not s.sunset_now()
    s.data, s.got = {"outside": CLEAR}, time.time() - 7200      # stale: she doesn't know any more
    assert s.prompt_line("hi") is None


def test_new_signs():
    brewing = dict(CLEAR, pressure_trend="falling fast", storm_potential="high")
    s = sky_with(brewing)
    assert s.facts()["notable"][0] == "brewing" and s.candidate()[0] == "weather_brewing"
    assert "pressure is dropping fast" in s._sentence(s.facts())
    assert sky_with(dict(CLEAR, gusts_kmh=62)).candidate()[0] == "weather_wind"
    assert sky_with(dict(CLEAR, fog=True, visibility_km=0.4)).candidate()[0] == "weather_fog"
    assert sky_with(dict(CLEAR, temp=1.0)).candidate()[0] == "weather_frost"
    night = datetime.now().replace(hour=21, minute=0).timestamp()
    clear = sky_with(dict(CLEAR, clouds={"total": 5}))
    clear.got = night
    assert clear.facts(night)["notable"][0] == "stars"
    storm = dict(CLEAR, pressure_trend="falling fast", soon={"storm_at": "17:00"})
    assert sky_with(storm).facts()["notable"][0] == "storm"      # a forecast storm beats the barometer
