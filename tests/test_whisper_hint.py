"""scripts/whisper_model.transcribe: the fleet's names as Whisper's hint, and silence as "" (never
the hint echoed back). A fake model, so it runs without Whisper; the real one was tested with
Piper-spoken lines (10/10 understood, see whisper_model.NAMES)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import dream_fleet  # noqa: E402
import whisper_model  # noqa: E402


class FakeWhisper:
    def __init__(self, text):
        self.text, self.kw = text, None

    def transcribe(self, path, **kw):
        self.kw = kw
        return {"text": self.text}


@pytest.mark.parametrize("heard,want", [
    (" Tell KIDA01 to stop. ", "Tell KIDA01 to stop."),
    (whisper_model.NAMES, ""),                         # near-silence echoing the hint
    (" DREAM talks to the robots Nora, Mila", ""),     # part of it
    (" Thank you.", ""), ("you", ""),                  # Whisper's silence
])
def test_hint_and_guard(heard, want):
    m = FakeWhisper(heard)
    assert whisper_model.transcribe(m, "clip.wav") == want
    assert m.kw["initial_prompt"] == whisper_model.NAMES and m.kw["language"] == "en"
    assert m.kw["condition_on_previous_text"] is False and m.kw["fp16"] is False


@pytest.mark.parametrize("heard,robot", [
    ("Tell KIDA01 to stop.", "KIDA01"), ("What's key to 01 status?", "KIDA01"),
    ("Tell key to 00 to stop.", "KIDA00"), ("How is WIP doing?", "WHIP"), ("Ask Ida how she is.", "IDA"),
])
def test_how_whisper_hears_the_names(heard, robot, monkeypatch):
    monkeypatch.setattr(dream_fleet, "_roster", {"t": 1e18, "names": ["KIDA00", "KIDA01", "WHIP", "IDA", "ARM"]})
    assert dream_fleet.understand(heard)["to"] == robot
    assert dream_fleet.robot_named("the arm") == "ARM"
