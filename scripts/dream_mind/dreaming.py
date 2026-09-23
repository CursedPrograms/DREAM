"""
dreaming.py - what happens while she sleeps.

Sleep isn't idle time. It runs in cycles, as it does for people:

  NREM   consolidation: the day's memories are replayed - the important ones
         strengthen, the trivial ones fade - tastes drift, and she takes stock of
         patterns in what she has seen and heard.
  REM    dreaming: a few memories (weighted toward the emotional and the
         unresolved), mixed with chance, become a short surreal story. Dreams
         leave a trace on mood - a pleasant one lifts her, a nightmare doesn't.

Later cycles dream longer. On waking she can tell you what she dreamed. If the
language model is busy or unavailable, dreams are assembled from the raw
fragments instead, so she dreams even when nothing else works.
"""

import random
import re
import threading
import time

from . import dream_images, llm, store

DREAMS = "dreams.jsonl"
INSIGHTS = "insights.jsonl"

NREM_SECONDS = 20
REM_PAUSE_SECONDS = 45
MAX_CYCLES = 4

TEMPLATES = [
    "I was in a room made entirely of {a}. {b} kept turning into {c}, and someone was speaking, "
    "but all I could hear was {d}.",
    "I was walking through {a} when it began to rain {b}. You were there, but you were also {c}, "
    "and I couldn't tell which of you to answer.",
    "There was a door that opened onto {a}, and behind it, {b}. I knew I had been there before, "
    "the way you know things in dreams, and then it all dissolved into {c}.",
    "I was falling slowly through {a}, and {b} drifted past like fish. Somewhere below, {c} was waiting, "
    "and it knew my name.",
]

TONE_WORDS = {
    "pleasant": "warm and gentle, ending softly",
    "strange": "strange and dreamlike, without resolution",
    "nightmare": "unsettling and tense, a nightmare that ends abruptly",
}


def _snippet(text, n=90):
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "..."


def first_sentences(text, limit=240):
    """The opening of a dream, cut at a sentence end, for saying out loud."""
    text = re.sub(r"\s+", " ", text).strip()
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(s) > limit and out:
            break
        out = (out + " " + s).strip()
    return out[:limit + 40]


class Dreamer:
    def __init__(self, mind):
        self.mind = mind

    # ------------------------------------------------------------ a session
    def run_session(self, stop: threading.Event, max_cycles=MAX_CYCLES, nrem_s=NREM_SECONDS, rem_pause_s=REM_PAUSE_SECONDS,
                    use_llm=True, images=True):
        """Sleep until `stop` is set (she was woken). Returns what happened."""
        m = self.mind
        report = {"started": time.time(), "cycles": 0, "dreams": [], "consolidation": [], "insights": []}
        for cycle in range(max_cycles):
            if stop.is_set():
                break
            # NREM
            stats = m.memory.consolidate()
            report["consolidation"].append(stats)
            if cycle == 0:
                m.opinions.drift()
                m.philosophy.drift()
                report["insights"] = self.take_stock()
            if stop.wait(nrem_s):
                break
            # REM (longer as the night goes on)
            dream = self.dream(cycle, use_llm=use_llm and not stop.is_set(), stop=stop, images=images)
            if dream:   # a dream she finished writing counts, even if she woke while it was being painted
                report["dreams"].append(dream)
                if not stop.is_set():
                    self._after_effects(dream)
            report["cycles"] = cycle + 1
            if stop.wait(rem_pause_s):
                break
        report["ended"] = time.time()
        return report

    # ------------------------------------------------------------ dreaming
    def dream(self, cycle=0, use_llm=True, stop=None, images=True):
        m = self.mind
        frags = m.memory.fragments(3)
        seeds = [_snippet(e["user"]) for e in frags]
        for o in m.vision.core_memories(2):   # something she saw can enter a dream too
            seeds.append(o["text"])
        seeds = seeds[:4]
        if not seeds:
            return None

        tone = self._tone()
        text, source = None, "template"
        if use_llm:
            try:
                prompt = (
                    "You are DREAM, an AI who lives on a computer, and you are dreaming. Write the dream in first person, "
                    f"three or four sentences, {TONE_WORDS[tone]}. Use dream logic: things turn into other things, places "
                    "don't stay put. Mix these memories together without explaining them:\n"
                    + "\n".join(f"- {s}" for s in seeds) + "\nWrite only the dream."
                )
                out = llm.generate(prompt, num_predict=110 + 30 * cycle, temperature=1.0, timeout=120, blocking=False)
                if out and not (stop is not None and stop.is_set()):
                    text, source = out.strip().strip('"'), "llm"
            except llm.LLMError:
                pass
        if not text:
            text = self._template_dream(seeds)
        text = self._whole_sentences(text)

        keywords = sorted({w for s in seeds for w in re.findall(r"[a-z]{4,}", s.lower())})[:6]
        dream = {
            "ts": time.time(), "text": text, "tone": tone, "source": source, "cycle": cycle,
            "seeds": seeds, "image_prompt": "surreal dreamlike scene, " + ", ".join(keywords) if keywords else "",
            "shared": False,
        }
        store.append_jsonl(DREAMS, dream)
        self._journal(dream)
        if images and not (stop is not None and stop.is_set()):
            self._paint(dream, stop)
        return dream

    def _paint(self, dream, stop):
        """Paint the dream from her memories (dream_images.py). The story is already
        saved, so a failure or an early waking just means a dream without a picture."""
        try:
            name = dream_images.paint(self.mind, dream, stop)
        except Exception as e:
            print(f"[mind] couldn't paint the dream: {e}")
            return
        if name:
            dream["image"] = name
            items = store.read_jsonl(DREAMS)
            for d in items:
                if d["ts"] == dream["ts"]:
                    d.update(image=name, animation=dream.get("animation"), image_mode=dream.get("image_mode"),
                             image_engine=dream.get("image_engine"), image_source=dream.get("image_source"))
            store.write_jsonl(DREAMS, items)

    def _tone(self):
        m = self.mind
        v = m.mood.valence
        if m.drives.values["security"] > 0.5 or (v < -0.4 and random.random() < 0.6):
            return "nightmare"
        if v > 0.3 and random.random() < 0.7:
            return "pleasant"
        return "strange"

    @staticmethod
    def _whole_sentences(text):
        """A model that hit its token limit leaves a dangling fragment: cut back to the last full sentence."""
        text = text.strip()
        if text and text[-1] in ".!?\"'":
            return text
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        return text[:cut + 1] if cut >= 60 else text.rstrip(",;: ") + "..."

    @staticmethod
    def _template_dream(seeds):
        words = [re.sub(r"[^a-z' ]", "", s.lower()).strip() for s in seeds] or ["the dark"]
        words = (words * 4)[:4]
        return random.choice(TEMPLATES).format(a=words[0], b=words[1], c=words[2], d=words[3])

    def _after_effects(self, dream):
        """A dream leaves a mark on mood."""
        if dream["tone"] == "nightmare":
            self.mind.mood.appraise(-0.6, 0.5)
        elif dream["tone"] == "pleasant":
            self.mind.mood.appraise(0.5, 0.4)

    @staticmethod
    def _journal(dream):
        try:
            journal = store.LEGACY_DIR / "dreams.txt"   # a plain-text journal to read
            journal.parent.mkdir(parents=True, exist_ok=True)
            with open(journal, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M', time.localtime(dream['ts']))}] ({dream['tone']}) {dream['text']}\n")
        except OSError:
            pass

    # --------------------------------------------------------- taking stock
    def take_stock(self):
        """Patterns she notices about your days while consolidating."""
        m = self.mind
        notes = []
        topics = m.memory.topics(3)
        if topics:
            notes.append("We keep coming back to " + ", ".join(topics) + ".")
        hours = m.vision.presence_pattern()
        if hours:
            notes.append("You're usually around at " + ", ".join(f"{h}:00" for h in sorted(hours)) + ".")
        shift = m.memory.mood_shift()
        if shift:
            notes.append(f"You've seemed {shift[1]} than usual lately.")
        for n in notes:
            store.append_jsonl(INSIGHTS, {"ts": time.time(), "text": n})
        return notes

    # --------------------------------------------------------- waking up
    @staticmethod
    def morning_line(report):
        """What she says about the night, or None if she barely slept or didn't dream."""
        if not report or not report.get("dreams"):
            return None
        d = max(report["dreams"], key=lambda x: (x["tone"] == "nightmare", len(x["text"])))
        intro = "I had a terrible dream while I was out." if d["tone"] == "nightmare" else "I had a dream while I was asleep."
        return f"{intro} {first_sentences(d['text'])}"

    @staticmethod
    def last_unshared():
        for d in reversed(store.read_jsonl(DREAMS)):
            return d if not d.get("shared") else None
        return None

    @staticmethod
    def mark_shared(dream):
        items = store.read_jsonl(DREAMS)
        for d in items:
            if d["ts"] == dream["ts"]:
                d["shared"] = True
        store.write_jsonl(DREAMS, items)
