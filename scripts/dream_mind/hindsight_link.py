"""
hindsight_link.py - DREAM's memories, also kept in Hindsight (bank "dream").

Her own memory (memory.py) stays the one she relies on. Hindsight, the memory
server she shares with TINA, adds recall by meaning across everything you've
ever told her, and it learns facts about you from it. So:

  each exchange   is also sent to Hindsight - from a background queue, never
                  slowing a reply - with the episode's id as its document id
  answering you   she asks Hindsight what it remembers, but gives up after
                  RECALL_TIMEOUT_S: a slow or missing server never delays her
  forgetting      "forget what I said about X" deletes those documents there
                  too, and "forget everything" deletes her whole bank

Off unless config.json has Config.DREAM.Memory.Hindsight = true (the self-test
leaves it off).

DREAM hosts the server herself (memory_server.py, started by run.bat), so
HindsightUrl is normally localhost.
"""

import asyncio
import json
import queue
import threading
import time

from . import store

RECALL_TIMEOUT_S = 1.5
ENABLED = True            # selftest turns this off


def _settings():
    try:
        with open(store.ROOT / "config.json", encoding="utf-8") as f:
            m = json.load(f)["Config"]["DREAM"].get("Memory", {})
    except (OSError, ValueError, KeyError):
        m = {}
    return bool(m.get("Hindsight", False)), m.get("HindsightUrl", "http://localhost:8888"), m.get("Bank", "dream")


class HindsightLink:
    def __init__(self):
        on, self.url, self.bank = _settings()
        self.on = on and ENABLED
        self._client = None
        self._down_until = 0.0
        self._jobs = queue.Queue()
        self._forgotten = set()        # episodes forgotten while their upload was still queued
        if self.on:
            threading.Thread(target=self._worker, daemon=True, name="dream-hindsight").start()

    def _get(self):
        if self._client is None:
            try:
                from hindsight_client import Hindsight
            except ImportError:
                self.on = False          # the client isn't installed: quietly stay on her own memory
                return None
            self._client = Hindsight(base_url=self.url, timeout=30)
        return self._client

    def _worker(self):
        while True:
            job = self._jobs.get()
            if time.time() < self._down_until:
                time.sleep(max(0.0, self._down_until - time.time()))
            try:
                job()
            except Exception:
                self._down_until = time.time() + 60     # server away: try again in a minute
                self._jobs.put(job)                       # nothing is dropped

    # -- storing
    def retain_episode(self, episode):
        if not self.on:
            return
        content = f"They said: {episode['user']}\nDREAM answered: {episode.get('reply', '')}"
        when = episode.get("ts", time.time())

        def job():
            if str(episode["id"]) in self._forgotten:
                return
            c = self._get()
            if c:
                from datetime import datetime
                c.retain(bank_id=self.bank, content=content, document_id=str(episode["id"]), retain_async=True,
                         timestamp=datetime.fromtimestamp(when), context="a conversation with DREAM")
        self._jobs.put(job)

    def retain_text(self, text, context):
        if not self.on:
            return

        def job():
            c = self._get()
            if c:
                c.retain(bank_id=self.bank, content=text, context=context, retain_async=True)
        self._jobs.put(job)

    # -- remembering
    def recall(self, query, k=2):
        """A few things Hindsight remembers about this, or [] - never slower than RECALL_TIMEOUT_S."""
        if not self.on or time.time() < self._down_until:
            return []
        out = []

        def ask():
            try:
                c = self._get()
                if c:
                    res = c.recall(bank_id=self.bank, query=query, max_tokens=600, budget="low")
                    out.extend(r.text for r in (res.results or [])[:k] if getattr(r, "text", None))
            except Exception:
                self._down_until = time.time() + 60

        t = threading.Thread(target=ask, daemon=True)
        t.start()
        t.join(RECALL_TIMEOUT_S)
        return list(out) if not t.is_alive() else []

    # -- forgetting
    def forget_episodes(self, episode_ids):
        if not self.on or not episode_ids:
            return
        self._forgotten.update(str(e) for e in episode_ids)   # anything still queued for them won't be sent

        def job():
            c = self._get()
            if c:
                async def drop():
                    for eid in episode_ids:
                        try:
                            await c.documents.delete_document(bank_id=self.bank, document_id=str(eid))
                        except Exception:
                            pass
                asyncio.run(drop())
        self._jobs.put(job)

    def forget_all(self):
        if not self.on:
            return
        while True:                       # nothing queued before "forget everything" may be sent after it
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break

        def job():
            c = self._get()
            if c:
                c.delete_bank(self.bank)
        self._jobs.put(job)
