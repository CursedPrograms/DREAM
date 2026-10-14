# TODO

## Smarter brain: move off phi3:mini

DREAM still talks with `phi3:mini` (`LlmModel` in `config.json`), the weakest part of her
stack. Plan:

- [x] **Switch her voice model to `qwen3:4b-instruct-2507`** (`ollama pull qwen3:4b-instruct-2507-q4_K_M`,
      ~2.5 GB): the same Qwen3 4B as TINA, minus the always-on thinking. Smarter than phi3
      (reasoning, maths, instructions), about as fast, never reasons out loud, and works with
      her existing prompts as they are, so only `LlmModel` changes.
      Done 2026-10-14 (`LlmModel`, the defaults in dream_models.py / dream_mind/llm.py, the options list).
      Tested on her real prompts: chat replies in ~4 s (phi3: 9-13 s, and it kept writing extra
      "Human: / You:" turns), in character, never thinking out loud; dreams and the reflection JSON
      come out right. It likes `*emphasis*` and emoji, which Piper reads out ("smirking face"), so
      `ask_llm` strips them (SPOKEN_JUNK in app.py and scripts/dream.py).
- [x] **Hand TINA's kind of work to TINA.** For multi-step tasks and tools, DREAM sends the task
      to TINA (`http://127.0.0.1:5013/`) instead of copying her code, and **speaks only TINA's
      final answer**.
      Done 2026-10-14: scripts/dream_fleet.py. "Hey TINA, ...", "ask TINA to ...", code / repo jobs and
      "why isn't X working" go to TINA's `POST /task`; DREAM follows `GET /task/<id>` and speaks only
      `result.answer` (no code blocks or markdown, cut at ~600 characters). TINA's questions are passed
      on for a spoken yes / no. `HandToTina` in config.json turns the automatic part off.
- [x] **"Show thoughts" toggle in the app**, off by default: TINA's reasoning (and anything else
      DREAM thinks) shown on screen when switched on. **Reasoning never goes to Piper.**
      Done 2026-10-14: `T` in the desktop app (a panel bottom left), 💭 on the browser page
      (dream.html, remembered per browser), `ShowThoughts` in config.json for the start. Shows TINA's
      reasoning, her tool calls and what RIFT / the robots answered.

### Why not TINA's `qwen3:4b` for DREAM's voice too

Tested 2026-10-13 on the RTX 2060 (Ollama):

- `qwen3:4b` is Qwen's thinking-only 2507 build (`thinking` capability, on by default).
  `"think": false` does **not** stop it: it reasons in the reply text instead
  (*"Hmm, the user is asking me…"*), which Piper would read out.
- `"think": true` on `/api/chat` with proper system/user messages separates the two cleanly
  (reasoning in `message.thinking`, answer in `message.content`), about 3 s per short reply.
- But with DREAM's one-string `System: … Human: … You:` prompt on `/api/generate` it over-thinks:
  1,174 tokens of reasoning, **no answer**, 14.5 s. Her dreaming and reflection use that style
  too, so the thinking model would need all of `scripts/dream_mind/llm.py` and its callers
  rewritten for chat messages plus a separate thinking budget. The instruct model avoids all of it.
