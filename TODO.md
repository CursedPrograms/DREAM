# TODO

## Smarter brain: move off phi3:mini

DREAM still talks with `phi3:mini` (`LlmModel` in `config.json`), the weakest part of her
stack. Plan:

- [ ] **Switch her voice model to `qwen3:4b-instruct-2507`** (`ollama pull qwen3:4b-instruct-2507-q4_K_M`,
      ~2.5 GB): the same Qwen3 4B as TINA, minus the always-on thinking. Smarter than phi3
      (reasoning, maths, instructions), about as fast, never reasons out loud, and works with
      her existing prompts as they are, so only `LlmModel` changes.
- [ ] **Hand TINA's kind of work to TINA.** For multi-step tasks and tools, DREAM sends the task
      to TINA (`http://127.0.0.1:5013/`) instead of copying her code, and **speaks only TINA's
      final answer**.
- [ ] **"Show thoughts" toggle in the app**, off by default: TINA's reasoning (and anything else
      DREAM thinks) shown on screen when switched on. **Reasoning never goes to Piper.**

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
