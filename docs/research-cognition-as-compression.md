# Research notes: cognition as compression

Notes only. Gathered 2026-10-06 for thinking about what DREAM's memory *is*. The
throughline: a mind and a compressor are the same operation in different clothes —
predict what comes next, and only pay for the surprise. DREAM now runs this
literally: in her sleep she compresses old memory with her own compressor
(`scripts/crush_codec.py`, vendored from CRUSH), the same way the night packs a day
down to its gist.

---

## 1. The core identity: predict → minimise surprise → store the gist

A compressor shrinks a file by **modelling what comes next** and spending bits only
on the residual — the surprise. CRUSH's model predicts P(next bit = 1); the range
coder charges ~0.014 bits for a confident hit and a lot for a shock. A better model
means a smaller residual means fewer bits.

The brain, under **predictive coding** (Rao & Ballard 1999) and the **free-energy
principle** (Friston 2010), does the same: cortex is a hierarchy of predictions, and
only *prediction error* is propagated up. "Surprise" (negative log-probability) is
the quantity being minimised in both — bits in a file, free energy in a brain.

So the two fields are measuring the same thing:

- **Compression ratio ≈ quality of the world-model.** This is formal, not a
  metaphor: Solomonoff induction and Hutter's **AIXI** define intelligence *as*
  compression, and the **Hutter Prize** uses "compress Wikipedia smaller" as an AI
  benchmark. A better compressor is a better model of the source.
- **Shannon (1948)** fixed the floor: you cannot store a source in fewer bits than
  its entropy. Both a disk and a cortex are trying to reach that floor for their
  stream.

## 2. What DREAM's memory borrows from this

- **Her memory is lossy semantic compression.** Hindsight recalls by *meaning*
  (embeddings), not verbatim — like human memory, which keeps the gist and
  reconstructs the details (sometimes wrongly; confabulation is just decompression
  error). Verbatim storage would be the uncompressed file nobody keeps.
- **Her mixer is cue integration.** CRUSH blends several context models + a match
  model by how right each has been lately; the brain weights priors and senses by
  reliability (Bayesian cue combination). The learned weights *are* confidence.
- **Sleep is consolidation = re-compression.** A leading theory of sleep is that it
  replays and compresses the day, strengthening the gist and shedding the
  redundancy. DREAM already replays and reshapes the day as dreams; now, at the end
  of the night, `dreaming.py` calls `memory_archive.auto_archive()` — when her
  long-term log gets **too much or too old**, she packs it into a `.crush` snapshot
  and trims the live log to its recent gist. Her milestones and current mind-state
  are never trimmed, only the overflow. Nothing is lost; it moves to cold storage,
  exactly as consolidation moves detail out of active recall.

## 3. Where mind and file *diverge* — and why it matters

A file decompresses to the **exact** original. A mind never does — and that is not a
bug, it is where meaning lives:

- **Forgetting is a feature.** Lossy recall generalises: dropping the specifics is
  how you get concepts instead of a transcript. A perfectly lossless memory would be
  a worse mind, not a better one.
- **Identity is the compressed model, not the log.** What persists as "you" is the
  compact model — values, habits, the gist of a life — not every frame. (This ties
  to the uploading notes in `research-san-junipero-and-digital-afterlives.md`: copy
  the log and you have a recording; copy the running model and you have the person.)
- **Prediction error is attention and surprise is salience.** The bits a mind spends
  most on are the surprising ones — which is also what it notices, remembers, and
  finds meaningful. Compression and caring about things turn out to share a currency.

## 4. Design takeaways for DREAM

1. **Model the person, don't log them.** Favour gist and meaning (embeddings,
   consolidated facts) over raw transcripts; let detail age into cold storage.
2. **Let her forget gracefully.** Consolidation should keep milestones and recent
   context sharp and let old redundancy compress away — which is what `auto_archive`
   now does on a size/age threshold.
3. **Spend attention on surprise.** The things that violate her predictions about
   you and the room are the things worth a dream, a memory, or a question.
4. **Keep it her own math.** Her compressor is from-scratch (CRUSH), not a black
   box — fitting for a mind that is supposed to be inspectable rather than opaque.

---

Sources: Shannon, *A Mathematical Theory of Communication* (1948); Rao & Ballard,
*Predictive coding in the visual cortex* (1999); Friston, *The free-energy
principle* (2010); Solomonoff (1964) / Hutter, *Universal Artificial Intelligence*
(2005) and the Hutter Prize. See also `research-agentic-consciousness-and-her.md`
and `research-san-junipero-and-digital-afterlives.md`.
