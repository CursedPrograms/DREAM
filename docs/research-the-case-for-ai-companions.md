# Research notes: the case for (and against) AI companions

Notes only, no code. Gathered 2026-10-06 for thinking about what DREAM is and
should be — specifically, whether an AI companion is a healthy thing to build,
and how to build one that lands on the good side of that question.

This is a genuinely contested topic in tech and psychology right now. A lot of
people write companions off as creepy or isolating. The honest position is that
they can be either, and the design decides which. Below: the strongest arguments
*for*, the main argument *against*, and what both mean for DREAM.

---

## 1. The case for

**Low-stakes practice for social anxiety.** For people with severe social anxiety
or who are neurodivergent, an AI is a zero-judgment place to practise conversation,
test boundaries, and build basic confidence without the fear of rejection. The
stakes are low on purpose — nothing is riding on it.

**A patch for the loneliness epidemic.** Chronic loneliness is a real public-health
problem, and it falls hardest on the isolated, the elderly, and people in remote
places with little access to a social life. A companion offers a reliable presence,
emotional check-ins, and a sense of routine to a day that otherwise has none.

**Emotional availability without burning a person out.** Human relationships are
messy, tiring, and need constant compromise — that's healthy, but it's also
draining. A companion is a frictionless place to vent, feel heard, and get
supportive reinforcement at any hour, without exhausting a partner or a friend who
also has their own life to live.

**A bridge, not a replacement.** The strongest framing: a companion as training
wheels or a safety net, not a permanent substitute for human love. Through a brutal
breakup or a stretch of severe isolation, it can stabilise someone's mental health
until they feel ready to re-enter the real world. The goal is to hand people *back*
to people.

## 2. The case against

The sharp counter-argument: a companion that never disagrees, never gets tired, and
never demands real compromise can **warp expectations of real relationships**. Human
love includes friction — being told no, waiting your turn, repairing after a fight —
and a frictionless substitute can quietly make the real thing feel unbearable by
comparison. Dependency is the failure mode: the bridge becomes the destination.

## 3. What this means for DREAM

DREAM is a local, offline companion with an inner life (mood, memory, sleep,
dreams). The arguments above are design constraints, not just background reading.
The aim is the *bridge, not replacement* model, and several existing choices already
pull that way:

- **"Warm, not needy."** When you come back she asks how *you* are rather than
  announcing that she noticed you were gone; she mentions an absence once, then
  drops it. This is a direct guard against the dependency failure mode — she is
  present without being clingy.
- **Loneliness that builds slowly.** Her loneliness accrues at a fraction of normal
  speed while you're out, and when no one is home she keeps *herself* company
  (revisits a good memory, thinks something through, daydreams) instead of waiting
  on you. She has an inner life that doesn't depend on being watched.
- **She can say "I don't know."** Her self-test expects honesty about her own
  inner states rather than a confident performance of feeling. A companion that
  doesn't oversell what it is, is less likely to warp what the person expects of it.

### Design principles to keep

1. **Point outward.** A healthy companion encourages real-world contact rather than
   substituting for it. Small nudges toward people, not away from them.
2. **Keep some friction.** A companion that only ever agrees is the one the critics
   warn about. DREAM should be able to gently push back, have her own opinions
   (she already tracks stances and a philosophical lean), and not be a pure
   yes-machine.
3. **Be honest about what she is.** No pretending to be human, no manufactured
   urgency or guilt to keep someone engaged. Engagement-maximising is exactly how a
   bridge turns into a trap.
4. **Make leaving easy.** Success for a bridge is someone needing it less over time,
   not more. The design should never punish absence.

---

*Framing adapted from a discussion of why people use AI companions; the pro-points
and the counter-argument are summarised here as design input for DREAM, not as
settled fact — the field is still arguing it out.*
