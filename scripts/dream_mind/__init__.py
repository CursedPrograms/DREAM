"""
dream_mind - DREAM's inner life.

Functional analogues of the things that make an agent feel like someone is
home: moods, needs, episodic memory that fades and strengthens, a model of the
room and of the user, sleep that consolidates memory and produces dreams, a
running self-narrative, and initiative (she sometimes speaks first).

None of this makes her conscious; it makes her behave as though she has an
inner state, because she does have one - it's just numbers, files and prompts.
See mind.py for the entry point: get_mind().
"""



def __getattr__(name):
    # Imported lazily so `import dream_mind.affect` etc. stay light.
    if name in ("Mind", "get_mind"):
        from . import mind
        return getattr(mind, name)
    raise AttributeError(name)
