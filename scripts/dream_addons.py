"""
Optional add-ons. Each one is a folder in addons/ (not part of this repo, and
gitignored) holding an addon.py. Nothing here does anything until an add-on's
own installer has put one there. An addon.json beside it with "enabled": false
switches the add-on off without removing it (true, or no addon.json, is on).

An addon.py may define:
  pools(pools)  -> {state: [video paths]}: the states it plays its own clips for.
                   pools is {state: [video paths]} as DREAM found them; a state
                   it leaves out, or gives no clips that exist, keeps DREAM's.
  request(text) -> {"say": line to speak, "video": path of a clip to play once},
                   or None when the add-on has nothing for what was said.

    python scripts/dream_addons.py           list the installed add-ons, and which are off
"""

import hashlib
import importlib.util
import json
import os

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADDONS_DIR = os.path.join(ROOT_DIR, "addons")

_loaded = None   # [(name, module)], read once


def enabled(name):
    """False only when addons/<name>/addon.json says "enabled": false."""
    try:
        with open(os.path.join(ADDONS_DIR, name, "addon.json"), encoding="utf-8") as f:
            return json.load(f).get("enabled", True) is not False
    except (OSError, ValueError, AttributeError):
        return True


def loaded():
    global _loaded
    if _loaded is not None:
        return _loaded
    _loaded = []
    if not os.path.isdir(ADDONS_DIR):
        return _loaded
    for name in sorted(os.listdir(ADDONS_DIR)):
        path = os.path.join(ADDONS_DIR, name, "addon.py")
        if not os.path.isfile(path):
            continue
        if not enabled(name):
            print(f"Add-on: {name} (off: addon.json has \"enabled\": false)")
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"dream_addon_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Add-on {name}: could not load it: {e}")
            continue
        _loaded.append((name, module))
        print(f"Add-on: {name}")
    return _loaded


def pools(found):
    """found is {state: [video paths]}; returns it with the add-ons' clips in place."""
    out = {state: list(clips) for state, clips in found.items()}
    for name, module in loaded():
        fn = getattr(module, "pools", None)
        if fn is None:
            continue
        try:
            replaced = fn(out) or {}
        except Exception as e:
            print(f"Add-on {name}: pools failed: {e}")
            continue
        for state, clips in replaced.items():
            clips = [os.path.abspath(c) for c in clips if os.path.isfile(c)]
            if clips:
                out[state] = clips
    return out


def request(text):
    """The first add-on's answer to what was said: {"say", "video"} or None."""
    for name, module in loaded():
        fn = getattr(module, "request", None)
        if fn is None:
            continue
        try:
            answer = fn(text)
        except Exception as e:
            print(f"Add-on {name}: request failed: {e}")
            continue
        if answer and answer.get("video") and os.path.isfile(answer["video"]):
            answer["video"] = os.path.abspath(answer["video"])
            answer["say"] = answer.get("say") or ""
            return answer
    return None


# The web server plays add-on clips from wherever they are: each gets a URL
# /addon-videos/<token>/<name>, and app.py looks the token up here.
_served = {}


def url_for(path):
    path = os.path.abspath(path)
    token = hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]
    _served[token] = path
    return f"/addon-videos/{token}/{os.path.basename(path)}"


def served_path(token):
    return _served.get(token)


if __name__ == "__main__":
    if not loaded():
        print(f"No add-ons switched on in {ADDONS_DIR}")
