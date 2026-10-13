"""
Reads mode.json in the project root.

xxx_version follows the add-on pack's key file, xxx.json in the project
root (not part of this repo). The key's "path" points at the pack folder;
a relative path is read from the project root, and with no path the pack
is expected in the project root itself. While the key is here and the pack
it points at is installed, xxx_version is "yes". Otherwise it is "no".
"""

import json
import os

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODE_PATH = os.path.join(ROOT_DIR, "mode.json")
KEY_PATH  = os.path.join(ROOT_DIR, "xxx.json")
XXX_KEY   = "DREAM_XXX"


def xxx_pack_dir():
    """The pack folder named by xxx.json, or None without a valid key and pack."""
    try:
        with open(KEY_PATH, encoding="utf-8") as f:
            key = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(key, dict) or key.get("key") != XXX_KEY:
        return None
    path = os.path.expanduser(str(key.get("path") or "."))
    pack = os.path.normpath(os.path.join(ROOT_DIR, path))
    images = os.path.join(pack, "images", "xxx")
    if not (os.path.isdir(images) and any(os.scandir(images))):
        print(f"xxx.json: no pack at {pack} (images/xxx is missing or empty)")
        return None
    return pack


def load_mode():
    try:
        with open(MODE_PATH, encoding="utf-8") as f:
            mode = json.load(f)
    except (OSError, ValueError):
        mode = {}
    if not isinstance(mode, dict):
        mode = {}

    wanted = "yes" if xxx_pack_dir() else "no"
    if mode.get("xxx_version") != wanted:
        print(f"mode.json: setting xxx_version to '{wanted}'")
        mode["xxx_version"] = wanted
        try:
            with open(MODE_PATH, "w", encoding="utf-8") as f:
                json.dump(mode, f, indent=2)
                f.write("\n")
        except OSError as e:
            print(f"mode.json: could not write it back: {e}")

    return mode


def xxx_enabled():
    return load_mode()["xxx_version"] == "yes"


if __name__ == "__main__":
    print(load_mode())
