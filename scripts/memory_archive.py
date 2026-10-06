#!/usr/bin/env python3
"""
memory_archive.py — pack DREAM's long-term text memory with her own compressor.

Her learned facts, milestones, opinions, goals and philosophy are text and JSON:
highly compressible, and they only grow. This packs them into a single .crush file
(via crush_codec, her vendored CRUSH compressor) so old memory can go to cold
storage without filling the disk — the same move sleep makes on memory, keeping the
gist and dropping the redundancy.

    python memory_archive.py pack                 -> memories/archive_<date>.crush
    python memory_archive.py restore <file> [dir]
    python memory_archive.py test <file>

Nothing is deleted. Archiving is a copy; you decide when to clear the originals.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crush_codec as cc

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEM = os.path.join(BASE, "memories")


def _memory_files():
    """DREAM's long-term text memory: facts, milestones, and her inner-life state."""
    paths = []
    for name in ("memories.txt", "mymilestones.txt"):
        p = os.path.join(MEM, name)
        if os.path.exists(p):
            paths.append(p)
    mind = os.path.join(MEM, "mind")
    if os.path.isdir(mind):
        for f in sorted(os.listdir(mind)):
            if f.endswith(".json"):
                paths.append(os.path.join(mind, f))
    return paths


def do_pack():
    files = _memory_files()
    if not files:
        print("no memory files found yet.")
        return
    out = os.path.join(MEM, f"archive_{time.strftime('%Y%m%d_%H%M%S')}.crush")
    total_in, arc = cc.pack(files, out)
    saved = 100.0 * (1 - arc / total_in) if total_in else 0.0
    print(f"packed {len(files)} file(s): {total_in} -> {arc} bytes ({saved:.1f}% saved)")
    print(f"  {out}")
    print("  (originals left in place — remove them yourself when you're ready)")


# ── automatic consolidation: compress when memory gets too much or too old ────
SIZE_LIMIT = 256 * 1024      # memories.txt past this is "too much"
AGE_DAYS = 120               # memory entries older than this are "too old" to keep live


def _entry_time(line):
    # memories.txt lines look like: [2026-09-21 14:05] (kind) text
    try:
        return time.mktime(time.strptime(line[1:17], "%Y-%m-%d %H:%M"))
    except Exception:
        return None


def auto_archive(size_limit=SIZE_LIMIT, age_days=AGE_DAYS, trim=True):
    """Archive + consolidate when the append-log grows too big or too old.

    Non-destructive first: everything (facts, milestones, mind state) is packed
    into a verified .crush snapshot. Only then, and only if trim=True, is the live
    memories.txt pruned — old or surplus entries drop out (they're safe in the
    archive). Milestones and the mind JSONs are never trimmed. Safe to call every
    wake-up; it does nothing until a threshold is crossed. Returns the archive path
    or None.
    """
    mt = os.path.join(MEM, "memories.txt")
    if not os.path.exists(mt):
        return None
    size = os.path.getsize(mt)
    with open(mt, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    now = time.time()
    cutoff = now - age_days * 86400
    too_old = any((_entry_time(l) is not None and _entry_time(l) < cutoff) for l in lines)
    too_much = size > size_limit
    if not (too_old or too_much):
        return None

    files = _memory_files()
    out = os.path.join(MEM, f"archive_{time.strftime('%Y%m%d_%H%M%S')}.crush")
    cc.pack(files, out)
    if not cc.test(out):                       # never trim against a bad archive
        try:
            os.remove(out)
        except OSError:
            pass
        return None

    if trim:
        # Keep entries that are recent; drop the old ones (preserved in the archive).
        keep = [l for l in lines if not (_entry_time(l) is not None and _entry_time(l) < cutoff)]
        # Still too big purely by size? keep the most recent half.
        if len("".join(keep).encode("utf-8")) > size_limit:
            keep = keep[len(keep) // 2:]
        if len(keep) < len(lines):
            with open(mt, "w", encoding="utf-8") as f:
                f.writelines(keep)
    return out


def do_restore(archive, dest):
    files = cc.unpack(archive, dest)
    print(f"restored {len(files)} file(s) to {dest} (all CRC-verified)")


def do_test(archive):
    ok = cc.test(archive)
    print("archive OK." if ok else "archive FAILED integrity check.")
    return ok


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    cmd = argv[1]
    if cmd == "pack":
        do_pack()
    elif cmd == "auto":
        out = auto_archive()
        print(f"consolidated -> {out}" if out else "nothing to consolidate yet.")
    elif cmd == "restore" and len(argv) >= 3:
        do_restore(argv[2], argv[3] if len(argv) > 3 else os.path.join(MEM, "restored"))
    elif cmd == "test" and len(argv) >= 3:
        return 0 if do_test(argv[2]) else 2
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
