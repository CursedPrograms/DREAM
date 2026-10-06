"""
crush_codec.py — DREAM's own compressor, vendored from the CRUSH project.

Same idea as her mind: predict what comes next, and only pay for the surprise.
A binary range coder (arithmetic coding) drives an adaptive order-1 context model
that learns the data as it reads it. No zlib, no external libraries.

DREAM uses this to pack her long-term *text* memory — learned facts, milestones,
transcripts, mind snapshots — into a compact .crush archive (see memory_archive.py).
It's the same move sleep makes on memory: keep the gist, drop the redundancy.

API:
    compress(data: bytes) -> bytes
    decompress(comp: bytes, out_size: int) -> bytes
    pack(paths, archive_path)         # many files -> one .crush
    unpack(archive_path, dest=".")    # .crush -> files (CRC-checked)
    test(archive_path) -> bool        # integrity check
"""

import os
import struct
from array import array

# ── range coder + order-1 model (fpaq0 lineage) ──────────────────────────────
PBITS = 12
PSCALE = 1 << PBITS
PHALF = PSCALE >> 1
ADAPT = 5
CTX_SIZE = 256 * 256
MASK32 = 0xFFFFFFFF


class _Encoder:
    def __init__(self):
        self.x1 = 0
        self.x2 = MASK32
        self.out = bytearray()

    def encode(self, bit, p):
        xmid = self.x1 + (((self.x2 - self.x1) >> PBITS) * p)
        if bit:
            self.x2 = xmid
        else:
            self.x1 = xmid + 1
        while ((self.x1 ^ self.x2) & 0xFF000000) == 0:
            self.out.append((self.x2 >> 24) & 0xFF)
            self.x1 = (self.x1 << 8) & MASK32
            self.x2 = ((self.x2 << 8) & MASK32) | 0xFF

    def finish(self):
        for _ in range(4):
            self.out.append((self.x1 >> 24) & 0xFF)
            self.x1 = (self.x1 << 8) & MASK32
        return bytes(self.out)


class _Decoder:
    def __init__(self, data):
        self.x1 = 0
        self.x2 = MASK32
        self.data = data
        self.pos = 0
        self.x = 0
        for _ in range(4):
            self.x = ((self.x << 8) | self._get()) & MASK32

    def _get(self):
        if self.pos < len(self.data):
            b = self.data[self.pos]
            self.pos += 1
            return b
        return 0

    def decode(self, p):
        xmid = self.x1 + (((self.x2 - self.x1) >> PBITS) * p)
        if self.x <= xmid:
            bit = 1
            self.x2 = xmid
        else:
            bit = 0
            self.x1 = xmid + 1
        while ((self.x1 ^ self.x2) & 0xFF000000) == 0:
            self.x1 = (self.x1 << 8) & MASK32
            self.x2 = ((self.x2 << 8) & MASK32) | 0xFF
            self.x = ((self.x << 8) | self._get()) & MASK32
        return bit


def _model():
    return array('H', [PHALF]) * CTX_SIZE


def compress(data):
    enc = _Encoder()
    t = _model()
    prev = 0
    for byte in data:
        ctx = prev << 8
        node = 1
        for shift in (7, 6, 5, 4, 3, 2, 1, 0):
            bit = (byte >> shift) & 1
            idx = ctx | node
            p = t[idx]
            enc.encode(bit, p)
            t[idx] = p + ((PSCALE - p) >> ADAPT) if bit else p - (p >> ADAPT)
            node = (node << 1) | bit
        prev = byte
    return enc.finish()


def decompress(comp, out_size):
    dec = _Decoder(comp)
    t = _model()
    prev = 0
    out = bytearray(out_size)
    for i in range(out_size):
        ctx = prev << 8
        node = 1
        for _ in range(8):
            idx = ctx | node
            p = t[idx]
            bit = dec.decode(p)
            t[idx] = p + ((PSCALE - p) >> ADAPT) if bit else p - (p >> ADAPT)
            node = (node << 1) | bit
        byte = node & 0xFF
        out[i] = byte
        prev = byte
    return bytes(out)


# ── CRC32 (our own table, no zlib) ───────────────────────────────────────────
_CRC = []
for _n in range(256):
    _c = _n
    for _ in range(8):
        _c = (_c >> 1) ^ 0xEDB88320 if (_c & 1) else (_c >> 1)
    _CRC.append(_c)


def crc32(data):
    c = 0xFFFFFFFF
    for b in data:
        c = _CRC[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


# ── .crush container (compatible with the CRUSH project's v1 format) ─────────
MAGIC = b"CRUSH\x01"
RAW, CRUSHED = 0, 1


def pack(paths, archive_path):
    total_in = total_out = 0
    with open(archive_path, "wb") as w:
        w.write(MAGIC)
        w.write(struct.pack("<I", len(paths)))
        for path in paths:
            with open(path, "rb") as f:
                data = f.read()
            comp = compress(data)
            method, stored = (CRUSHED, comp) if len(comp) < len(data) else (RAW, data)
            name = os.path.basename(path).encode("utf-8")
            w.write(struct.pack("<B", method))
            w.write(struct.pack("<H", len(name)))
            w.write(name)
            w.write(struct.pack("<Q", len(data)))
            w.write(struct.pack("<I", crc32(data)))
            w.write(struct.pack("<Q", len(stored)))
            w.write(stored)
            total_in += len(data)
            total_out += len(stored)
    return total_in, os.path.getsize(archive_path)


def _entries(r):
    if r.read(len(MAGIC)) != MAGIC:
        raise ValueError("not a CRUSH archive")
    (count,) = struct.unpack("<I", r.read(4))
    for _ in range(count):
        (method,) = struct.unpack("<B", r.read(1))
        (nlen,) = struct.unpack("<H", r.read(2))
        name = r.read(nlen).decode("utf-8")
        (osize,) = struct.unpack("<Q", r.read(8))
        (crc,) = struct.unpack("<I", r.read(4))
        (ssize,) = struct.unpack("<Q", r.read(8))
        stored = r.read(ssize)
        yield method, name, osize, crc, stored


def _restore(method, osize, stored):
    return stored if method == RAW else decompress(stored, osize)


def unpack(archive_path, dest="."):
    os.makedirs(dest, exist_ok=True)
    restored = []
    with open(archive_path, "rb") as r:
        for method, name, osize, crc, stored in _entries(r):
            data = _restore(method, osize, stored)
            if crc32(data) != crc or len(data) != osize:
                raise ValueError(f"CRC mismatch on {name} — archive corrupt")
            out = os.path.join(dest, os.path.basename(name))
            with open(out, "wb") as w:
                w.write(data)
            restored.append(out)
    return restored


def test(archive_path):
    with open(archive_path, "rb") as r:
        for method, name, osize, crc, stored in _entries(r):
            data = _restore(method, osize, stored)
            if crc32(data) != crc or len(data) != osize:
                return False
    return True
