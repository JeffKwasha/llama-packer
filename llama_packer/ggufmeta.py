"""Minimal GGUF metadata reader for vision geometry.

Reads only the key/value header — a few kilobytes at the start of the file,
never the tensors — so it is cheap even on platter storage. The GGUF KV
format is small and stable, so this needs no dependency; pulling in a full
GGUF library to read four integers would not earn its keep.

What we actually need lives in the mmproj (the vision projector's GGUF):

    clip.vision.patch_size          — side of one ViT patch, e.g. 16
    clip.vision.spatial_merge_size  — patches merged per output token, e.g. 2
    clip.projector_type             — e.g. qwen3vl_merger, minicpmv4_6

For a pixel-proportional projector (the Qwen-VL family) each output token
covers ``(patch_size * spatial_merge_size) ** 2`` pixels, which is how a
megapixel figure becomes a token count. Projectors that resample to a fixed
token count (MiniCPM-V, Gemma 3, LLaVA) do not carry a merge size, and the
megapixel knob cannot be honoured for them — see docs/models/vision-context.md.
"""
from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_MAGIC = b"GGUF"

# GGUF metadata value type tags.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY = 6, 7, 8, 9
_UINT64, _INT64, _FLOAT64 = 10, 11, 12

_SCALAR = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4), _BOOL: ("<B", 1),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8), _FLOAT64: ("<d", 8),
}


class _Reader:
    """Sequential reader over the GGUF header."""

    def __init__(self, fh):
        self._fh = fh

    def raw(self, n: int) -> bytes:
        b = self._fh.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def value(self, t: int):
        if t in _SCALAR:
            fmt, size = _SCALAR[t]
            v = struct.unpack(fmt, self.raw(size))[0]
            return bool(v) if t == _BOOL else v
        if t == _STRING:
            return self.string()
        if t == _ARRAY:
            et = self.u32()
            return [self.value(et) for _ in range(self.u64())]
        raise ValueError(f"unknown GGUF value type {t}")


def read_metadata(path: Path | str) -> dict:
    """Return every key/value in a GGUF file's header (tensors are skipped)."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        if r.raw(4) != _MAGIC:
            raise ValueError("not a GGUF file")
        r.u32()                      # version
        r.u64()                      # tensor count
        n_kv = r.u64()
        return {r.string(): r.value(r.u32()) for _ in range(n_kv)}


@dataclass(frozen=True)
class VisionGeometry:
    """How many pixels one image token covers, when that is well defined."""

    patch_size: int
    merge_size: int
    projector_type: str | None

    @property
    def pixels_per_token(self) -> int:
        return (self.patch_size * self.merge_size) ** 2

    @property
    def proportional(self) -> bool:
        """Whether token count scales with pixel area.

        True for projectors that merge a fixed grid of patches (Qwen-VL
        family, signalled by ``spatial_merge_size``). Resampler/fixed-size
        projectors carry no merge size and produce a roughly constant token
        count regardless of resolution, so a megapixel budget means nothing
        to them.
        """
        return self.merge_size > 1


@lru_cache(maxsize=256)
def vision_geometry(path_str: str) -> VisionGeometry | None:
    """Read the vision geometry from an mmproj GGUF, or None if unavailable."""
    try:
        md = read_metadata(path_str)
    except (OSError, ValueError) as exc:
        logger.warning("mmproj metadata: %s: %s", path_str, exc)
        return None
    patch = md.get("clip.vision.patch_size")
    if not isinstance(patch, int) or patch <= 0:
        return None
    merge = md.get("clip.vision.spatial_merge_size")
    if not isinstance(merge, int) or merge <= 0:
        merge = 1
    proj = md.get("clip.projector_type")
    return VisionGeometry(patch_size=patch, merge_size=merge,
                          projector_type=proj if isinstance(proj, str) else None)
