"""GGUF metadata and tensor-footprint reads, via the public ``gguf`` module.

Every GGUF parse in this project goes through ``gguf.GGUFReader`` — the
well-maintained public module — so the byte-level format (KV value types,
tensor-info layout, offset semantics) is never hand-rolled.  The ``gguf``
module is the only GGUF reader/writer here.

What llama-packer needs from a GGUF header:

    clip.vision.patch_size / spatial_merge_size / projector_type   — mmproj
    general.architecture / <arch>.context_length                   — discovery
    every tensor name carrying ``nextn``                           — MTP detection
    the data-section span (``file_size − data_offset``)           — weight term
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    from gguf import GGUFReader
except ImportError:  # pragma: no cover - gguf is a declared dependency
    GGUFReader = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_GGUF_WARNED = False


def _gguf_available() -> bool:
    global _GGUF_WARNED
    if GGUFReader is None and not _GGUF_WARNED:
        _GGUF_WARNED = True
        logger.warning("gguf module missing — pip install gguf")
    return GGUFReader is not None


_READER_CACHE: dict[tuple[str, int], "GGUFReader | None"] = {}


def gguf_reader(path: Path | str) -> GGUFReader | None:
    """Open *path* as a GGUF, or None when it is not a readable GGUF.

    ``GGUFReader`` raises a handful of error types for a truncated or
    non-GGUF file; the caller-facing contract is a single None.  The reader
    is cached per (path, mtime) so a model's header is parsed only once per
    run even when several helpers (context length, arch probe, MTP detection,
    tensor footprint) read it.
    """
    if not _gguf_available():
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns)
    if key not in _READER_CACHE:
        try:
            _READER_CACHE[key] = GGUFReader(str(path))
        except Exception:
            _READER_CACHE[key] = None
    return _READER_CACHE[key]


def read_metadata(path: Path | str) -> dict:
    """Return every metadata KV pair as plain Python values.

    The synthetic ``GGUF.version`` / ``GGUF.tensor_count`` fields the reader
    exposes are header fields, not file KV pairs, so they are excluded.
    Raises ValueError for a non-GGUF or unreadable file — the historical
    contract of this function.
    """
    reader = gguf_reader(path)
    if reader is None:
        raise ValueError(f"cannot read GGUF header: {path}")
    return {key: field.contents() for key, field in reader.fields.items()
            if not key.startswith("GGUF.")}


def tensor_names(path: Path | str) -> list[str]:
    """Every tensor name in the file, in header order (metadata only)."""
    reader = gguf_reader(path)
    if reader is None:
        raise ValueError(f"cannot read GGUF header: {path}")
    return [tensor.name for tensor in reader.tensors]


def gguf_tensor_bytes(path: Path | str) -> int | None:
    """On-disk byte footprint of the file's tensors, or None.

    ``GGUFReader.data_offset`` is the start of the tensor data section, so
    the footprint is ``file_size − data_offset``: the span the loader maps,
    inter-tensor padding included, with no GGUF type-size table needed.

    This is the ``model_mib`` term.  ``llama-fit-params`` reports a
    header-derived *logical* size that drops layers it does not count
    (baked-in MTP/nextn heads and the CPU-mapped split), under-reporting the
    real weight buffer by ~1.3 GB on such models (measured: fit 18569 vs
    real 19899.31 MiB on Dirk).
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return None
    reader = gguf_reader(p)
    if reader is None:
        return None
    return max(0, size - reader.data_offset)


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
