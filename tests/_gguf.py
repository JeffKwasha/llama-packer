"""Test-only GGUF writer, via the public ``gguf`` module.

GGUF is never read or written by hand anywhere in this project — tests
included. This helper is the single place a synthetic GGUF is built, so the
byte-level format stays the ``gguf`` module's business.
"""
from __future__ import annotations

from pathlib import Path


def write_gguf(path, *, arch="qwen3", context_length=131072, kv=None,
               tensor_names=(), tensor_shape=(16, 16)):
    """Write a minimal GGUF and return *path*.

    *kv* adds metadata (str/int/float/bool values). *tensor_names* adds
    zero-filled f32 tensors so name-based readers (MTP detection) can be
    exercised. ``context_length=None`` omits the ``<arch>.context_length``
    key (used for mmproj fixtures).
    """
    import numpy as np
    from gguf import GGUFWriter

    path = Path(path)
    writer = GGUFWriter(str(path), arch=arch)
    if context_length is not None:
        writer.add_uint32(f"{arch}.context_length", context_length)
    for key, value in (kv or {}).items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            raise TypeError(f"unsupported kv value for {key}: {value!r}")
    for name in tensor_names:
        writer.add_tensor(name, np.zeros(tensor_shape, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
