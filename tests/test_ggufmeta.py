# tests/test_ggufmeta.py
"""GGUF header walks: the on-disk tensor footprint."""

from __future__ import annotations

from pathlib import Path

import pytest

from _gguf import write_gguf
from llama_packer.ggufmeta import gguf_tensor_bytes


def test_gguf_tensor_bytes_reports_data_span(tmp_path):
    p = tmp_path / "m.gguf"
    # two (16,16) f32 tensors = 2 * 1024 bytes of data section
    write_gguf(p, tensor_names=["t0.weight", "t1.weight"])
    assert gguf_tensor_bytes(p) == 2048


def test_gguf_tensor_bytes_single_tensor(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p, tensor_names=["t0.weight"])
    assert gguf_tensor_bytes(p) == 1024


def test_gguf_tensor_bytes_none_for_non_gguf(tmp_path):
    p = tmp_path / "m.gguf"
    p.write_bytes(b"not a gguf")
    assert gguf_tensor_bytes(p) is None


def test_gguf_tensor_bytes_none_for_missing(tmp_path):
    assert gguf_tensor_bytes(tmp_path / "nope.gguf") is None


def test_gguf_tensor_bytes_empty_tensor_list(tmp_path):
    p = tmp_path / "m.gguf"
    write_gguf(p)
    assert gguf_tensor_bytes(p) == 0


_DIRK = Path(
    "/mnt/ai/huggingface/hub/models--peculiar-ragdoll--Dirk-Qwen3.8-27B-GGUF"
    "/snapshots/fecc6f4207063d68b35e08248dbffb1bffc4e28b"
    "/Dirk-Qwen3.8-27B-UD-Q5_K_XL.gguf")


@pytest.mark.skipif(not _DIRK.exists(), reason="Dirk GGUF not present")
def test_gguf_tensor_bytes_real_dirk():
    tb = gguf_tensor_bytes(_DIRK)
    assert tb is not None
    mib = tb / (1024 * 1024)
    # fit-params reports 18569; the real weight buffer is ~19899 MiB
    # (Vulkan0 18904.68 + CPU_Mapped 994.63 from the -lv 5 log). The header
    # walk must land near the latter, not the former.
    assert 19800 < mib < 19920
