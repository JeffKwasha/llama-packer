# tests/conftest.py
"""Shared fixtures for llama-packer unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_packer.model import Model


@pytest.fixture
def make_model(tmp_path):
    """Factory: build a Model backed by a dummy GGUF file in ``tmp_path``.

    The GGUF file is not a real GGUF, so ``gguf_context_length`` reads None and
    ``_design_ctx`` falls back to the sidecar ``context_length``.  No
    subprocess is invoked because tests seed a ``fit-params`` block.
    """

    def _make(stem: str = "test", **frontmatter) -> Model:
        gguf = tmp_path / f"{stem}.gguf"
        gguf.write_bytes(b"dummy")
        md_path = tmp_path / f"{stem}.md"
        fm: dict = {"name": stem, "context_length": 32768}
        fm.update(frontmatter)
        return Model(md_path, fm)

    return _make


@pytest.fixture
def fit_params_block():
    """A valid fit-params frontmatter block (cache_type q8_0, parallel 1)."""
    return {
        "model_mib": 10000,
        "ctx_factor": 0.5,
        "compute_mib": 1000,
        "source": "fit-params",
        "cache_type": "q8_0",
        "parallel": 1,
    }
