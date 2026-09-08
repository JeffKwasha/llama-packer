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
    subprocess is invoked because tests seed a ``llama-server`` block.
    """

    def _make(stem: str = "test", **frontmatter) -> Model:
        gguf = tmp_path / f"{stem}.gguf"
        gguf.write_bytes(b"dummy")
        md_path = tmp_path / f"{stem}.md"
        fm: dict = {"name": stem, "context_length": 32768}
        fm.update(frontmatter)
        model = Model(md_path, fm)
        model.resolve_companions()
        return model

    return _make


@pytest.fixture
def fit_params_block():
    """A valid serve-shaped derived block (cache_type q8_0)."""
    return {
        "model_mib": 10000,
        "kv_per_token_mib": 0.5,
        "slot_mib": 0.0,
        "compute_mib": 1000,
        "source": "llama-server",
        "cache_type": "q8_0",
    }


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Tests never touch the machine-local measurement artifacts
    (corrections, lock, journal) — one test wrote a garbage correction
    row into the live cache (2026-09-08, rep_stem "m")."""
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
