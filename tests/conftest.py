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
    """A valid serve-shaped derived block (cache_type q8_0, empty shape:
    measured with no extra flags)."""
    return {
        "model_mib": 10000,
        "kv_per_token_mib": 0.5,
        "slot_mib": 0.0,
        "compute_mib": 1000,
        "source": "llama-server",
        "cache_type": "q8_0",
        "shape": "",
    }


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Tests never touch the machine-local measurement artifacts
    (corrections, lock, journal) — one test wrote a garbage correction
    row into the live cache (2026-09-08, rep_stem "m")."""
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLAMA_PACKER_CORRECTIONS",
                       str(tmp_path / "cache" / "serve-corrections.yaml"))


def _hf_tree(tmp_path, repo="org/repo", revs=("abc123",),
             files=("model.gguf",), ref: str | None = "first", contents=None):
    """Fake HF hub cache: every rev in *revs* holds *files*, plus an
    optional ``refs/main`` pointer (``ref="first"`` points at ``revs[0]``,
    ``ref=None`` writes no pointer). *contents* maps ``(rev, file)`` to
    bytes. Returns the HF_HOME root (``tmp_path / "hf"``)."""
    if isinstance(revs, str):
        revs = (revs,)
    hub = tmp_path / "hf" / "hub"
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    for rev in revs:
        snap = repo_dir / "snapshots" / rev
        snap.mkdir(parents=True)
        for f in files:
            p = snap / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes((contents or {}).get((rev, f), b"x"))
    if ref == "first":
        ref = revs[0]
    if ref is not None:
        refs = repo_dir / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(ref)
    return tmp_path / "hf"
