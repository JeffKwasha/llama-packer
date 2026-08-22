# tests/test_utils_parsers.py
"""Memory string parsers."""

from __future__ import annotations

import pytest

from llama_packer.utils import parse_context_length, parse_mem_mb


def test_resolve_spare_suffixed_gigabytes():
    assert parse_mem_mb("2G", 32768) == 2048


def test_resolve_spare_suffixed_megabytes():
    assert parse_mem_mb("512m", 32768) == 512


def test_resolve_spare_bare_gb_hint():
    # bare number < 3 * VRAM(GB) -> treated as GB
    assert parse_mem_mb("2", 32768) == 2048


def test_resolve_spare_bare_mb_hint():
    # bare number >= 3 * VRAM(GB) -> treated as MB
    assert parse_mem_mb("512", 32768) == 512


def test_resolve_spare_invalid_returns_zero():
    assert parse_mem_mb("nonsense", 32768) == 0


def test_parse_context_length_k():
    assert parse_context_length("128k") == 131072


def test_parse_context_length_m():
    assert parse_context_length("1m") == 1048576


def test_parse_context_length_bare():
    assert parse_context_length("65536") == 65536


# ── HF cache grouping ─────────────────────────────────────────────────────


def test_compute_env_prefixes_hf_grouping(tmp_path, monkeypatch):
    from llama_packer.utils import compute_env_prefixes, hf_cache_root
    models = tmp_path / "models"
    hf = tmp_path / "hf"
    models.mkdir(); hf.mkdir()
    gguf = models / "a.gguf"
    gguf.write_bytes(b"x")
    ct = hf / "chat_template.jinja"
    ct.write_text("x")
    monkeypatch.setenv("HF_HOME", str(hf))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)

    _p2v, v2v = compute_env_prefixes([str(gguf), str(ct)])
    assert v2v["HF_HOME"] == hf_cache_root()
    assert v2v["MODELS_DIR"] == str(models)
    # The chat-template path must NOT widen MODELS_DIR up to tmp_path.
    assert v2v["MODELS_DIR"] == str(models)


def test_compute_env_prefixes_hf_home_override(tmp_path, monkeypatch):
    from llama_packer.utils import compute_env_prefixes
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # no default cache dir
    models = tmp_path / "models"
    hf = tmp_path / "custom_hf"
    models.mkdir(); hf.mkdir()
    gguf = models / "a.gguf"; gguf.write_bytes(b"x")
    ct = hf / "ct.jinja"; ct.write_text("x")

    _p2v, v2v = compute_env_prefixes([str(gguf), str(ct)], hf_home=str(hf))
    assert v2v["HF_HOME"] == str(hf)
    assert v2v["MODELS_DIR"] == str(models)
