# tests/test_companions.py
"""Companion (mmproj / MTP draft) VRAM folding via VramBudget.effective_static."""

from __future__ import annotations

import pytest

from llama_packer.vram import _DRAFT_COMPUTE_MB, _DRAFT_CTX_SAFETY, _MMPROJ_COMPUTE_MB

MIB = 1024 * 1024


@pytest.fixture
def fit_params_block():
    return {
        "model_mib": 10000,
        "ctx_factor": 0.5,
        "compute_mib": 1000,
        "source": "fit-params",
        "cache_type": "q8_0",
        "parallel": 1,
    }


def test_effective_static_folds_mmproj_and_mtp(tmp_path, make_model, fit_params_block):
    # Companion files must exist before Model construction resolves them.
    (tmp_path / "main-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    (tmp_path / "main.mtp.gguf").write_bytes(b"x" * 2 * MIB)
    m = make_model("main",
                   **{"fit-params": dict(fit_params_block),
                      "mmproj": "main-mmproj.gguf",
                      "speculative": "main.mtp.gguf"})

    model_mib, ctx_factor, compute_mib = m.vram.effective_static(
        fit_bin="unused", cache_type="q8_0", parallel=1)

    assert model_mib == 10000 + 3 + 2
    assert compute_mib == 1000 + _MMPROJ_COMPUTE_MB + _DRAFT_COMPUTE_MB
    # Draft KV factor: main factor scaled by size ratio, padded by safety.
    expected_draft = 0.5 * (2 / 10000) * _DRAFT_CTX_SAFETY
    assert ctx_factor == pytest.approx(0.5 + expected_draft)


def test_effective_static_mmproj_only_has_zero_kv_factor(tmp_path, make_model,
                                                         fit_params_block):
    (tmp_path / "solo-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    m = make_model("solo", **{"fit-params": dict(fit_params_block),
                              "mmproj": "solo-mmproj.gguf"})
    model_mib, ctx_factor, compute_mib = m.vram.effective_static(
        fit_bin="unused", cache_type="q8_0", parallel=1)
    assert model_mib == 10003
    assert ctx_factor == 0.5          # mmproj adds no per-token cost
    assert compute_mib == 1000 + _MMPROJ_COMPUTE_MB


def test_effective_static_vllm_skips_companions(tmp_path, make_model,
                                                fit_params_block):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    m = make_model("v", backend="vllm", hf_repo="org/model",
                   mmproj="v-mmproj.gguf",
                   **{"fit-params": dict(fit_params_block)})
    assert m.vram.effective_static(fit_bin="unused") == (10000, 0.5, 1000)


def test_effective_static_result_is_cached(tmp_path, make_model,
                                           fit_params_block, monkeypatch):
    (tmp_path / "c1-mmproj.gguf").write_bytes(b"x" * 2 * MIB)
    m = make_model("c1", **{"fit-params": dict(fit_params_block),
                            "mmproj": "c1-mmproj.gguf"})

    calls = {"n": 0}
    real = m.vram._companion_fit

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(m.vram, "_companion_fit", counting)
    first = m.vram.effective_static(fit_bin="unused")
    second = m.vram.effective_static(fit_bin="unused")
    assert first == second
    assert calls["n"] == 1  # second call served from _effective_cache
