# tests/test_vram_budget.py
"""calc_ctx and solve_matrix_ctx budget math."""

from __future__ import annotations

import pytest

from llama_packer import utils
from llama_packer.vram import solve_matrix_ctx


# ── calc_ctx ──────────────────────────────────────────────────────────────


def test_calc_ctx_fits_design(make_model, fit_params_block):
    model = make_model("a", **{"fit-params": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # available = 32768 - 2048 = 30720; remaining = 30720 - 10000 - 1000 = 19720
    # ctx_at_design = 0.5*32768 = 16384 <= 19720 -> design context
    assert ctx == 32768


def test_calc_ctx_scales_down_and_rounds(make_model):
    fm = {"model_mib": 25000, "ctx_factor": 0.5, "compute_mib": 1000,
          "cache_type": "q8_0", "parallel": 1}
    model = make_model("b", **{"fit-params": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # remaining = 30720 - 26000 = 4720; 4720/0.5 = 9440 -> rounded to 8192
    assert ctx == 8192


def test_calc_ctx_floors_at_min(make_model):
    fm = {"model_mib": 30000, "ctx_factor": 0.5, "compute_mib": 0,
          "cache_type": "q8_0", "parallel": 1}
    model = make_model("c", **{"fit-params": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == utils._MIN_CTX_SIZE


def test_calc_ctx_applies_spare(make_model, fit_params_block):
    model = make_model("d", **{"fit-params": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=3072)
    # available = 32768 - 2048 - 3072 = 27648; remaining = 27648-11000 = 16648
    # ctx_at_design = 16384 <= 16648 -> design still fits
    assert ctx == 32768


def test_calc_ctx_cpu_resident_returns_design(make_model):
    model = make_model("e", device="cpu", context_length=8192)
    ctx = model.vram.calc_ctx(1024, fit_bin="unused")
    assert ctx == 8192


def test_calc_ctx_vllm_no_estimate_returns_design(make_model):
    # vLLM model with no estimator and no local safetensors: graceful fallback.
    model = make_model("f", backend="vllm", hf_repo="org/model",
                       context_length=65536)
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == 65536


# ── cache-type scaling ────────────────────────────────────────────────────


def test_saved_for_cache_type_mismatch_returns_none(make_model, fit_params_block):
    model = make_model("s", **{"fit-params": fit_params_block})
    assert model.vram.saved_for("q8_0", 1) is not None
    assert model.vram.saved_for("f16", 1) is None
    assert model.vram.saved_for("q8_0", 2) is None  # parallel mismatch too


def test_fit_params_static_scales_cache_type(make_model, fit_params_block):
    model = make_model("s", **{"fit-params": fit_params_block})
    fp = model.vram.fit_params_static("unused", cache_type="f16", parallel=1)
    assert fp is not None
    assert fp.source == "fit-params-scaled"
    assert fp.cache_type == "f16"
    # q8_0 -> f16 scales the KV factor by bytes/elem ratio 2.0 / 1.0625
    assert fp.ctx_factor == pytest.approx(0.5 * 2.0 / 1.0625)
    # weights and compute are cache-independent
    assert fp.model_mib == fit_params_block["model_mib"]
    assert fp.compute_mib == fit_params_block["compute_mib"]


# ── solve_matrix_ctx ──────────────────────────────────────────────────────


def test_solve_matrix_ctx_basic(make_model):
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768,
        spare_mb=0,
        chat_models=[(chat, 8000, 0.4, 500)],
        embed_params=(500, 0.1, 100),
        rerank_params=None,
        embed_ctx=8192,
        rerank_ctx=0,
    )
    # available = 30720; embed overhead = 500+100+0.1*8192 = 1419
    # remaining_for_chat = 29301; chat_budget = 29301-8000-500 = 20801
    # ctx = 20801/0.4 = 52002 -> round 49152 -> min(arch 32768) = 32768
    assert ctx == 32768


def test_solve_matrix_ctx_no_chat_models():
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0, chat_models=[],
        embed_params=None, rerank_params=None,
    )
    assert ctx == utils._MIN_CTX_SIZE


def test_solve_matrix_ctx_exhausted_budget(make_model):
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[(chat, 40000, 0.4, 0)],
        embed_params=None, rerank_params=None,
    )
    # chat_budget = 30720 - 40000 < 0 -> skipped -> best_ctx 0 -> MIN_CTX
    assert ctx == utils._MIN_CTX_SIZE
