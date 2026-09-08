# tests/test_vram_budget.py
"""calc_ctx and solve_matrix_ctx budget math under the affine VRAM law:

    VRAM(C, p) = model_mib + compute_mib + c*C + p*D

with C the total KV pool (= per-slot context X times parallel) and X the
solved, emitted value.
"""

from __future__ import annotations

import pytest

from llama_packer.consts import _MIN_CTX_SIZE
from llama_packer.vram import FitParams, parse_device_buffers, solve_matrix_ctx


# ── serve-shaped log parsing ──────────────────────────────────────────────

def test_parse_device_buffers_sums_device_only():
    """Real Dirk (-text) numbers: weights max, both contexts' KV/RS/compute,
    host-visible buffers excluded."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size =     0.00 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =   994.63 MiB
0.24.044 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB
0.29.025 I llama_kv_cache:    Vulkan0 KV buffer size =  1024.00 MiB
0.28.115 I llama_memory_recurrent:    Vulkan0 RS buffer size =   448.88 MiB
0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.894.943 I sched_reserve:    Vulkan0 compute buffer size =  1312.06 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    b = parse_device_buffers(text)
    assert b["weights"] == pytest.approx(18904.68)  # max, not sum; host/CPU out
    assert b["kv"] == pytest.approx(8704.00 + 1024.00)  # main + draft contexts
    assert b["rs"] == pytest.approx(448.88)
    assert b["compute"] == pytest.approx(1552.33 + 1312.06)  # host excluded
    assert b["output"] == 0.0  # host-visible → excluded


def test_parse_device_buffers_empty_log():
    assert parse_device_buffers("garbage") == {
        "weights": 0.0, "kv": 0.0, "rs": 0.0, "compute": 0.0, "output": 0.0}


# ── calc_ctx ──────────────────────────────────────────────────────────────


def test_calc_ctx_fits_design(make_model, fit_params_block):
    model = make_model("a", **{"measured": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # available = 32768 - 2048 = 30720; remaining = 30720 - 10000 - 1000 = 19720
    # design cost = (0.5*32768 + 0) * 1 = 16384 <= 19720 -> design context
    assert ctx == 32768


def test_calc_ctx_scales_down_and_rounds(make_model):
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server"}
    model = make_model("b", **{"measured": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # remaining = 30720 - 26000 = 4720; X = 4720/(0.5*1) = 9440 -> rounded 8192
    assert ctx == 8192


def test_calc_ctx_slot_mib_charged_per_slot(make_model):
    # A nonzero D is charged once per slot: at p=2 the KV pool doubles AND
    # the slot cost doubles, so the affordable per-slot context shrinks.
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 300.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server"}
    model = make_model("b2", **{"measured": fm})
    p1 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=1)
    p2 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=2)
    # p=1: X = (4720 - 300)/0.5 = 8840 -> 8192
    # p=2: X = (4720 - 600)/(0.5*2) = 4120 -> 4096
    assert p1 == 8192
    assert p2 == 4096


def test_calc_ctx_floors_at_min(make_model):
    fm = {"model_mib": 30000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 0, "cache_type": "q8_0", "source": "llama-server"}
    model = make_model("c", **{"measured": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == _MIN_CTX_SIZE


def test_calc_ctx_applies_spare(make_model, fit_params_block):
    model = make_model("d", **{"measured": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=3072)
    # available = 32768 - 2048 - 3072 = 27648; remaining = 27648-11000 = 16648
    # design cost = 16384 <= 16648 -> design still fits
    assert ctx == 32768


def test_calc_ctx_memory_margin_shrinks_ctx(make_model):
    fm = {"model_mib": 50000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server"}
    model = make_model("mm", **{"measured": fm})
    plain = model.vram.calc_ctx(65536, fit_bin="unused", memory_margin=0.0)
    margined = model.vram.calc_ctx(65536, fit_bin="unused", memory_margin=0.01)
    # available = 63520 (plain) vs 63520/1.01 = 62891.1 (margined)
    # plain: remaining 12520 -> 25040 -> 24576; margined: 11891 -> 23782 -> 16384
    assert plain == 24576
    assert margined == 16384


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


# ── affine pair measurement ───────────────────────────────────────────────


def test_fit_params_static_derives_affine_pair(make_model, monkeypatch):
    model = make_model("ap", **{"measured": None})

    def fake_pair(fit_bin, cache_type, llama_args=""):
        return FitParams(model_mib=18905, kv_per_token_mib=0.0332,
                         slot_mib=448.25, compute_mib=4036,
                         source="fit-estimate", cache_type=cache_type)

    monkeypatch.setattr(model.vram, "_fit_params_serve", fake_pair)
    fp = model.vram.fit_params_static("/bin/fit", cache_type="q8_0")
    assert fp is not None
    assert fp.source == "fit-estimate"
    # v3 block persisted with the calibrated estimate
    block = model.measured_block()
    assert block["kv_per_token_mib"] == 0.0332
    assert block["slot_mib"] == 448.25
    assert block["source"] == "fit-estimate"
    assert "ctx_factor" not in block and "parallel" not in block


def test_fit_params_static_measure_failure_uses_transient_fit_params(
        make_model, monkeypatch):
    model = make_model("ap2", **{"measured": None})
    c_true, d_true = 0.0332, 150.0

    def fake_fit_params(*, fit_bin, fit_ctx, cache_type, parallel,
                        model_path=None, label=None, llama_args=""):
        kv = round(c_true * fit_ctx)
        ctx_mib = kv + int(d_true * parallel)
        return {"model": 18114, "context": ctx_mib, "compute": 505,
                "kv": float(kv), "rs": 0.0, "rs_cells": 0}

    monkeypatch.setattr(model.vram, "_fit_params_serve",
                        lambda *a, **k: None)
    monkeypatch.setattr(model.vram, "fit_params", fake_fit_params)
    fp = model.vram.fit_params_static("/bin/fit", cache_type="q8_0")
    assert fp is not None
    assert fp.source == "fit-params"
    assert fp.kv_per_token_mib == pytest.approx(c_true, abs=2e-3)
    assert fp.slot_mib == pytest.approx(d_true, abs=2e-2)
    # Transient: never persisted — the next run retries the calibrated path.
    assert model.measured_block() is None


def test_fit_params_static_vram_mib_prediction(make_model, fit_params_block):
    """The persisted constants predict memory for any (context, slots)."""
    model = make_model("pred", **{"measured": fit_params_block})
    fp = model.vram.fit_params_static("unused")
    # model + compute + c*pool + p*D
    assert fp.vram_mib(ctx_per_slot=8192, parallel=4) == int(
        10000 + 1000 + 0.5 * (8192 * 4) + 0.0 * 4)


def test_fit_params_static_pair_failure_falls_back(make_model, monkeypatch):
    model = make_model("pf", **{"measured": None})
    monkeypatch.setattr(model.vram, "fit_params", lambda *a, **k: None)
    assert model.vram.fit_params_static("/bin/fit") is None


# ── legacy blocks ─────────────────────────────────────────────────────────


def test_legacy_block_stale_and_rewritten(make_model, fit_params_block):
    legacy = dict(fit_params_block)
    legacy.pop("kv_per_token_mib")
    legacy.pop("slot_mib")
    legacy["ctx_factor"] = 0.5
    legacy["parallel"] = 8
    model = make_model("lg", **{"fit-params": legacy})
    # The pre-affine block cannot be decomposed into (c, D): stale.
    assert model.vram.saved_for("q8_0") is None
    assert model.vram.fit_params_static("unused") is None  # binary unusable


def test_fit_params_static_cache_type_mismatch_remeasures(
        make_model, fit_params_block):
    model = make_model("s", **{"measured": fit_params_block})
    assert model.vram.fit_params_static("unused", cache_type="q8_0") is not None
    # A different cache type has no readable block and no binary: None.
    assert model.vram.fit_params_static("unused", cache_type="f16") is None


# ── image token floor ─────────────────────────────────────────────────────

def _vision(make_model, tmp_path, stem, fit, **extra):
    """Vision model (mmproj companion) with a fit-params block."""
    (tmp_path / f"{stem}-mmproj.gguf").write_bytes(b"mm")
    block = {"file": f"{stem}-mmproj.gguf", "capabilities": ["image"]}
    tokens = {k: v for k, v in extra.items()
              if k in ("image_min_tokens", "image_max_tokens")}
    rest = {k: v for k, v in extra.items() if k not in tokens}
    block.update(tokens)
    fm = {"measured": fit, "mmproj": block}
    fm.update(rest)
    return make_model(stem, **fm)


def test_calc_ctx_image_floor_raises_ctx(make_model, tmp_path):
    # A declared image_max_tokens must fit the solved context: the ctx is
    # raised from the rounded-down value to the floor when affordable.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server"}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=9000)
    ctx = model.view_for(True).vram.calc_ctx(
        32768, fit_bin="unused", include_mmproj=True)
    # remaining = 30720 - 26000 - mmproj(0 + 150) = 4570; affordable 9140
    # tokens; rounded ctx 8192 -> raised to the 9000 floor (affordable)
    assert ctx == 9000


def test_calc_ctx_image_floor_unaffordable_warns(make_model, tmp_path, caplog):
    import logging

    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server"}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=16384)
    with caplog.at_level(logging.WARNING):
        ctx = model.view_for(True).vram.calc_ctx(
            32768, fit_bin="unused", include_mmproj=True)
    # floor 16384 > affordable 9140: ctx raised to what fits (9140), warning
    assert ctx == 9140
    assert any("large images may still not fit" in r.message
               for r in caplog.records)


def test_calc_ctx_image_floor_no_mmproj_no_floor(make_model, tmp_path):
    # image_max_tokens without an attached mmproj: no flags, no floor.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server"}
    model = make_model("i", **{"measured": fit, "image_max_tokens": 9440})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", include_mmproj=True)
    assert ctx == 8192


def test_calc_ctx_text_variant_ignores_image_floor(make_model, tmp_path):
    # The -text variant serves no vision, so the floor must not apply.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server"}
    model = _vision(make_model, tmp_path, "i", fit, image_max_tokens=9440)
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", include_mmproj=False)
    assert ctx == 8192


# ── solve_matrix_ctx ──────────────────────────────────────────────────────


def test_solve_matrix_ctx_basic(make_model):
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768,
        spare_mb=0,
        chat_models=[(chat, 8000, 0.4, 0.0, 500, 1, 0)],
        embed_params=(500, 0.1, 0.0, 100),
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
    assert ctx == _MIN_CTX_SIZE


def test_solve_matrix_ctx_exhausted_budget(make_model):
    chat = make_model("chat", context_length=32768)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[(chat, 40000, 0.4, 0.0, 0, 1, 0)],
        embed_params=None, rerank_params=None,
    )
    # chat_budget = 30720 - 40000 < 0 -> skipped -> best_ctx 0 -> MIN_CTX
    assert ctx == _MIN_CTX_SIZE


def test_solve_matrix_ctx_fixed_overhead_mb(make_model):
    chat = make_model("chat", context_length=32768)
    chat_models = [(chat, 8000, 0.2, 0.0, 500, 1, 0)]
    # available = 14336; budget = 5836 -> 29180 -> round 24576
    assert solve_matrix_ctx(
        vram_total_mb=16384, spare_mb=0, chat_models=chat_models,
        embed_params=None, rerank_params=None,
    ) == 24576
    # 2000 MB of co-loads: budget = 3836 -> 19180 -> round 16384
    assert solve_matrix_ctx(
        vram_total_mb=16384, spare_mb=0, chat_models=chat_models,
        embed_params=None, rerank_params=None, fixed_overhead_mb=2000,
    ) == 16384


# ── fixed-overhead backends ───────────────────────────────────────────────


def test_vram_mb_pin_is_authoritative(make_model):
    model = make_model("s", backend="whisper-server", vram_mb=1280,
                       context_length=8192)
    # Pinned: the pin IS the sizing — weights, zero KV terms, zero compute.
    assert model.vram.effective_static("unused") == (1280, 0.0, 0.0, 0)


def test_vram_mb_pin_invalid_ignored(make_model):
    model = make_model("s", backend="whisper-server", vram_mb="big",
                       context_length=8192)
    quad = model.vram.effective_static("unused")
    # Falls back to file size (dummy bytes -> 0 MB) + whisper compute buffer.
    assert quad[0] == 0 and quad[1] == 0.0 and quad[2] == 0.0
    assert quad[3] == 100  # _WHISPER_COMPUTE_MB


def test_whisper_fixed_compute_is_small(make_model):
    model = make_model("s", backend="whisper-server", context_length=8192)
    # Measured on Vulkan (nemo-speech): footprint ~= sum of files, so the
    # per-model buffer is small — not the 512 MB sd-server constant.
    quad = model.vram.effective_static("unused")
    assert quad[3] == 100
