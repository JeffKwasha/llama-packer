# tests/test_vram_budget.py
"""calc_ctx and solve_matrix_ctx budget math under the affine VRAM law:

    VRAM(C, p) = model_mib + compute_mib + c*C + p*D

with C the total KV pool (= per-slot context X times parallel) and X the
solved, emitted value.
"""

from __future__ import annotations

import json
import os

import pytest

from llama_packer import gpu_state, vram
from llama_packer.consts import _MIN_CTX_SIZE
from llama_packer.vram import (FitParams, parse_device_buffers,
                               parse_spill_mib, solve_matrix_ctx)


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


# ── spill validity check (2026-09-07 post-mortem) ─────────────────────────

def test_parse_spill_mib_host_kv_is_spill():
    """Host KV = the capacity-spill signature; structural CPU-mapped
    weights (<= 2 GiB) and host compute/output staging are not spill."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =   994.63 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  4352.00 MiB
0.28.745 I llama_kv_cache: Vulkan_Host KV buffer size =  4352.00 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    assert parse_spill_mib(text) == pytest.approx(4352.00)


def test_parse_spill_mib_cpu_map_beyond_tolerance():
    """CPU-mapped weights above the structural headroom = offload failure."""
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size =  8000.00 MiB
0.24.044 I load_tensors:   CPU_Mapped model buffer size =  12000.00 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  1000.00 MiB
"""
    assert parse_spill_mib(text) == pytest.approx(12000.00 - 2048.0)


def test_parse_spill_mib_clean_log_is_zero():
    text = """\
0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB
0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB
0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB
0.28.928 I sched_reserve: Vulkan_Host compute buffer size =  1104.34 MiB
0.00.440 I llama_context: Vulkan_Host  output buffer size =     0.95 MiB
"""
    assert parse_spill_mib(text) == 0.0


# ── guarded serve measurement lifecycle ───────────────────────────────────

def _stub_server(tmp_path, body: str) -> str:
    path = tmp_path / "stub-measure-server.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


_OK_BODY = """\
echo '0.00.437 I load_tensors:      Vulkan0 model buffer size = 18904.68 MiB'
echo '0.28.744 I llama_kv_cache:    Vulkan0 KV buffer size =  8704.00 MiB'
echo '0.28.928 I sched_reserve:    Vulkan0 compute buffer size =  1552.33 MiB'
echo listening on http://127.0.0.1:1
exec sleep 30
"""


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    """Clear pre-flight, tmp cache dir, fast stall window."""
    monkeypatch.setattr(vram.gpu_state, "llama_residents", lambda: [])
    monkeypatch.setattr(vram, "_MEASURE_STALL_S", 1)
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def test_run_measure_server_ok_and_child_dead(make_model, tmp_path, guard_env):
    model = make_model("a")
    buffers = model.vram._run_measure_server(
        _stub_server(tmp_path, _OK_BODY), "q8_0", 8192, 1, "")
    assert buffers is not None
    assert buffers["kv"] == pytest.approx(8704.0)
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "ok" and entry["pid"]
    with pytest.raises(ProcessLookupError):
        os.killpg(entry["pid"], 0)  # our child died with its measurement


def test_run_measure_server_rejects_spill(make_model, tmp_path, guard_env):
    body = ("echo '0.00 I llama_kv_cache: Vulkan_Host KV buffer size = "
            "8704.00 MiB'\necho listening on http://127.0.0.1:1\nexec sleep 30\n")
    model = make_model("a")
    assert model.vram._run_measure_server(
        _stub_server(tmp_path, body), "q8_0", 8192, 1, "") is None
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "spill"


def test_run_measure_server_refused_beside_residents(make_model, tmp_path,
                                                     monkeypatch):
    monkeypatch.setattr(vram.gpu_state, "llama_residents",
                        lambda: ["123 jk /x/llama-server -m y"])
    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    model = make_model("a")
    # A nonexistent binary proves the refusal happened before any spawn.
    assert model.vram._run_measure_server(
        str(tmp_path / "never-spawned"), "q8_0", 8192, 1, "") is None
    entry = json.loads((tmp_path / "cache" / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "refused-residents"


def test_run_measure_server_stall_abandons_quickly(make_model, tmp_path,
                                                   guard_env):
    model = make_model("a")
    stub = _stub_server(tmp_path, "echo starting\nexec sleep 60\n")
    buffers = model.vram._run_measure_server(stub, "q8_0", 8192, 1, "")
    assert buffers is None  # well under _MEASURE_TIMEOUT_S
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "stall"
    with pytest.raises(ProcessLookupError):
        os.killpg(entry["pid"], 0)


def test_run_measure_server_exit_detected(make_model, tmp_path, guard_env):
    model = make_model("a")
    stub = _stub_server(tmp_path, "exit 1\n")
    assert model.vram._run_measure_server(stub, "q8_0", 8192, 1, "") is None
    entry = json.loads((guard_env / "measure-journal.jsonl")
                       .read_text().splitlines()[-1])
    assert entry["outcome"] == "exited"


# ── calc_ctx ──────────────────────────────────────────────────────────────


def test_calc_ctx_fits_design(make_model, fit_params_block):
    model = make_model("a", **{"derived": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # available = 32768 - 2048 = 30720; remaining = 30720 - 10000 - 1000 = 19720
    # design cost = (0.5*32768 + 0) * 1 = 16384 <= 19720 -> design context
    assert ctx == 32768


def test_calc_ctx_scales_down_and_rounds(make_model):
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("b", **{"derived": fm})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    # remaining = 30720 - 26000 = 4720; X = 4720/(0.5*1) = 9440 -> rounded 8192
    assert ctx == 8192


def test_calc_ctx_slot_mib_charged_per_slot(make_model):
    # A nonzero D is charged once per slot: at p=2 the KV pool doubles AND
    # the slot cost doubles, so the affordable per-slot context shrinks.
    fm = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 300.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("b2", **{"derived": fm})
    p1 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=1)
    p2 = model.vram.calc_ctx(32768, fit_bin="unused", parallel=2)
    # p=1: X = (4720 - 300)/0.5 = 8840 -> 8192
    # p=2: X = (4720 - 600)/(0.5*2) = 4120 -> 4096
    assert p1 == 8192
    assert p2 == 4096

def test_calc_ctx_floors_at_min(make_model, caplog):
    import logging

    fm = {"model_mib": 30000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 0, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("c", **{"derived": fm})
    with caplog.at_level(logging.WARNING):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=1024)
    assert ctx == _MIN_CTX_SIZE
    # The warning must say what the user gets and why: needs vs budget and
    # the verdict (the model cannot load — the planner disables it).
    msg = next(r.message for r in caplog.records
               if "need" in r.message and "budget" in r.message)
    assert "30000" in msg          # needs MiB
    assert "29696" in msg          # budgeted MiB (32768 - 2048 - 1024)
    assert "cannot load" in msg
    # The budget verdict is recorded for the planner's gating check.
    assert model.vram.over_budget_reason is not None
    assert "30000" in model.vram.over_budget_reason
    assert model.vram.vram_squeezed is False


def test_calc_ctx_applies_spare(make_model, fit_params_block):
    model = make_model("d", **{"derived": fit_params_block})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=3072)
    # available = 32768 - 2048 - 3072 = 27648; remaining = 27648-11000 = 16648
    # design cost = 16384 <= 16648 -> design still fits
    assert ctx == 32768


# ── over-budget / squeeze gating flags ───────────────────────────────────


def test_calc_ctx_over_budget_when_available_nonpositive(make_model, caplog):
    import logging

    fm = {"model_mib": 1000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 100, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("nb", **{"derived": fm})
    with caplog.at_level(logging.WARNING):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused", spare_mb=40000)
    assert ctx == _MIN_CTX_SIZE
    assert model.vram.over_budget_reason is not None
    assert "exhausted" in model.vram.over_budget_reason
    assert model.vram.vram_squeezed is False


def test_calc_ctx_squeezed_flag_when_design_does_not_fit(make_model):
    # Weights fit but the full 32768 design context does not: the solve
    # falls back smaller and flags the squeeze for the description note.
    fm = {"model_mib": 32000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 100, "cache_type": "q8_0",
          "source": "llama-server", "shape": ""}
    model = make_model("sq", **{"derived": fm})
    ctx = model.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    # available = 47104; remaining = 47104 - 32100 = 15004
    # design cost = 16384 > 15004 -> squeezed; not over budget.
    assert ctx < 32768
    assert model.vram.over_budget_reason is None
    assert model.vram.vram_squeezed is True


def test_calc_ctx_flags_reset_between_calls(make_model, fit_params_block):
    """Flags are per-solve: a tight call's verdict must not leak into the
    next call's plan (groups may carry different spare)."""
    tight = make_model("r1", **{"derived": {
        "model_mib": 50000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
        "compute_mib": 100, "cache_type": "q8_0",
        "source": "llama-server", "shape": ""}})
    tight.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    assert tight.vram.over_budget_reason is not None

    # Same budget object, looser effective budget via the fitting block:
    # swap the measured quad for one that fits and the flags must clear.
    tight.vram.effective_static = lambda *a, **k: (
        1000, 0.5, 0.0, 100)
    ctx = tight.vram.calc_ctx(49152, fit_bin="unused", spare_mb=0)
    assert tight.vram.over_budget_reason is None
    assert tight.vram.vram_squeezed is False
    assert ctx >= _MIN_CTX_SIZE


def test_calc_ctx_memory_margin_shrinks_ctx(make_model):
    fm = {"model_mib": 50000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
          "compute_mib": 1000, "cache_type": "q8_0", "source": "llama-server", "shape": ""}
    model = make_model("mm", **{"derived": fm})
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


# ── unestimable fallback (no estimate source works) ──────────────────────


def test_calc_ctx_unestimable_falls_back_to_design(make_model, caplog):
    """No measurement source -> design ctx + unestimated flag, never raise."""
    import logging

    model = make_model("ue", context_length=32768)
    model.vram.effective_static = lambda *a, **k: None
    with caplog.at_level(logging.WARNING, logger="llama_packer.vram"):
        ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == 32768
    assert model.vram.unestimated_reason is not None
    assert any("VRAM measurement failed" in r.message for r in caplog.records)
    # The warning is deduped: a repeat solve logs it once.
    model.vram.calc_ctx(32768, fit_bin="unused")
    assert sum("VRAM measurement failed" in r.message
               for r in caplog.records) == 1


def test_fit_params_undecodable_output_returns_none(make_model, monkeypatch,
                                                    caplog):
    """fit-params logs can carry non-UTF-8 bytes; strict decoding must not
    blow up inside subprocess (LFM2.5-VL regression: UnicodeDecodeError)."""
    import logging
    import subprocess

    model = make_model("uf")
    budget = model.vram

    def fake_run(cmd, **kw):
        raise UnicodeDecodeError("utf-8", b"\xc4", 0, 1,
                                 "invalid continuation byte")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.WARNING, logger="llama_packer.vram"):
        assert budget.fit_params("fit-bin", fit_ctx=1024) is None
    assert any("fit-params failed" in r.message for r in caplog.records)


def test_calc_ctx_vllm_no_estimate_returns_design(make_model):
    # vLLM model with no estimator and no local safetensors: graceful fallback.
    model = make_model("f", backend="vllm", hf_repo="org/model",
                       context_length=65536)
    ctx = model.vram.calc_ctx(32768, fit_bin="unused")
    assert ctx == 65536


# ── affine pair measurement ───────────────────────────────────────────────


def test_fit_params_static_derives_affine_pair(make_model, monkeypatch):
    model = make_model("ap", **{"derived": None})

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
    model = make_model("ap2", **{"derived": None})
    c_true, d_true = 0.0332, 150.0

    def fake_fit_params(*, fit_bin, fit_ctx, cache_type, parallel,
                        model_path=None, label=None, llama_args=""):
        # The KV pool line carries the per-slot cost (SWA ring / RS cells)
        # exactly as the real -lv 5 log does.
        return {"model": 18114, "context": 500, "compute": 505,
                "kv": c_true * fit_ctx + d_true * parallel,
                "rs": 0.0, "rs_cells": 0}

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
    model = make_model("pred", **{"derived": fit_params_block})
    fp = model.vram.fit_params_static("unused")
    # model + compute + c*pool + p*D
    assert fp.vram_mib(ctx_per_slot=8192, parallel=4) == int(
        10000 + 1000 + 0.5 * (8192 * 4) + 0.0 * 4)


def test_fit_params_static_pair_failure_falls_back(make_model, monkeypatch):
    model = make_model("pf", **{"derived": None})
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
    model = make_model("s", **{"derived": fit_params_block})
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
    fm = {"derived": fit, "mmproj": block}
    fm.update(rest)
    return make_model(stem, **fm)


def test_calc_ctx_image_floor_raises_ctx(make_model, tmp_path):
    # A declared image_max_tokens must fit the solved context: the ctx is
    # raised from the rounded-down value to the floor when affordable.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
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
           "source": "llama-server", "shape": ""}
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
           "source": "llama-server", "shape": ""}
    model = make_model("i", **{"derived": fit, "image_max_tokens": 9440})
    ctx = model.vram.calc_ctx(32768, fit_bin="unused", include_mmproj=True)
    assert ctx == 8192


def test_calc_ctx_text_variant_ignores_image_floor(make_model, tmp_path):
    # The -text variant serves no vision, so the floor must not apply.
    fit = {"model_mib": 25000, "kv_per_token_mib": 0.5, "slot_mib": 0.0,
           "compute_mib": 1000, "cache_type": "q8_0",
           "source": "llama-server", "shape": ""}
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


def test_solve_matrix_ctx_rider_does_not_set_the_bar(make_model):
    """An unestimable rider (zero-cost placeholder, kv_factor 0) rides the
    measurable models' solve — its native max must not inflate the shared
    chat context (which would suppress tools demotion fleet-wide and serve
    the unverified model at a context nothing bounded)."""
    measurable = make_model("m", context_length=131072)
    rider = make_model("rider", context_length=262144)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[
            (measurable, 8000, 0.2, 0.0, 500, 1, 0),
            (rider, 0, 0.0, 0.0, 0, 1, 0),
        ],
        embed_params=None, rerank_params=None,
    )
    # rider excluded: budget = 30720 - 8500 = 22220 -> 111100 -> round to
    # 106496 -> min(arch 131072); NOT the rider's native 262144.
    assert ctx == 106496


def test_solve_matrix_ctx_all_riders_fall_to_min(make_model):
    rider = make_model("rider", context_length=262144)
    ctx = solve_matrix_ctx(
        vram_total_mb=32768, spare_mb=0,
        chat_models=[(rider, 0, 0.0, 0.0, 0, 1, 0)],
        embed_params=None, rerank_params=None,
    )
    # Nothing measurable bounds the solve -> the floor.
    assert ctx == _MIN_CTX_SIZE


# ── _measure_affine_trio: the (C, C/2) derivation ─────────────────────────


def test_measure_affine_trio_small_design_exact(make_model):
    """The pool difference divides by the ACTUAL ctx delta: designs below
    8192 (the former r3 floor at 4096) previously mis-derived c by ~1.5x
    and persisted it as a trusted fit-estimate block."""
    model = make_model("t", context_length=6144)
    pools = {6144: 400.0, 3072: 220.0}  # 180 MiB linear over 3072 tokens

    def fake_fit_params(*, fit_bin="", fit_ctx=0, cache_type="q8_0",
                        parallel=1, llama_args="", **kw):
        return {"model": 1000, "kv": pools[fit_ctx], "rs": 0.0,
                "compute": 100}

    model.vram.fit_params = fake_fit_params
    fp = model.vram._measure_affine_trio(6144, "q8_0", "", "fit-estimate",
                                         "unused")
    assert fp is not None
    assert fp.kv_per_token_mib == pytest.approx(180 / 3072)
    assert fp.model_mib == 1000 and fp.compute_mib == 100
    assert fp.slot_mib == 0.0 and fp.source == "fit-estimate"


def test_measure_affine_trio_sub4k_design_still_measures(make_model):
    """Even a sub-4k design (an error or a testcase for text models) now
    measures exactly instead of returning None — no per-run retry tax."""
    model = make_model("t", context_length=4096)
    pools = {4096: 300.0, 2048: 200.0}  # 100 MiB over 2048 tokens

    def fake_fit_params(*, fit_bin="", fit_ctx=0, cache_type="q8_0",
                        parallel=1, llama_args="", **kw):
        return {"model": 900, "kv": pools[fit_ctx], "rs": 0.0,
                "compute": 50}

    model.vram.fit_params = fake_fit_params
    fp = model.vram._measure_affine_trio(4096, "q8_0", "", "fit-estimate",
                                         "unused")
    assert fp is not None
    assert fp.kv_per_token_mib == pytest.approx(100 / 2048)


def test_design_context_sub4k_text_warns(make_model, caplog):
    """A text model with a sub-4k context ceiling is almost certainly a
    mislabeled/corrupt GGUF — one note, not a crash."""
    import logging

    m = make_model("tiny", context_length=2048)
    with caplog.at_level(logging.WARNING):
        assert m.design_context == 2048
        assert m.design_context == 2048  # second read: no repeat
    notes = [r for r in caplog.records if "sub-4k" in r.message]
    assert len(notes) == 1
    # embed/rerank models are legitimate at small contexts: no note
    e = make_model("emb", role="embeddings", context_length=2048)
    with caplog.at_level(logging.WARNING):
        assert e.design_context == 2048
    assert not [r for r in caplog.records if "sub-4k" in r.message
                and "emb" in r.message]


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


def test_correction_keys_on_baked_in_draft_not_companion(
        monkeypatch, tmp_path, make_model, fit_params_block):
    # delta_d on an mtp=1 row prices a draft baked into the main GGUF;
    # a separate draft companion must key mtp=0 and get its draft from
    # the companion fold instead (double-count guard)
    import llama_packer.vram as vram_mod

    monkeypatch.setenv("LLAMA_PACKER_CACHE_DIR", str(tmp_path / "cache"))
    (tmp_path / "draft.gguf").write_bytes(b"x")
    corrections = gpu_state.cache_dir() / "serve-corrections.json"
    corrections.parent.mkdir(exist_ok=True)
    corrections.write_text(json.dumps({
        "fakearch|q8_0|0": {"delta_fixed": 0, "delta_c": 0.0, "delta_d": 0.0},
        "fakearch|q8_0|1": {"delta_fixed": 0, "delta_c": 0.0, "delta_d": 448.9},
    }))

    def fake_trio(design, cache_type, llama_args="", source="",
                  fit_bin=None):
        return FitParams(1000, 0.5, 10.0, 500, "fit-params", "q8_0")

    monkeypatch.setattr(vram_mod.VramBudget, "_measure_affine_trio",
                        staticmethod(fake_trio))

    # companion draft: "mtp" in the name, but not baked in -> mtp=0 row
    m = make_model("cd", **{"derived": dict(fit_params_block),
                            "speculative": "draft.gguf"})
    m._file._header = ("fakearch", True, None)
    fp = m.vram._fit_params_serve("unused", "q8_0")
    assert fp is not None and fp.slot_mib == pytest.approx(10.0)

    # baked-in draft (mtp declared, no companion) -> mtp=1 row
    m2 = make_model("bi", **{"derived": dict(fit_params_block), "mtp": True})
    m2._file._header = ("fakearch", True, None)
    monkeypatch.setattr(vram_mod.utils, "gguf_has_mtp_layers",
                        lambda p: True)
    fp2 = m2.vram._fit_params_serve("unused", "q8_0")
    assert fp2 is not None
    assert fp2.slot_mib == pytest.approx(10.0 + 448.9)
