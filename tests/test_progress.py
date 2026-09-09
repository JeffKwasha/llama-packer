# tests/test_progress.py
"""Progress bar no-op behavior + slow-op logging guards."""

from __future__ import annotations

import logging
import os
import sys
import types

from llama_packer.progress import PackerProgress


def test_progress_noop_when_disabled():
    p = PackerProgress(enabled=False)
    p.start(3, "budgeting")
    assert p._progress is None
    p.advance("x")
    assert p._done == 1
    p.stop()


def test_progress_noop_under_pytest():
    # PYTEST_CURRENT_TEST is set by the runner, so the bar never activates.
    assert os.environ.get("PYTEST_CURRENT_TEST")
    p = PackerProgress(enabled=True)
    p.start(3, "budgeting")
    assert p._progress is None
    p.advance("x")
    p.stop()


def test_active_bar_routes_logs_through_console(monkeypatch):
    """While the bar is live, log records render through the bar's rich
    console (which prints them ABOVE the bar) and the previous handlers are
    restored on stop."""
    import llama_packer.progress as prog

    p = prog.PackerProgress(enabled=True)
    monkeypatch.setattr(prog.PackerProgress, "_active", lambda self: True)
    root = logging.getLogger()
    before = [(type(h), h) for h in root.handlers]
    p.start(2, "budgeting")
    try:
        assert p._progress is not None
        assert any(isinstance(h, prog._ConsoleLogHandler)
                   for h in root.handlers)
        calls = []
        monkeypatch.setattr(p._progress.console, "print",
                            lambda *a, **k: calls.append(a))
        logging.getLogger("test.active").warning("boom")
        assert calls and "boom" in str(calls[0][0])
    finally:
        p.stop()
    after = [(type(h), h) for h in root.handlers]
    assert not any(isinstance(h, prog._ConsoleLogHandler) for h, _ in after)
    assert after == before


def test_active_bar_restores_handlers_even_on_stop_error(monkeypatch):
    import llama_packer.progress as prog

    p = prog.PackerProgress(enabled=True)
    monkeypatch.setattr(prog.PackerProgress, "_active", lambda self: True)
    root = logging.getLogger()
    before = [(type(h), h) for h in root.handlers]
    p.start(2, "budgeting")
    assert any(isinstance(h, prog._ConsoleLogHandler) for h in root.handlers)
    monkeypatch.setattr(p._progress, "stop",
                        lambda: (_ for _ in ()).throw(RuntimeError("nope")))
    p.stop()  # must not raise
    after = [(type(h), h) for h in root.handlers]
    assert after == before


def test_estimator_forces_offline_and_restores(monkeypatch):
    from llama_packer import vllm_estimate

    seen = {}

    class FakeParams:
        nominal_gib = 10.0

    class FakeKV:
        nominal_gib = 1.0

    class FakeEst:
        parameters = FakeParams()
        activations = FakeParams()
        workspace = FakeParams()
        vllm_overhead = FakeParams()
        kv_cache = FakeKV()

    def fake_estimate(inputs):
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        return None, FakeEst()

    fake_mod = types.ModuleType("memory_estimator")
    fake_mod.EstimatorInputs = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_mod.estimate_from_inputs = fake_estimate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "memory_estimator", fake_mod)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    out = vllm_estimate.estimate_vllm("org/repo", 4096)
    assert out is not None
    assert seen["offline"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ


def test_estimate_vllm_kv_scales_with_cache_type(monkeypatch):
    # The estimator prices KV at the native (auto) dtype ~2 B/elem; the
    # cache_type decision (q8_0 -> --kv-cache-dtype fp8) must rescale it.
    from llama_packer import vllm_estimate

    class FakeOne:
        nominal_gib = 10.0

    class FakeKV:
        nominal_gib = 1.0

    class FakeEst:
        parameters = FakeOne()
        activations = FakeOne()
        workspace = FakeOne()
        vllm_overhead = FakeOne()
        kv_cache = FakeKV()

    fake_mod = types.ModuleType("memory_estimator")
    fake_mod.EstimatorInputs = lambda **kw: types.SimpleNamespace(**kw)  # type: ignore[attr-defined]
    fake_mod.estimate_from_inputs = lambda inputs: (None, FakeEst())  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "memory_estimator", fake_mod)

    model_mib, ctx_factor, _ = vllm_estimate.estimate_vllm(
        "org/repo", 4096, cache_type="q8_0")
    assert model_mib == 10 * 1024
    # 1 GiB bf16 KV -> 1.0625/2 = 0.53125 of it at fp8-equivalent bytes
    assert ctx_factor == (1.0 * 1024 * 1.0625 / 2.0) / 4096


def test_fit_params_logs_before_run(make_model, fit_params_block, monkeypatch, caplog):
    import subprocess

    m = make_model("pl")
    m.frontmatter.pop("fit-params", None)
    budget = m.vram

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(
            stdout="Vulkan0 100 200 50\n", stderr="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with caplog.at_level(logging.INFO, logger="llama_packer.vram"):
        budget.fit_params("fit-bin", fit_ctx=1024)
    assert any("measuring VRAM" in r.message for r in caplog.records)


def test_fit_params_cache_hit_is_silent(make_model, fit_params_block,
                                        monkeypatch, caplog):
    import subprocess

    m = make_model("pc", **{"derived": fit_params_block})
    budget = m.vram
    # Prime the in-memory cache via a first call backed by saved frontmatter.
    budget.fit_params_static("fit-bin")
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
                            AssertionError("must not run")))
    with caplog.at_level(logging.INFO, logger="llama_packer.vram"):
        budget.fit_params_static("fit-bin")
    assert calls == []
    assert not any("measuring VRAM" in r.message for r in caplog.records)
