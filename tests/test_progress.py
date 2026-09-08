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

    m = make_model("pc", **{"measured": fit_params_block})
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
