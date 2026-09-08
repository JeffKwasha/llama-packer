# tests/test_memory_probe.py
"""VRAM probe: family representatives, serve-shaped grid totals, affine
least-squares fit with residual degrees of freedom, FAIL-tolerant table
output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llama_packer.memory_probe import (
    _fit_affine,
    _probe_grid,
    affine_report,
    format_reports,
    pick_family_representatives,
    run_probe,
)
from llama_packer.vram import affine_from_pair


def _fake(stem, tmp, arch="qwen3", size=100, suffix=".gguf",
          **kw) -> Any:
    path = tmp / f"{stem}{suffix}"
    path.write_bytes(b"\x00" * 8)
    kw.setdefault("on_cpu", False)
    kw.setdefault("_override_error", None)
    return SimpleNamespace(
        stem=stem, arch=arch, size_mb=size, gguf_path=path, **kw)


def test_affine_from_pair_exact():
    # ctx(1)=2879, ctx(2)=3038 at C=262144 -> c=0.010376, D=159 (gemma4-26B)
    c, d = affine_from_pair(2879, 3038, 262144)
    assert d == 159.0
    assert abs(c - (2879 - 159) / 262144) < 1e-12
    # negative measurement noise clamps D to 0
    c, d = affine_from_pair(1000, 999, 32768)
    assert d == 0.0
    assert abs(c - 1000 / 32768) < 1e-12


def test_picks_largest_per_family(tmp_path):
    models = [
        _fake("small", arch="qwen3", size=100, tmp=tmp_path),
        _fake("big", arch="qwen3", size=500, tmp=tmp_path),
        _fake("other", arch="llama", size=50, tmp=tmp_path),
    ]
    reps = pick_family_representatives(models)
    assert list(reps) == ["llama", "qwen3"]  # arch-sorted
    assert reps["qwen3"].stem == "big"
    assert reps["llama"].stem == "other"


def test_picks_fast_storage_over_larger_platter(tmp_path):
    # a fast-tier model wins the family even when a bigger file sits on
    # the slow branch (the SSD loads ~6x faster; (c, D) are arch-given)
    ssd = tmp_path / "ssd"
    hdd = tmp_path / "hdd"
    ssd.mkdir()
    hdd.mkdir()
    models = [
        _fake("big-slow", arch="gemma4", size=21842, tmp=hdd),
        _fake("small-fast", arch="gemma4", size=8053, tmp=ssd),
    ]
    def is_fast(path: str) -> bool:
        return path.startswith(str(ssd))
    reps = pick_family_representatives(models, None, is_fast)
    assert reps["gemma4"].stem == "small-fast"
    # without tier knowledge, the largest still wins
    reps = pick_family_representatives(models)
    assert reps["gemma4"].stem == "big-slow"
    # fast-vs-fast falls back to largest
    hdd_fast = tmp_path / "hdd2"
    hdd_fast.mkdir()
    (hdd_fast / "x").write_bytes(b"")
    models2 = models + [
        _fake("big-fast", arch="gemma4", size=21000, tmp=ssd),
    ]
    reps = pick_family_representatives(models2, None, is_fast)
    assert reps["gemma4"].stem == "big-fast"


def test_pick_family_representatives_only_archs(tmp_path):
    models = [
        _fake("a", arch="qwen35", size=100, tmp=tmp_path),
        _fake("b", arch="gemma4", size=200, tmp=tmp_path),
    ]
    reps = pick_family_representatives(models, ("qwen35",))
    assert list(reps) == ["qwen35"]
    assert reps["qwen35"].stem == "a"


def test_skips_unmeasurable(tmp_path):
    missing = SimpleNamespace(
        stem="ghost", arch="qwen3", size_mb=10,
        gguf_path=tmp_path / "ghost.gguf",
        on_cpu=False, _override_error=None)
    models = [
        missing,
        _fake("cpu", tmp=tmp_path, on_cpu=True),
        _fake("err", tmp=tmp_path, _override_error="boom"),
        _fake("st", suffix=".safetensors", tmp=tmp_path),
        _fake("noarch", arch="", tmp=tmp_path),
        _fake("ok", tmp=tmp_path),
    ]
    reps = pick_family_representatives(models)
    assert list(reps) == ["qwen3"]
    assert reps["qwen3"].stem == "ok"


def test_probe_grid_spans_both_axes(tmp_path):
    grid = _probe_grid(262144)
    # three C points (C-linearity check) x two p values, deduplicated
    assert grid == ((65536, 1), (65536, 2), (32768, 1), (32768, 2),
                    (16384, 1))
    # small design: collapses onto its own scales, still spans C
    assert _probe_grid(32768) == ((32768, 1), (32768, 2), (16384, 1),
                                  (16384, 2), (8192, 1))


def test_fit_affine_exact_and_overdetermined():
    # exactly-determined trio reproduces the law (real Dirk numbers:
    # totals 25435 / 26005 / 23891 at (65536,1), (65536,2), (32768,1))
    fit = _fit_affine([(65536, 1, 25435), (65536, 2, 26005),
                       (32768, 1, 23891)])
    assert fit is not None
    fixed, c, d = fit
    assert d == pytest.approx(570.0)
    assert c == pytest.approx(0.047119140625, abs=1e-9)
    assert fixed == pytest.approx(21777.0, abs=0.5)
    # overdetermined grid still recovers the same constants
    pts = [(C, p, 1100 + round(0.01 * C) + 150 * p)
           for C in (65536, 32768) for p in (1, 2, 4)]
    fit2 = _fit_affine(pts)
    assert fit2 is not None
    # MiB-rounding noise in the synthetic points bounds precision ~1e-4
    assert fit2[1] == pytest.approx(0.01, abs=1e-4)
    assert fit2[2] == pytest.approx(150.0, abs=0.01)
    assert fit2[0] == pytest.approx(1100.0, abs=1.0)
    # degenerate (all same C and p) cannot be fit
    assert _fit_affine([(1000, 1, 5), (1000, 1, 6), (1000, 1, 7)]) is None


def test_affine_report_exact_law_passes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    C64 = 1 / 64  # exact MiB/token: every grid pool is an integer

    class FakeVram:
        def _run_measure_server(self, server_bin, cache_type, ctx, parallel,
                                llama_args):
            # truth: fixed 1100 + c*C + p*D, with the ctx-scaling part in
            # the KV pool line (where the derivation reads it)
            return {"weights": 1100.0,
                    "kv": float(ctx // 64 + 150 * parallel),
                    "rs": 0.0, "compute": 0.0, "output": 0.0}

    m = _fake("m", tmp=tmp_path)
    m.design_context = 32000
    m._mtp_info = lambda: (False, 0)
    m.vram = FakeVram()
    monkeypatch.setattr(
        "llama_packer.memory_probe.save_serve_correction",
        lambda *a, **k: None)
    # estimate is the truth minus a known correction: fit-params sees
    # c=C64/2, D=100, fixed=1000 -> correction closes to (C64, 150, 1100)
    m.vram._fit_params_serve = lambda *a, **k: SimpleNamespace(
        model_mib=900, compute_mib=100, kv_per_token_mib=C64 / 2,
        slot_mib=100.0)
    rep = affine_report(m, "/bin/fit", "/bin/llama-server", "q8_0")
    assert rep is not None
    assert rep["ok"] is True
    assert rep["slot_mib"] == pytest.approx(150.0, abs=1e-6)
    assert rep["kv_per_token_mib"] == pytest.approx(C64, abs=1e-9)
    assert rep["fixed_mib"] == 1100
    assert len(rep["grid"]) == 5
    assert rep["max_residual"] <= 0.005
    assert rep["corr"] == {"delta_fixed": 100,
                           "delta_c": pytest.approx(C64 / 2),
                           "delta_d": pytest.approx(50.0),
                           "rep_stem": "m", "cache_type": "q8_0",
                           "shape": ""}


def test_affine_report_nonlinear_fails(tmp_path):
    from types import SimpleNamespace

    class FakeVram:
        def _run_measure_server(self, server_bin, cache_type, ctx, parallel,
                                llama_args):
            # one gross outlier point; the rest sits on an exact pool law
            extra = 50000 if (ctx, parallel) == (32768, 2) else 0.0
            return {"weights": 1000.0,
                    "kv": float(ctx // 64 + 150 * parallel + extra),
                    "rs": 0.0, "compute": 0.0, "output": 0.0}
        def _fit_params_serve(self, *a, **k):
            return SimpleNamespace(model_mib=900, compute_mib=100,
                                   kv_per_token_mib=1 / 64, slot_mib=150.0)

    m = _fake("m", tmp=tmp_path)
    m.design_context = 32768
    m._mtp_info = lambda: (False, 0)
    m.vram = FakeVram()
    rep = affine_report(m, "/bin/fit", "/bin/llama-server", "q8_0")
    assert rep is not None
    assert rep["ok"] is False


def test_affine_report_missing_baseline(tmp_path):
    class FakeVram:
        def _run_measure_server(self, *a, **k):
            return None
        def _fit_params_serve(self, *a, **k):
            return None

    m = _fake("m", tmp=tmp_path)
    m.design_context = 32768
    m.vram = FakeVram()
    assert affine_report(m, "/bin/fit", "/bin/llama-server", "q8_0") is None


def test_affine_report_insufficient_points(tmp_path):
    class FakeVram:
        def _run_measure_server(self, server_bin, cache_type, ctx, parallel,
                                llama_args):
            return None if parallel == 2 else {"weights": 100, "kv": 0.0,
                                               "rs": 0.0, "compute": 0.0,
                                               "output": 0.0}
        def _fit_params_serve(self, *a, **k):
            return None

    m = _fake("m", tmp=tmp_path)
    m.design_context = 32768
    m.vram = FakeVram()
    # only three points measure: no residual degrees of freedom -> None
    assert affine_report(m, "/bin/fit", "/bin/llama-server", "q8_0") is None


def test_format_reports_shows_verdicts(tmp_path):
    good = {"model": _fake("good", tmp=tmp_path, size=500),
            "kv_per_token_mib": 0.01, "slot_mib": 150.0, "fixed_mib": 1100,
            "corr": {"delta_fixed": 100, "delta_c": 0.002,
                     "delta_d": 50.0},
            "grid": [(65536, 1), (65536, 2), (32768, 1)], "residuals": {},
            "max_residual": 0.0, "ok": True}
    text = format_reports([good, None])
    assert "PASS" in text and "FAIL" in text
    assert "c_mib/tok" in text and "slot_mib" in text
    assert "+100" in text and "+0.002" in text


def test_run_probe_end_to_end(tmp_path, monkeypatch):
    # The probe pre-flight reads live processes; this repo's normal state is
    # a resident llama-swap, so the test stubs a clean GPU.
    monkeypatch.setattr("llama_packer.gpu_state.llama_residents", lambda: [])

    class FakeVram:
        def _run_measure_server(self, server_bin, cache_type, ctx, parallel,
                                llama_args):
            # exact law: fixed 1100 + c*C + p*D, c=1/64, D=100
            return {"weights": 1100.0,
                    "kv": float(ctx // 64 + 100 * parallel),
                    "rs": 0.0, "compute": 0.0, "output": 0.0}
        def _fit_params_serve(self, *a, **k):
            return SimpleNamespace(model_mib=1000, compute_mib=100,
                                   kv_per_token_mib=1 / 64, slot_mib=100.0)

    a = _fake("qa", arch="qwen3", size=100, tmp=tmp_path)
    b = _fake("qb", arch="qwen3", size=400, tmp=tmp_path)
    for m in (a, b):
        m.design_context = 32768
        m._mtp_info = lambda: (False, 0)
        m.vram = FakeVram()
    monkeypatch.setattr(
        "llama_packer.memory_probe.save_serve_correction",
        lambda *a, **k: None)
    text = run_probe([a, b], "/bin/fit", "/bin/llama-server", "q8_0")
    assert "qb" in text and "qa" not in text  # largest represents qwen3
    assert "PASS" in text and "FAIL" not in text
