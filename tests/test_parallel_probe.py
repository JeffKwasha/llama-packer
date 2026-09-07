# tests/test_parallel_probe.py
"""Parallel-scaling probe: one representative per arch family, linearity
ratios against the p=1 baseline, FAIL-tolerant table output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llama_packer.parallel_probe import (
    format_table,
    linearity_rows,
    pick_family_representatives,
    probe_model,
    run_probe,
)


def _fake(stem, tmp, arch="qwen3", size=100, suffix=".gguf",
          **kw) -> Any:
    path = tmp / f"{stem}{suffix}"
    path.write_bytes(b"\x00" * 8)
    kw.setdefault("on_cpu", False)
    kw.setdefault("_override_error", None)
    return SimpleNamespace(
        stem=stem, arch=arch, size_mb=size, gguf_path=path, **kw)


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


def test_probe_model_sweeps_without_persisting(tmp_path):
    calls = []

    class FakeVram:
        def fit_params(self, *, fit_bin, fit_ctx, fit_target_mib,
                       cache_type, parallel):
            calls.append((fit_bin, fit_ctx, cache_type, parallel))
            return (1000, 8000 * parallel, 100)

    m = _fake("m", tmp=tmp_path)
    m.design_context = 131072
    m.vram = FakeVram()
    out = probe_model(m, "/bin/llama-fit-params", "q8_0")
    assert [c[3] for c in calls] == [1, 2, 4, 8]
    assert all(c[0] == "/bin/llama-fit-params" and c[1] == 131072
               and c[2] == "q8_0" for c in calls)
    assert out[8] == (1000, 64000, 100)
    # no persist hook exists on the fake: any persist attempt would raise


def test_linearity_rows_ratios_and_failures():
    results = {1: (1000, 8000, 100), 2: (1000, 16000, 100),
               4: None, 8: (1000, 64000, 120)}
    rows = linearity_rows("m", "qwen3", 500, results)
    by_p = {r["parallel"]: r for r in rows}
    assert by_p[1]["ctx_ratio"] == 1.0
    assert by_p[1]["compute_ratio"] == 1.0
    assert by_p[1]["ok"] is True
    assert by_p[2]["ctx_ratio"] == 1.0
    assert by_p[4]["ok"] is False and by_p[4]["ctx_ratio"] is None
    assert by_p[8]["ctx_ratio"] == 1.0
    assert by_p[8]["compute_ratio"] == 1.2


def test_linearity_rows_missing_baseline():
    rows = linearity_rows("m", "qwen3", 500, {2: (1000, 16000, 100)})
    assert rows and all(r["ok"] is False for r in rows)


def test_format_table_shows_fail_and_ratios():
    rows = linearity_rows("m", "qwen3", 500,
                          {1: (1000, 8000, 100), 2: None})
    text = format_table(rows)
    assert "ctx_ratio" in text
    assert "FAIL" in text
    assert "1.000" in text


def test_run_probe_end_to_end(tmp_path):
    class FakeVram:
        def fit_params(self, **kw):
            p = kw["parallel"]
            return (1000, 8000 * p, 100)

    a = _fake("qa", arch="qwen3", size=100, tmp=tmp_path)
    b = _fake("qb", arch="qwen3", size=400, tmp=tmp_path)
    for m in (a, b):
        m.design_context = 32768
        m.vram = FakeVram()
    text = run_probe([a, b], "/bin/fit", "q8_0")
    assert "qb" in text and "qa" not in text  # largest represents qwen3
    assert text.count("1.000") == 4  # p=1/2/4/8 all linear
