# tests/test_fit_params.py
"""FitParams validation and (de)serialization."""

from __future__ import annotations

from llama_packer.vram import FitParams


def make_block(**overrides) -> dict:
    base = {
        "model_mib": 10000,
        "ctx_factor": 0.5,
        "compute_mib": 1000,
        "source": "fit-params",
        "cache_type": "q8_0",
        "parallel": 1,
    }
    base.update(overrides)
    return base


def test_from_dict_roundtrip():
    fp = FitParams.from_dict(make_block(), "q8_0", 1)
    assert fp is not None
    assert fp.model_mib == 10000
    assert fp.ctx_factor == 0.5
    assert fp.compute_mib == 1000
    assert fp.source == "fit-params"


def test_from_dict_missing_key_returns_none():
    assert FitParams.from_dict({"model_mib": 1, "ctx_factor": 1.0}, "q8_0", 1) is None


def test_from_dict_non_numeric_returns_none():
    assert FitParams.from_dict(make_block(model_mib="x"), "q8_0", 1) is None


def test_from_dict_nonpositive_returns_none():
    assert FitParams.from_dict(make_block(model_mib=0), "q8_0", 1) is None
    assert FitParams.from_dict(make_block(ctx_factor=0.0), "q8_0", 1) is None


def test_from_dict_stale_cache_type_returns_none():
    assert FitParams.from_dict(make_block(cache_type="q4_0"), "q8_0", 1) is None


def test_from_dict_stale_parallel_returns_none():
    assert FitParams.from_dict(make_block(parallel=2), "q8_0", 1) is None


def test_from_dict_non_dict_returns_none():
    assert FitParams.from_dict("nope", "q8_0", 1) is None


def test_to_dict_roundtrip():
    fp = FitParams(10000, 0.5, 1000, "vllm-estimate", "q8_0", 2)
    d = fp.to_dict()
    assert d["source"] == "vllm-estimate"
    assert d["parallel"] == 2
    fp2 = FitParams.from_dict(d, "q8_0", 2)
    assert fp2 == fp
