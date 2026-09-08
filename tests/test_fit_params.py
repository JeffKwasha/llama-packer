# tests/test_fit_params.py
"""FitParams validation and (de)serialization."""

from __future__ import annotations

from llama_packer.vram import FitParams


def make_block(**overrides) -> dict:
    base = {
        "model_mib": 10000,
        "kv_per_token_mib": 0.5,
        "slot_mib": 0.0,
        "compute_mib": 1000,
        "source": "llama-server",
        "cache_type": "q8_0",
    }
    base.update(overrides)
    return base


def test_from_dict_roundtrip():
    fp = FitParams.from_dict(make_block(), "q8_0")
    assert fp is not None
    assert fp.model_mib == 10000
    assert fp.kv_per_token_mib == 0.5
    assert fp.slot_mib == 0.0
    assert fp.compute_mib == 1000
    assert fp.source == "llama-server"


def test_from_dict_missing_key_returns_none():
    assert FitParams.from_dict({"model_mib": 1, "kv_per_token_mib": 1.0},
                               "q8_0") is None


def test_from_dict_non_numeric_returns_none():
    assert FitParams.from_dict(make_block(model_mib="x"), "q8_0") is None


def test_from_dict_nonpositive_returns_none():
    assert FitParams.from_dict(make_block(model_mib=0), "q8_0") is None
    assert FitParams.from_dict(make_block(kv_per_token_mib=0.0), "q8_0") is None
    assert FitParams.from_dict(make_block(slot_mib=-1.0), "q8_0") is None


def test_from_dict_stale_cache_type_returns_none():
    assert FitParams.from_dict(make_block(cache_type="q4_0"), "q8_0") is None


def test_from_dict_non_dict_returns_none():
    assert FitParams.from_dict("nope", "q8_0") is None


def test_to_dict_roundtrip():
    fp = FitParams(10000, 0.5, 150.0, 1000, "vllm-estimate", "q8_0")
    d = fp.to_dict()
    assert d["source"] == "vllm-estimate"
    assert "parallel" not in d and "ctx_factor" not in d  # legacy keys gone
    fp2 = FitParams.from_dict(d, "q8_0")
    assert fp2 == fp


def test_vram_mib_affine_prediction():
    fp = FitParams(18114, 0.0332, 150.0, 505, "llama-server", "q8_0")
    # The law: model + compute + c*pool + p*D, pool = per-slot * slots.
    assert fp.vram_mib(ctx_per_slot=32768, parallel=8) == int(
        18114 + 505 + 0.0332 * (32768 * 8) + 150 * 8)
    assert fp.vram_mib(ctx_per_slot=262144, parallel=1) == int(
        18114 + 505 + 0.0332 * 262144 + 150)
