# tests/test_throughput.py
"""throughput_factor heuristic and its helpers."""

from __future__ import annotations

import pytest


def test_throughput_base(make_model):
    # 12B Q4_K_XL (4.5 bpw) -> base = 54/(12*4.5) = 1.0
    model = make_model("t", parameters="12B", quantization="Q4_K_XL")
    assert model.throughput_factor() == pytest.approx(1.0, abs=1e-3)


def test_throughput_mtp_speedup(make_model):
    # MTP n_max=2, accuracy 0.9 -> speedup = 1 + 2*0.9 = 2.8
    model = make_model(
        "t", parameters="12B", quantization="Q4_K_XL",
        mtp=True, mtp_accuracy=0.9,
    )
    assert model.throughput_factor() == pytest.approx(2.8, abs=1e-3)


def test_throughput_missing_params_returns_none(make_model):
    model = make_model("t")
    assert model.throughput_factor() is None


def test_param_counts_moe(make_model):
    model = make_model("t", parameters="26B-A4B")
    total, active = model._param_counts()
    assert total == 26.0
    assert active == 4.0


def test_quant_bits_ud_prefix(make_model):
    model = make_model("t", quantization="UD-Q4_K_XL")
    assert model._quant_bits() == 4.5
