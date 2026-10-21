# tests/test_throughput.py
"""throughput_factor heuristic and its helpers.

Design parts: SPEC.md, 'Derived fields (computed, not authored)' — throughput_factor is a
relative speed index (not tok/s), and the parameters/quantization fields it
reads.
"""

from __future__ import annotations

import pytest


def test_throughput_base(make_model):
    # 12B Q4_K_XL (4.5 bpw) -> base = 54/(12*4.5) = 1.0
    # Validates: SPEC.md, 'Model Metadata': the relative speed index is 54 / (active_B x quant_bits) when MTP is off.
    model = make_model("t", parameters="12B", quantization="Q4_K_XL")
    assert model.throughput_factor() == pytest.approx(1.0, abs=1e-3)


def test_throughput_mtp_speedup(make_model):
    # MTP n_max=2, accuracy 0.9 -> speedup = 1 + 2*0.9 = 2.8
    # Validates: SPEC.md, 'Model Metadata': MTP multiplies the index by (1 + draft_n x accept_prob).
    model = make_model(
        "t", parameters="12B", quantization="Q4_K_XL",
        mtp=True, mtp_accuracy=0.9,
    )
    assert model.throughput_factor() == pytest.approx(2.8, abs=1e-3)


def test_throughput_missing_params_returns_none(make_model):
    # Validates: SPEC.md, 'Model Metadata': without a parameters field there is no index to report.
    model = make_model("t")
    assert model.throughput_factor() is None


def test_param_counts_moe(make_model):
    # Validates: SPEC.md, 'Model Metadata': a MoE parameters field carries total and active billions.
    model = make_model("t", parameters="26B-A4B")
    total, active = model._param_counts()
    assert total == 26.0
    assert active == 4.0


def test_quant_bits_ud_prefix(make_model):
    # Validates: SPEC.md, 'Model Metadata': the quantisation spelling sets the bits-per-weight the index divides by.
    model = make_model("t", quantization="UD-Q4_K_XL")
    assert model._quant_bits() == 4.5
