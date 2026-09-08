# tests/test_companions.py
"""Companion (mmproj / MTP draft) VRAM folding via VramBudget.effective_static."""

from __future__ import annotations

import pytest

from llama_packer.vram import _DRAFT_COMPUTE_MB, _DRAFT_CTX_SAFETY, _MMPROJ_COMPUTE_MB

MIB = 1024 * 1024


@pytest.fixture
def fit_params_block():
    return {
        "model_mib": 10000,
        "kv_per_token_mib": 0.5,
        "slot_mib": 150.0,
        "compute_mib": 1000,
        # safetensors-estimate: an accepted measured source that did NOT
        # measure the MTP draft, so effective_static folds companions.
        "source": "safetensors-estimate",
        "cache_type": "q8_0",
    }


def test_effective_static_folds_mmproj_and_mtp(tmp_path, make_model, fit_params_block):
    # Companion files must exist before Model construction resolves them.
    (tmp_path / "main-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    (tmp_path / "main.mtp.gguf").write_bytes(b"x" * 2 * MIB)
    m = make_model("main",
                   **{"derived": dict(fit_params_block),
                      "mmproj": {"file": "main-mmproj.gguf"},
                      "speculative": "main.mtp.gguf"})

    model_mib, kv_factor, slot_mib, compute_mib = m.vram.effective_static(
        fit_bin="unused", cache_type="q8_0")

    assert model_mib == 10000 + 3 + 2
    assert compute_mib == 1000 + _MMPROJ_COMPUTE_MB + _DRAFT_COMPUTE_MB
    # Draft affine terms: main constants scaled by size ratio, padded by safety.
    expected_draft = (2 / 10000) * _DRAFT_CTX_SAFETY
    assert kv_factor == pytest.approx(0.5 + 0.5 * expected_draft)
    assert slot_mib == pytest.approx(150.0 + 150.0 * expected_draft)


def test_effective_static_mmproj_only_has_zero_kv_contribution(
        tmp_path, make_model, fit_params_block):
    (tmp_path / "solo-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    m = make_model("solo", **{"derived": dict(fit_params_block),
                              "mmproj": {"file": "solo-mmproj.gguf"}})
    model_mib, kv_factor, slot_mib, compute_mib = m.vram.effective_static(
        fit_bin="unused", cache_type="q8_0")
    assert model_mib == 10003
    assert kv_factor == 0.5          # mmproj adds no per-token cost
    assert slot_mib == 150.0         # ...nor any per-slot cost
    assert compute_mib == 1000 + _MMPROJ_COMPUTE_MB


def test_effective_static_vllm_skips_companions(tmp_path, make_model,
                                                fit_params_block):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x" * 3 * MIB)
    m = make_model("v", backend="vllm", hf_repo="org/model",
                   mmproj={"file": "v-mmproj.gguf"},
                   **{"derived": dict(fit_params_block)})
    assert m.vram.effective_static(fit_bin="unused") == (
        10000, 0.5, 150.0, 1000)


def test_effective_static_result_is_cached(tmp_path, make_model,
                                           fit_params_block, monkeypatch):
    (tmp_path / "c1-mmproj.gguf").write_bytes(b"x" * 2 * MIB)
    m = make_model("c1", **{"derived": dict(fit_params_block),
                            "mmproj": {"file": "c1-mmproj.gguf"}})

    calls = {"n": 0}
    real = m.vram._companion_fit

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(m.vram, "_companion_fit", counting)
    first = m.vram.effective_static(fit_bin="unused")
    second = m.vram.effective_static(fit_bin="unused")
    assert first == second
    assert calls["n"] == 1  # second call served from _effective_cache


# ── baked-in MTP header gate (2026-09-08 GLM-4.7-Flash UD incident) ───────

def _mini_gguf(path, tensor_names):
    """A parseable GGUF header with the given tensor names, no data."""
    import struct
    out = b"GGUF" + struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensor_names)) + struct.pack("<Q", 0)
    for name in tensor_names:
        nb = name.encode()
        out += struct.pack("<Q", len(nb)) + nb
        out += struct.pack("<I", 1) + struct.pack("<Q", 4096)
        out += struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.write_bytes(out)


def test_mtp_gate_blocks_stripped_gguf(tmp_path, make_model):
    m = make_model("a", mtp=True)
    _mini_gguf(m.gguf_path, ["blk.0.attn_q.weight"])
    assert m._mtp_info() == (False, 0)  # declared, but no nextn → off


def test_mtp_gate_keeps_real_nextn(tmp_path, make_model):
    m = make_model("b", mtp=True, mtp_draft_n_max=6)
    _mini_gguf(m.gguf_path, ["blk.0.attn_q.weight",
                             "blk.1.nextn_eh_proj.weight"])
    assert m._mtp_info() == (True, 6)


def test_mtp_gate_undecidable_keeps_declared(tmp_path, make_model):
    (tmp_path / "c.gguf").write_bytes(b"dummy")
    m = make_model("c", mtp=True)
    assert m._mtp_info() == (True, 2)  # unparseable header: declared intent


def test_gguf_has_mtp_layers(tmp_path):
    from llama_packer.utils import gguf_has_mtp_layers
    _mini_gguf(tmp_path / "with.gguf", ["blk.1.nextn_eh_proj.weight"])
    _mini_gguf(tmp_path / "without.gguf", ["blk.0.attn_q.weight"])
    assert gguf_has_mtp_layers(tmp_path / "with.gguf") is True
    assert gguf_has_mtp_layers(tmp_path / "without.gguf") is False
    assert gguf_has_mtp_layers(tmp_path / "absent.gguf") is None
