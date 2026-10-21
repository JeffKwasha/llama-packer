# tests/test_fit_params.py
"""FitParams validation and (de)serialization."""

from __future__ import annotations

from _gguf import write_gguf
from llama_packer.vram import FitParams


def make_block(**overrides) -> dict:
    base = {
        "model_mib": 10000,
        "kv_per_token_mib": 0.5,
        "slot_mib": 0.0,
        "compute_mib": 1000,
        "source": "llama-server",
        "cache_type": "q8_0",
        "shape": "",
    }
    base.update(overrides)
    return base


def test_from_dict_roundtrip():
    # Validates: SPEC.md, 'Measured-Block Persistence': a complete block loads with its terms and its source.
    fp = FitParams.from_dict(make_block(), "q8_0")
    assert fp is not None
    assert fp.model_mib == 10000
    assert fp.kv_per_token_mib == 0.5
    assert fp.slot_mib == 0.0
    assert fp.compute_mib == 1000
    assert fp.source == "llama-server"


def test_from_dict_missing_key_returns_none():
    # Validates: SPEC.md, 'Measured-Block Persistence': an incomplete block is not a measurement.
    assert FitParams.from_dict({"model_mib": 1, "kv_per_token_mib": 1.0},
                               "q8_0") is None


def test_from_dict_non_numeric_returns_none():
    # Validates: SPEC.md, 'Measured-Block Persistence': the derived block holds the measured terms: model_mib, kv_per_token_mib, slot_mib, compute_mib
    assert FitParams.from_dict(make_block(model_mib="x"), "q8_0") is None


def test_from_dict_nonpositive_returns_none():
    # Validates: SPEC.md, 'Measured-Block Persistence': the derived block holds the measured terms: model_mib, kv_per_token_mib, slot_mib, compute_mib
    assert FitParams.from_dict(make_block(model_mib=0), "q8_0") is None
    assert FitParams.from_dict(make_block(kv_per_token_mib=0.0), "q8_0") is None
    assert FitParams.from_dict(make_block(slot_mib=-1.0), "q8_0") is None


def test_from_dict_stale_cache_type_returns_none():
    # Validates: SPEC.md, 'Measured-Block Persistence': a block measured under another cache type is not reused.
    assert FitParams.from_dict(make_block(cache_type="q4_0"), "q8_0") is None


def test_from_dict_non_dict_returns_none():
    # Validates: SPEC.md, 'Measured-Block Persistence': something that is not a block is not a measurement.
    assert FitParams.from_dict("nope", "q8_0") is None


def test_to_dict_roundtrip():
    # Validates: SPEC.md, 'Measured-Block Persistence': the block round-trips, and the legacy keys are gone.
    fp = FitParams(10000, 0.5, 150.0, 1000, "vllm-estimate", "q8_0")
    d = fp.to_dict()
    assert d["source"] == "vllm-estimate"
    assert "parallel" not in d and "ctx_factor" not in d  # legacy keys gone
    fp2 = FitParams.from_dict(d, "q8_0")
    assert fp2 == fp


def test_from_dict_shape_mismatch_returns_none():
    # The compute term depends on batch/attention flags: a block measured
    # under a different flag shape must not be reused.
    # Validates: SPEC.md, 'Measured-Block Persistence': the compute term depends on the batch/attention flags, so a block measured under different flags is not reused.
    assert FitParams.from_dict(make_block(shape="-ub 2048"), "q8_0",
                               shape="") is None
    assert FitParams.from_dict(make_block(shape=""), "q8_0",
                               shape="--flash-attn on") is None


def test_from_dict_shape_match_roundtrips():
    # Validates: SPEC.md, 'Measured-Block Persistence': a block whose shape matches the current flags loads with that shape.
    fp = FitParams.from_dict(make_block(shape="--flash-attn on -b 4096"),
                             "q8_0", shape="--flash-attn on -b 4096")
    assert fp is not None
    assert fp.shape == "--flash-attn on -b 4096"


def test_from_dict_missing_shape_key_returns_none():
    # Pre-shape blocks: the flags they measured under are unknown, so no
    # current shape can vouch for their compute term. Always re-measure.
    # Validates: SPEC.md, 'Measured-Block Persistence': a block with no recorded shape cannot vouch for its compute term; re-measure.
    block = make_block()
    del block["shape"]
    assert FitParams.from_dict(block, "q8_0", shape="") is None


def test_from_dict_shape_not_enforced_for_flag_free_sources():
    # vllm/safetensors estimates are flag-independent: any stored shape is
    # accepted (presence still required).
    # Validates: SPEC.md, 'Measured-Block Persistence': an estimate that does not depend on flags is accepted under any shape.
    fp = FitParams.from_dict(make_block(source="vllm-estimate",
                                        shape="whatever"), "q8_0",
                             shape="")
    assert fp is not None


def test_to_dict_records_shape():
    # Validates: SPEC.md, 'Measured-Block Persistence': the shape the numbers were measured under is recorded with them.
    fp = FitParams(10000, 0.5, 0.0, 1000, "fit-estimate", "q8_0",
                   shape="--flash-attn on")
    assert FitParams.from_dict(fp.to_dict(), "q8_0",
                               shape="--flash-attn on") == fp


def test_vram_mib_affine_prediction():
    # Validates: docs/plans/estimation-functions-facts.md, 'The affine model': the law is model + compute + c*pool + p*D, with the pool per slot times slots.
    fp = FitParams(18114, 0.0332, 150.0, 505, "llama-server", "q8_0")
    # The law: model + compute + c*pool + p*D, pool = per-slot * slots.
    assert fp.vram_mib(ctx_per_slot=32768, parallel=8) == int(
        18114 + 505 + 0.0332 * (32768 * 8) + 150 * 8)
    assert fp.vram_mib(ctx_per_slot=262144, parallel=1) == int(
        18114 + 505 + 0.0332 * 262144 + 150)


# ── invariant (shape-independent) terms ────────────────────────────────────

def test_invariant_from_dict_ignores_shape():
    # Validates: docs/plans/estimation-functions-facts.md, 'Measurement shape and reuse': saved_for(cache_type, shape) rejects blocks whose shape differs
    block = make_block(shape="--flash-attn on -ub 512", source="fit-estimate")
    assert FitParams.invariant_from_dict(block, "q8_0") == (
        10000, 0.5, 0.0, "fit-estimate")


def test_invariant_from_dict_rejects_bad_cache_type():
    # Validates: SPEC.md, 'Measured-Block Persistence': the VRAM constants are invalidated when cache_type changes
    assert FitParams.invariant_from_dict(
        make_block(cache_type="q4_0"), "q8_0") is None


def test_invariant_from_dict_rejects_missing_term():
    # Validates: SPEC.md, 'Measured-Block Persistence': the derived block holds model_mib, kv_per_token_mib, slot_mib, compute_mib, source, cache_type and shape
    block = make_block()
    del block["slot_mib"]
    assert FitParams.invariant_from_dict(block, "q8_0") is None


def test_invariant_from_dict_rejects_unmeasured_source():
    # Validates: SPEC.md, 'Measured-Block Persistence': the block's source records where the numbers came from
    assert FitParams.invariant_from_dict(
        make_block(source="guess"), "q8_0") is None


def test_from_dict_still_rejects_shape_mismatch():
    # The full block is shape-bound for compute; only the invariant terms
    # survive a shape change (see the C7 reuse path below).
    # Validates: SPEC.md, 'Measured-Block Persistence': the full block stays shape-bound for compute; only the invariant terms survive a shape change.
    assert FitParams.from_dict(make_block(shape="-ub 2048"), "q8_0",
                               shape="") is None


# ── weight term from the GGUF header ────────────────────────────────────────

def test_weight_mib_prefers_tensor_bytes(make_model):
    # Validates: docs/plans/vram-estimation-target-context-plan.md, 'C3 — Fix the weight term at the source (highest leverage)': the header's own tensor footprint wins; the estimate is the floor
    model = make_model("w", context_length=32768)
    write_gguf(model.gguf_path, tensor_names=["w.weight"],
               tensor_shape=(2048, 1024))
    assert model.vram._weight_mib(5) == 8       # tensor footprint wins
    assert model.vram._weight_mib(100) == 100   # header estimate is the floor


# ── C7: reuse invariant terms across a shape change ─────────────────────────

def test_fit_params_static_reuses_invariant_on_shape_change(
        make_model, monkeypatch):
    # Validates: docs/plans/vram-estimation-target-context-plan.md, 'C7 — Split shape-invariant from shape-dependent constants': the shape-invariant constants are split from the shape-dependent ones
    model = make_model(
        "c7", context_length=32768,
        derived={
            "model_mib": 10000, "kv_per_token_mib": 0.5, "slot_mib": 100.0,
            "compute_mib": 1000, "source": "fit-estimate",
            "cache_type": "q8_0", "shape": "--flash-attn on -ub 512",
        })
    model.resolve_companions()
    vram = model.vram
    new_shape = "--flash-attn on -ub 256"
    # the saved block's shape differs, so the full block is rejected...
    assert vram.saved_for("q8_0", new_shape) is None
    # ...but its invariant terms are reused and only compute is re-measured.
    monkeypatch.setattr(vram, "_fit_params_compute", lambda *a, **k: 777)
    got = vram.fit_params_static("/bin/fit", "q8_0", llama_args=new_shape)
    assert got is not None
    assert (got.model_mib, got.kv_per_token_mib, got.slot_mib) == (
        10000, 0.5, 100.0)
    assert got.compute_mib == 777
    assert got.shape == new_shape


def test_fit_params_static_shape_change_without_invariant(
        make_model, monkeypatch):
    # Validates: docs/plans/vram-estimation-target-context-plan.md, 'C7 — Split shape-invariant from shape-dependent constants': the shape-invariant constants are split from the shape-dependent ones
    model = make_model(
        "c7b", context_length=32768,
        derived={
            "model_mib": 10000, "source": "fit-estimate",
            "cache_type": "q8_0", "shape": "--flash-attn on -ub 512",
        })
    model.resolve_companions()
    vram = model.vram
    monkeypatch.setattr(vram, "_fit_params_compute", lambda *a, **k: 777)
    # no invariant terms -> the C7 reuse path cannot fire; the full estimate
    # path is attempted (and returns None here: no fit-params binary).
    assert vram.fit_params_static("/bin/fit", "q8_0", llama_args="x") is None
