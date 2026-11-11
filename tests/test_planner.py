# tests/test_planner.py
"""Planner / Profiles / emit_config seams: vision variants, grouping, clamps."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from llama_packer.consts import _RESERVE_VIDEO
from llama_packer.profiles import Profiles
from conftest import entry_for, fake_tasks, only_entry, entry_id, tasks_section
from llama_packer.writer import MatrixKnobs, Planner, emit_config, _solve_matrix_context, _has_chat_slot

TVARS = {"llama_bin": "/opt/llama-server"}


@pytest.fixture
def profiles():
    return Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                     "profiles": {"default": {}, "big": {"temperature": 0.9}}})


def _scripted_ctx(model, by_mmproj, quad=(100, 0.01, 10, 100)):
    """Replace solve_context with a fake keyed on include_mmproj.

    Companion-on variants carry their own VRAM budget (bound to the serving
    view), so the fake is installed on the base budget and every view.
    The affine quad is faked the same way: the planner reports per-variant
    memory tags from effective_static, which must never reach a subprocess
    in hermetic tests.
    """
    def fake(vram_total_mb, *, server_bin=None, parallel=1, spare_mb=0,
             include_mmproj=True, baseline_mb=0, cache_type="q8_0",
             design_ctx=None, **kw):
        return by_mmproj[bool(include_mmproj)]
    model.vram.solve_context = fake
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.solve_context = fake

    def fake_static(fit_bin, cache_type="q8_0", design_ctx=None,
                    include_mmproj=True, llama_args="", allow_cpu=False):
        return quad
    model.vram.effective_static = fake_static
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.effective_static = fake_static


def _vision_model(tmp_path, make_model, name):
    """Model with an mmproj companion; file must exist before construction."""
    (tmp_path / f"{name}-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    return make_model(name, mmproj={"file": f"{name}-mmproj.gguf",
                                    "capabilities": ["image"]})


def _gated_ctx(model, by_mmproj):
    """solve_context fake keyed on include_mmproj: (ctx, over_reason, squeezed).

    Sets the gating flags on the budget that owns the call so plan()'s
    over-budget / squeeze checks see the per-view verdict.
    """
    def make_fake(vram_obj, flag):
        ctx, over, sq = by_mmproj[bool(flag)]

        def fake(*a, **k):
            vram_obj.over_budget_reason = over
            vram_obj.vram_squeezed = sq
            return ctx
        return fake

    def fake_static(*a, **k):
        return (100, 0.01, 10, 100)

    model.vram.solve_context = make_fake(model.vram, False)
    model.vram.effective_static = fake_static
    model.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=100, kv_per_token_mib=0.01, slot_mib=10.0,
        compute_mib=100, source="fit-params")
    on = model.view_for(True)
    if on is not model:
        on.vram.solve_context = make_fake(on.vram, True)
        on.vram.effective_static = fake_static
        on.vram.fit_params_static = model.vram.fit_params_static
    else:
        # No separate companion-on view: one budget, key on the kwarg.
        text_ctx, text_over, text_sq = by_mmproj[False]
        vis_ctx, vis_over, vis_sq = by_mmproj[True]

        def combo(*a, include_mmproj=True, **k):
            if include_mmproj:
                model.vram.over_budget_reason = vis_over
                model.vram.vram_squeezed = vis_sq
                return vis_ctx
            model.vram.over_budget_reason = text_over
            model.vram.vram_squeezed = text_sq
            return text_ctx
        model.vram.solve_context = combo


def test_profiles_spare_precedence():
    # Validates: SPEC.md, 'profiles.yaml': Fleet-wide spare VRAM in MB: ``defaults.spare`` > ``--spare`` > 0.
    p = Profiles({"defaults": {"spare": "2G"}, "profiles": {}})
    assert p.global_spare_mb(vram_total=48000) == 2048
    assert p.global_spare_mb("4G", vram_total=48000) == 2048  # defaults win
    q = Profiles({"profiles": {}})
    assert q.global_spare_mb("512m", vram_total=48000) == 512
    assert q.global_spare_mb() == 0
    assert p.spare_mb(preferred="1G", cli_override="4G", vram_total=48000) == 1024


def test_profiles_groups_fallback_when_nothing_matches(profiles, make_model):
    # Validates: SPEC.md, 'profiles.yaml': a model whose allowed profiles match nothing falls back to the default group
    m = make_model("m", allow_profiles="nomatch")
    groups = profiles.groups_for(m, vram_total=48000)
    # Single defaults-derived group; parallel_for(1)/cache_type_for(default)
    (key, group), = groups.items()
    assert key[:2] == (1, "q8_0")
    assert group == [("default", {"cache_type": "q8_0", "parallel": 1})]


def test_allow_profiles_bad_regex_names_the_sidecar(profiles, make_model,
                                                   caplog):
    # Validates: SPEC.md, 'Usage categories (`usage:`)': a diagnostic about one sidecar leads with the sidecar's name.
    """An invalid `allow_profiles` regex concerns one sidecar's declaration,
    so the warning leads with that sidecar's name."""
    import logging
    m = make_model("m", allow_profiles="(((")
    with caplog.at_level(logging.WARNING):
        profiles.groups_for(m, vram_total=48000)
    warned = [r.message for r in caplog.records
              if "allow_profiles" in r.message]
    assert warned, "expected the invalid regex to warn"
    assert all(r.startswith("m:") for r in warned), warned


def test_profiles_groups_by_cache_type(make_model):
    # Validates: SPEC.md, 'profiles.yaml': differing cache_type values split a model into separate profile groups
    m = make_model("m")
    p = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                  "profiles": {"a": {"cache_type": "f16"},
                               "b": {"cache_type": "q8_0"}}})
    groups = p.groups_for(m, vram_total=48000)
    assert len(groups) == 2
    names = [n for g in groups.values() for n, _ in g]
    assert sorted(names) == ["a", "b"]



def test_plan_unestimable_flag_on_companion_view(make_model, profiles, tmp_path):
    # Validates: SPEC.md, 'Companion (mmproj / MTP) VRAM accounting': Frontmatter view for a serving variant. The companion-off variant is this model itself: overlay keys live.
    """view_for() rebuilds VramBudget per view: a measurement failure
    flagged on the companion-on view's budget must still surface as
    estimate_error metadata — the re-check reads the solving budget."""
    m = _vision_model(tmp_path, make_model, "ue")
    _scripted_ctx(m, {True: 4096, False: 4096})
    on = m.view_for(True)
    on.vram.unestimated_reason = "VRAM measurement failed"
    variants = Planner([m], profiles, fit_bin="unused",
                       vram_total=48 * 1024).plan()["ue"]
    assert variants
    assert all(v.estimate_error == "VRAM measurement failed"
               for v in variants)


def test_profiles_rag_parallel_pinned_to_one(make_model, profiles, caplog):
    # Validates: SPEC.md, 'Residents': residents always serve single-slot: a declared parallel: on a resident is ignored
    """Embed/rerank serve single-slot: a declared parallel is ignored with
    a once-per-model note — resident parallelism must never shrink the
    main chat context it serves."""
    import logging

    emb = make_model("emb", role="embeddings", parallel=8)
    (key, _), = profiles.groups_for(emb, vram_total=48000).items()
    assert key[0] == 1
    assert any("single-slot" in r.message for r in caplog.records)
    # chat keeps its declared parallel
    chat = make_model("chat", parallel=8)
    (ckey, _), = profiles.groups_for(chat, vram_total=48000).items()
    assert ckey[0] == 8


def test_planner_below_min_keeps_vision_no_warning(make_model, profiles, tmp_path,
                                                   caplog):
    # Small design ctx: below min-context with AND without vision — dropping
    # buys nothing, so vision is kept and only an info line is logged.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    m = _vision_model(tmp_path, make_model, "tiny")
    _scripted_ctx(m, {True: 32768, False: 32768})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["tiny"]
    # Dropping the projection buys no context here, so there is no text
    # rung worth an entry: one vision entry, not two identical ones.
    assert len(variants) == 1
    assert variants[0].include_mmproj is True   # main entry keeps vision
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)



def test_bounded_ctx_clamps_design_then_cli(make_model, profiles):
    # Validates: SPEC.md, 'Context Tiers': VRAM-solved context clamped to the model's max trained context.
    m = make_model("m")  # sidecar context_length 32768
    m.vram.solve_context = lambda *a, **k: 999999
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      max_context=16384, min_context=0)
    ctx = planner._bounded_ctx(m, parallel=1, cache_type="q8_0", spare_mb=0,
                               include_mmproj=True, context_length=m.design_context)
    assert ctx == 16384
    planner.max_context = None
    ctx = planner._bounded_ctx(m, parallel=1, cache_type="q8_0", spare_mb=0,
                               include_mmproj=True, context_length=m.design_context)
    assert ctx == 32768


FP_PARAMS = (1000.0, 0.5, 0.0, 100.0)


def test_solve_matrix_excludes_roles_cpu_and_threads_drop_stems(
        make_model, profiles, monkeypatch):
    from llama_packer import writer

    chat_a = make_model("chat_a", role="chat")
    chat_b = make_model("chat_b", role="chat")
    embed = make_model("e", role="embeddings")
    cpu_chat = make_model("cpu", device="cpu")

    seen = {}

    def fake_effective_static(fit_bin, cache_type="q8_0",
                              design_ctx=None, include_mmproj=True, **kw):
        seen[include_mmproj] = seen.get(include_mmproj, 0) + 1
        return FP_PARAMS

    def fake_fit_params_static(fit_bin, cache_type="q8_0", **kw):
        return SimpleNamespace(model_mib=FP_PARAMS[0],
                               kv_per_token_mib=FP_PARAMS[1],
                               slot_mib=FP_PARAMS[2],
                               compute_mib=FP_PARAMS[3], source="fit-params")

    for m in (chat_a, chat_b, embed):
        m.vram.effective_static = fake_effective_static
        m.vram.fit_params_static = fake_fit_params_static

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 24576

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    result = _solve_matrix_context(
        [chat_a, chat_b, embed, cpu_chat], embed, embed,
        fit_bin="unused", vram_total=48 * 1024, spare="1G",
        profiles=profiles, baseline_mb=0, drop_stems={"chat_b"},
    )
    assert result is not None
    assert result.chat_ctx == 24576
    # Only the two real chat models enter the shared budget;
    # chat_b's mmproj is omitted via drop_stems.
    assert seen == {True: 1, False: 1}
    assert len(captured["chat_models"]) == 2
    assert captured["embed_params"] == FP_PARAMS


# ── matrix knobs ──────────────────────────────────────────────────────────


def test_matrix_knobs_defaults_and_overrides():
    # Validates: SPEC.md, 'Usage categories (`usage:`)': Parse ``usage:`` from profiles.yaml. A malformed category is warned about and dropped rather than.
    k = MatrixKnobs.from_cfg(None)
    assert (k.min_chat_ctx, k.tools_min_ctx, k.coload_min_ctx,
            k.ctx_gain_min, k.estimate_headroom) == (
        65536, 131072, 20480, 4096, 1.25)
    k = MatrixKnobs.from_cfg({"min_chat_ctx": 32000, "estimate_headroom": 1.5})
    assert k.min_chat_ctx == 32000
    assert k.estimate_headroom == 1.5


def test_matrix_knobs_retired_context_knobs_warn(caplog):
    # Validates: SPEC.md, 'Tasks': the RAG context is a task max_context.
    with caplog.at_level(logging.WARNING):
        MatrixKnobs.from_cfg({"embed_context": 8192, "rerank_context": 16384})
    assert "embed_context is retired" in caplog.text
    assert "rerank_context is retired" in caplog.text


def test_matrix_knobs_vllm_auto_parallel():
    # Validates: SPEC.md, 'Usage categories (`usage:`)': Parse ``usage:`` from profiles.yaml. A malformed category is warned about and dropped rather than.
    k = MatrixKnobs.from_cfg({})
    assert k.auto_parallel_max == 8
    assert k.vllm_auto_parallel_max == 16
    k = MatrixKnobs.from_cfg({"vllm_auto_parallel_max": 32})
    assert k.vllm_auto_parallel_max == 32
    k = MatrixKnobs.from_cfg({"vllm_auto_parallel_max": 0})
    assert k.vllm_auto_parallel_max == 16


def test_plan_vllm_uncapped_parallel_zero(make_model, profiles):
    # sidecar parallel 0 on a vLLM model = uncapped: ctx solves single-seq
    # (vLLM validates one max-length seq at startup), and the variant emits
    # parallel 0 so --max-num-seqs is omitted.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    m = make_model("uv", backend="vllm", parallel=0)
    _fake_vram(m, (1000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024)
    variants = planner.plan()["uv"]
    assert variants[0].parallel == 0
    assert variants[0].ctx_size == 32768


def test_plan_llama_parallel_zero_warns_and_defaults(make_model, profiles,
                                                     caplog):
    # parallel 0 is a vLLM-only convention; on llama.cpp it's invalid and
    # falls back to the fleet default with a warning.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    m = make_model("m0", parallel=0)
    _fake_vram(m, (1000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024)
    variants = planner.plan()["m0"]
    assert variants[0].parallel == 1
    assert any("parallel 0 is invalid" in r.message for r in caplog.records)


# ── VRAM gating: disable / squeeze-note ──────────────────────────────────


def test_plan_disables_over_budget_model(make_model, profiles, caplog):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a model whose every rung is over budget produces no entry at all
    """Weights exceed the budget: the model cannot load at any context and
    must be skipped — skipped stems never reach entry_ids_by_stem, so
    matrix vars/sets auto-prune (emission happens before var construction)."""
    m = make_model("huge")
    _fake_vram(m, (50000, 0.5, 100))  # 50G weights on a 48G card
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024)
    plan = planner.plan()
    # No variant at all: the model reaches no rung, so the stem never lands
    # in plan and therefore never reaches entry_ids_by_stem -- sets and matrix
    # vars auto-prune. "Emitting what is possible is the goal", so a model
    # that cannot load is absent rather than emitted at a context it cannot
    # hold.
    assert plan.get("huge", []) == []
    assert any("huge" in r.message for r in caplog.records), [
        r.message for r in caplog.records]
    # And it vanishes from emission entirely.
    config = emit_config([m], plan, profiles, TVARS)
    assert config["models"] == {}


def test_rejected_slot_count_is_not_logged_as_a_failure(make_model, caplog):
    # A slot count that leaves too little KV is one rejected candidate; the
    # solver may still find a feasible one.  Only the caller reports a model
    # that serves nowhere.
    import logging
    m = make_model("big")
    _fake_vram(m, (20000, 0.5, 100))
    with caplog.at_level(logging.INFO, logger="llama_packer.vram"):
        ctx = m.vram.solve_context(24576, fit_bin="unused", parallel=4,
                              spare_mb=0, include_mmproj=True,
                              baseline_mb=0, cache_type="q8_0",
                              design_ctx=32768)
    assert ctx == 0
    hits = [r for r in caplog.records if "below the 4096 minimum" in r.message]
    assert hits and all(r.levelno < logging.WARNING for r in hits)


def test_plan_vram_note_with_spare(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a solve squeezed below the design context under a declared spare gains the over-budget note
    """Squeezed below design ctx with --spare active → description note."""
    m = make_model("sq", context_length=32768)
    _fake_vram(m, (32000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare="2G")
    variants = planner.plan()["sq"]
    assert variants
    v = variants[0]
    assert v.vram_note is not None
    assert "over configured VRAM limits" in v.vram_note
    assert "re-run without" in v.vram_note
    assert v.vram_note != ""


def test_plan_no_vram_note_without_spare(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': with no reservation, no note is emitted
    """Same squeeze with no spare/baseline → no note (normal operation
    must never carry over-budget noise)."""
    m = make_model("sq0", context_length=32768)
    _fake_vram(m, (32000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare=None)
    variants = planner.plan()["sq0"]
    assert variants
    assert variants[0].vram_note is None


def test_emit_config_vram_note_reaches_description(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': the over-budget note lands in the entry description
    """End-to-end: plan → emit → the note lands in the entry description
    after the mem tag."""
    m = make_model("sqd", context_length=32768, description="Squeezed model.")
    _fake_vram(m, (32000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare="2G")
    variants = planner.plan()["sqd"]
    config = emit_config([m], {"sqd": variants}, profiles, TVARS)
    desc = only_entry(config)["description"]
    assert "Squeezed model." in desc
    assert "[VRAM" in desc
    assert "over configured VRAM limits" in desc
    # The note trails the mem tag.
    assert desc.index("[VRAM") < desc.index("over configured VRAM limits")


def test_plan_no_vram_note_with_baseline_at_reserve_floor(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a baseline at or below the fixed reserve floor never fires the note
    """hardware.baseline_mb: '128m' is swallowed by max(_RESERVE_VIDEO,
    baseline) — it does not change the reserve, so it must not fire notes."""
    m = make_model("b128", context_length=32768)
    _fake_vram(m, (32000, 0.5, 100))  # still squeezes below design
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare=None, baseline_mb=128)
    variants = planner.plan()["b128"]
    assert variants
    assert variants[0].ctx_size < 32768   # genuinely squeezed
    assert variants[0].vram_note is None  # but no note: 128 <= 1024 floor


def test_plan_vram_note_with_baseline_above_floor(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a baseline above the fixed reserve floor fires the note
    """An explicit baseline above the fixed floor is a real reservation —
    squeeze notes fire even with no --spare."""
    m = make_model("b2g", context_length=32768)
    _fake_vram(m, (32000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare=None,
                      baseline_mb=2 * _RESERVE_VIDEO)
    variants = planner.plan()["b2g"]
    assert variants
    assert variants[0].vram_note is not None
    assert "over configured VRAM limits" in variants[0].vram_note


def test_plan_no_vram_note_when_served_equals_design(make_model, profiles):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': an equal-to-design solve stays silent
    """vram_squeezed can fire against the matrix chat_ctx target while the
    final clamp lands on the model's own context_length — equality must not
    produce a 'serving at N of N' note."""
    m = make_model("eq", context_length=32768)
    _fake_vram(m, (100, 0.01, 10))

    def fake(*a, **k):
        m.vram.over_budget_reason = None
        m.vram.vram_squeezed = True  # spurious: matrix target, not own design
        return 32768

    m.vram.solve_context = fake
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, spare="2G")
    variants = planner.plan()["eq"]
    assert variants
    assert variants[0].ctx_size == 32768
    assert variants[0].vram_note is None


# ── vision over-budget: text-only fallback, companion suppression ─────────


def test_plan_vision_over_budget_falls_back_to_text(tmp_path, make_model,
                                                    profiles, caplog):
    # Validates: SPEC.md, 'Vision is a rung, not a second entry': vision is a rung, not a fallback: there is no vision-to-text retry
    """Vision view cannot load, text-only can: serve text-only instead of
    disabling the whole model.  Drop-pass keeps vision (min-context floor
    equals the over-budget return), so the group solve is what falls back."""
    (tmp_path / "vf-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("vf", mmproj={"file": "vf-mmproj.gguf",
                                 "capabilities": ["image"]},
                   context_length=262144)
    _gated_ctx(m, {
        True: (4096, "weights + compute need 40000 MiB, only 1000 budgeted",
               False),
        False: (131072, None, False),
    })
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    with caplog.at_level(logging.INFO, logger="llama_packer.writer"):
        variants = planner.plan()["vf"]
    assert variants
    v = variants[0]
    assert v.include_mmproj is False
    assert v.ctx_size == 131072
    assert v.estimate_error is None
    assert any("fallback" in r.message and "budgeted" in r.message
               for r in caplog.records), [r.message for r in caplog.records]
    # There is no companion any more: the text-only rung *is* the entry.
    assert not any(r.levelno >= logging.WARNING
                   and r.message.startswith("vf: disabled")
                   for r in caplog.records)



def test_plan_vision_and_text_both_over_budget_disables(tmp_path, make_model,
                                                        profiles):
    """Fallback is not a free pass: if no rung can load, nothing is emitted.

    The mechanism moved from a fleet-wide over-budget gate to
    solve_rung returning None for every shed level, but the contract is
    unchanged: a model that cannot load at any rung produces no entry rather
    than an entry at a context it cannot hold.

    Validates: SPEC.md: 'A category that cannot meet min_context at any rung
    emits nothing.'
    """
    (tmp_path / "vb-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("vb", mmproj={"file": "vb-mmproj.gguf",
                                 "capabilities": ["image"]},
                   context_length=262144)
    _gated_ctx(m, {True: (0, "weights need 40000 MiB", False),
                   False: (0, "weights still need 40000 MiB", False)})
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    assert planner.plan().get("vb", []) == []


def test_auto_parallel_vllm_reaches_higher_cap(make_model, profiles):
    # Same budget, same quads: the vLLM backend may use more slots than the
    # llama.cpp cap — continuous batching makes a slot cost KV only.
    # Pin-free profiles (no parallel in defaults) so auto-parallel runs;
    # an explicit sidecar min_context (cascade row 1) sets the floor so the
    # value function can actually reach both caps.
    # A declared category, so the entry runs the real slot solve
    # (parallel_power) rather than the null category's exponent-0 bias: this
    # test is about backend slot caps, not about the fallback's trade-off.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    no_pin = Profiles({"defaults": {"cache_type": "q8_0"},
                       "profiles": {"default": {}},
                       "tasks": tasks_section(
                           {"qa": {"min_context": "4k",
                                   "min_parallel": 1}})})
    vllm_m = make_model("av", backend="vllm", min_context=4096,
                        tasks=["qa"])
    del vllm_m.frontmatter["context_length"]  # no pin: cap = design
    _fake_vram(vllm_m, (1000, 0.5, 100))
    planner = Planner([vllm_m], no_pin, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    assert planner.plan()["av"][0].parallel == 16

    llama_m = make_model("al", min_context=4096, tasks=["qa"])
    del llama_m.frontmatter["context_length"]
    _fake_vram(llama_m, (1000, 0.5, 100))
    planner = Planner([llama_m], no_pin, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    assert planner.plan()["al"][0].parallel == 8


def test_matrix_knobs_invalid_values_warn_and_default(caplog):
    # Validates: SPEC.md, 'Usage categories (`usage:`)': Parse ``usage:`` from profiles.yaml. A malformed category is warned about and dropped rather than.
    with caplog.at_level(logging.WARNING):
        k = MatrixKnobs.from_cfg({"min_chat_ctx": -1, "coload_min_ctx": "big",
                                  "estimate_headroom": 0.5})
    assert k.min_chat_ctx == 65536
    assert k.coload_min_ctx == 20480
    assert k.estimate_headroom == 1.25
    assert "min_chat_ctx" in caplog.text
    assert "coload_min_ctx" in caplog.text
    assert "estimate_headroom" in caplog.text


# ── opportunistic co-load pass ────────────────────────────────────────────


def _fake_vram(m, params):
    mib, kv_factor, compute = params
    m.vram.effective_static = lambda *a, **k: (mib, kv_factor, 0.0, compute)
    m.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=mib, kv_per_token_mib=kv_factor, slot_mib=0.0,
        compute_mib=compute, source="fit-params")


# ── routing sets are the budget contract ──────────────────────────────────

def _pinned(m, mb):
    m.vram.effective_static = lambda *a, **k: (mb, 0.0, 0.0, 0)
    m.vram.fit_params_static = lambda *a, **k: None
    return m


def test_role_label_set_is_a_chat_set():
    # A set written with role labels is the syntax the router emits; the
    # solve must recognize it as a chat-capable set.
    assert _has_chat_slot("chat & s2t", chat_roles=frozenset({"chat"}))
    assert not _has_chat_slot("s2t & t2s", chat_roles=frozenset({"chat"}))
    assert _has_chat_slot("s2t | chat", chat_roles=frozenset({"chat"}))


def test_role_label_set_reserves_the_worst_branch(profiles, caplog):
    # The chat solve reserves the most expensive model a named role can
    # contribute, not just the selected RAG pair.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    small = _pinned(make("small", role="s2t", vram_mb=640,
                         context_length=8192), 640)
    big = _pinned(make("big", role="s2t", vram_mb=3000,
                       context_length=8192), 3000)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, small, big], embed, rerank,
            fit_bin="unused", vram_total=24 * 1024, spare=None,
            profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"voice": "chat & s2t"})
    assert any("budget follows sets" in r.message
               and "'voice'" in r.message
               and "reserves 3000 MB" in r.message
               for r in caplog.records)


def test_literal_model_id_in_set_is_priced(profiles, caplog):
    # A set may name one model by the id llama-swap publishes; it must be
    # priced like a role, not treated as an unknown zero-cost leaf.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    z = _pinned(make("Zerank-2", role="rerank", vram_mb=3000,
                     context_length=8192), 3000)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, z], embed, rerank, fit_bin="unused",
            vram_total=24 * 1024, spare=None, profiles=profiles,
            knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"ragz": "chat & zerank-2"})
    assert any("reserves 3000 MB" in r.message for r in caplog.records)


def test_cpu_role_in_a_set_costs_nothing(profiles, caplog):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", device="cpu", context_length=8192)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, s2t], embed, rerank, fit_bin="unused",
            vram_total=24 * 1024, spare=None, profiles=profiles,
            knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"voice": "chat & s2t"})
    assert any("reserves 0 MB" in r.message for r in caplog.records)


def test_role_not_in_any_set_costs_nothing(profiles, caplog):
    # A role no chat set names can only run alone, so it must not shrink
    # the chat solve.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    tts = _pinned(make("t", role="t2s", vram_mb=1200,
                       context_length=8192), 1200)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, tts], embed, rerank, fit_bin="unused",
            vram_total=24 * 1024, spare=None, profiles=profiles,
            knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"rag": "chat"})
    assert any("reserves 0 MB" in r.message for r in caplog.records)


def test_category_named_by_its_role_is_in_a_set(profiles, caplog):
    # `stt` is a category; the set names its role.  That is a reference,
    # and must not be reported as "in no set".
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    stt = _pinned(make("parakeet", role="s2t", vram_mb=1900,
                       context_length=8192), 1900)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, stt], embed, rerank, fit_bin="unused",
            vram_total=24 * 1024, spare=None, profiles=profiles,
            knobs=MatrixKnobs(min_chat_ctx=8192),
            fixed_categories=[("stt", stt)],
            routing_sets={"voice": "chat & s2t"})
    assert not any("in no set" in r.message for r in caplog.records)


def test_category_ctx_caps_at_one_seam(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #92
    # ctx is a cap honoured by the served context and never a floor.
    from llama_packer.tasks import TaskCatalogue
    from llama_packer.writer import resident_ctx
    tasks = TaskCatalogue.from_cfg({
        "embeddings": {"endpoint": "openai.embeddings", "max_context": 8192}})
    m = make("stt2", role="s2t", context_length=32768, device="cpu")
    assert resident_ctx(m, tasks, 2048) == 2048
    assert resident_ctx(m, tasks, 65536) == 32768
    emb = make("emb2", role="embeddings", context_length=65536,
               tasks=["embeddings"])
    assert resident_ctx(emb, tasks) == 8192
    assert resident_ctx(emb, tasks, 1024) == 1024
    assert resident_ctx(emb, tasks, 999999) == 8192
    chat = make("chat2", role="chat", context_length=65536, tasks=["chat"])
    assert resident_ctx(chat, tasks) == 65536


def test_host_unit_refusal_drops_a_category_that_cannot_fit_alone(
        profiles, monkeypatch):
    # Validates: docs/plans/requirement-trace.md, requirement #70
    # Each fixed CPU category is judged on its own quad against what is
    # free after the committed pair: the one that fits still emits an id
    # and commits to the ledger, the one that does not is refused outright.
    from llama_packer import writer
    chat = make("chat", role="chat")
    emb = make("e", role="embeddings", context_length=32768)
    rnk = make("r", role="rerank", context_length=32768)
    tts = make("t", role="t2s", context_length=32768, device="cpu")
    music = make("muz", role="s2t", context_length=32768, device="cpu")
    _fake_vram(chat, (1000, 0.5, 100))
    for m in (emb, rnk):
        _fake_vram(m, (100, 0.01, 0))
    _fake_vram(tts, (1000, 0.0, 0))
    _fake_vram(music, (9000, 0.0, 0))
    monkeypatch.setattr(writer, "solve_matrix_ctx", lambda **kw: 8192)
    planner = writer.Planner(
        [chat, emb, rnk, tts, music], profiles, fit_bin="unused",
        vram_total=64 * 1024, matrix_cfg={"sets": {}},
        embed_model=emb, rerank_model=rnk,
        fixed_categories=[("tts", tts), ("stt", music)],
        host_pool_mb=5000)
    plan = planner.plan()
    assert planner.host_unit_refused_vars == frozenset({"stt"})
    assert "t" in plan and "muz" not in plan


def test_resident_options_branch_alternatives_not_co_resident():
    # judged per and-branch: an alternation never reports same-role
    # co-residency, only a genuine conjunction does.
    from llama_packer.writer import _resident_options
    assert sorted(map(sorted, _resident_options(
        "three | (two & one)"))) == [["one", "two"], ["three"]]
    assert sorted(map(sorted, _resident_options(
        "a & (b | c)"))) == [["a", "b"], ["a", "c"]]


def _bucket(stem, mb, role="s2t"):
    """A fixed-overhead category with an authoritative pinned size."""
    m = make(stem, role=role, backend="whisper-server", vram_mb=mb,
             context_length=8192)
    m.vram.effective_static = lambda *a, **k: (mb, 0.0, 0.0, 0)
    m.vram.fit_params_static = lambda *a, **k: None
    return m


def _solve_sets(profiles, cats, sets, vram_total=32 * 1024):
    chat = make("chat", role="chat", context_length=262144)
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    return _solve_matrix_context(
        [chat, embed, rerank], embed, rerank, fit_bin="unused",
        vram_total=vram_total, spare=None, profiles=profiles,
        knobs=MatrixKnobs(min_chat_ctx=8192), fixed_categories=cats,
        routing_sets=sets)


def test_solve_matrix_reserves_worst_set_not_category_union(profiles):
    # charged the worst branch of the most expensive chat set, not the sum of
    # every declared category.  (three | (two & one)) costs max(4000, 3000).
    cats = [("one", _bucket("one", 1000)),
            ("two", _bucket("two", 2000)),
            ("three", _bucket("three", 4000))]
    composed = _solve_sets(profiles, cats, {
        "rag": "chat & emb & rnk",
        "combo": "chat & (three | (two & one))"})
    only_three = _solve_sets(profiles, cats, {
        "combo": "chat & three"})
    union = _solve_sets(profiles, cats, {
        "all": "chat & emb & rnk & one & two & three"})
    assert composed is not None and only_three is not None and union is not None
    assert composed.chat_ctx == only_three.chat_ctx
    assert composed.chat_ctx > union.chat_ctx


def test_solve_matrix_prices_a_nested_and_branch(profiles):
    # costs two + three when that branch is larger, so a nested group is never
    # priced as an unknown var worth 0.
    cats = [("one", _bucket("one", 1000)),
            ("two", _bucket("two", 2000)),
            ("three", _bucket("three", 4000))]
    nested = _solve_sets(profiles, cats,
                         {"s": "chat & (one | (two & three))"})
    flat = _solve_sets(profiles, cats,
                       {"s": "chat & (two & three)"})
    assert nested is not None and flat is not None
    assert nested.chat_ctx == flat.chat_ctx


def test_solve_matrix_unreferenced_category_reserves_nothing(profiles, caplog):
    # names can only run alone, so it must not shrink the chat solve.
    cats = [("three", _bucket("three", 4000))]
    sets = {"rag": "chat & emb & rnk"}
    with caplog.at_level("INFO"):
        referenced = _solve_sets(profiles, [], sets)
        unreferenced = _solve_sets(profiles, cats, sets)
    assert referenced is not None and unreferenced is not None
    assert unreferenced.chat_ctx == referenced.chat_ctx
    assert any("in no set" in r.message for r in caplog.records)


def test_resident_cost_detail_names_the_terms(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #87
    # prints is the same quad the budget charges, broken into its terms.
    from llama_packer.writer import resident_cost_detail
    m = make("jina", role="embeddings", context_length=4096)
    m.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=466, kv_per_token_mib=238 / 4096, slot_mib=0.0,
        compute_mib=346, source="fit-params")
    total, detail = resident_cost_detail(
        m, fit_bin="unused", cache_type="q8_0", profiles=profiles,
        llama_args="", ctx=4096)
    assert total == 1050
    assert detail == "466 weights + 346 compute + 238 kv @4096"


def test_role_in_a_set_is_reserved_other_roles_are_not(profiles, caplog):
    # A role named in a chat set is a resident the solve must afford; a role
    # no set names can only run alone and costs nothing.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = _pinned(make("s2t", role="s2t", vram_mb=640,
                       context_length=8192), 640)
    img = _pinned(make("img", role="image", vram_mb=40000,
                       context_length=8192), 40000)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        _solve_matrix_context(
            [chat, embed, rerank, s2t, img], embed, rerank,
            fit_bin="unused", vram_total=24 * 1024, spare=None,
            profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"voice": "chat & s2t"})
    assert any("reserves 640 MB" in r.message for r in caplog.records)


def test_set_budget_is_not_gated_by_a_chat_floor(profiles, caplog):
    # The set is the contract: if it permits a resident, the solve reserves
    # it even when that squeezes chat.  There is no "does not fit below the
    # floor, not reserved" pass in set mode.
    chat = make("chat", role="chat", capabilities=["tools"])
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    img = _pinned(make("img", role="image", vram_mb=20000,
                       context_length=8192), 20000)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        result = _solve_matrix_context(
            [chat, embed, rerank, img], embed, rerank, fit_bin="unused",
            vram_total=24 * 1024, spare=None, profiles=profiles,
            knobs=MatrixKnobs(tools_min_ctx=20000),
            routing_sets={"combo": "chat & image"})
    assert result is not None
    assert not any("not reserved" in r.message for r in caplog.records)


def test_solve_matrix_reads_block_tokens_from_on_view(
        tmp_path, make_model, profiles, monkeypatch):
    # image_max_tokens living only in the mmproj block must still reserve
    # the image floor in the matrix solve (read via the companion-on view).
    from llama_packer import writer
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x" * 1024)
    m = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                "capabilities": ["image"],
                                "image_max_tokens": 9000})
    assert m.image_max_tokens is None  # base frontmatter: no floor visible
    e = make_model("e", role="embeddings", context_length=8192)
    r = make_model("r", role="rerank", context_length=8192)
    _fake_vram(m.view_for(True), (8000, 0.4, 500))
    _fake_vram(e, (500, 0.1, 100))
    _fake_vram(r, (500, 0.1, 100))
    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 8192

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)
    result = _solve_matrix_context(
        [m, e, r], e, r, fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192),
        routing_sets={"rag": "chat & embeddings & rerank"},
    )
    assert result is not None
    floors = {mod.stem: floor for mod, _, _, _, _, _, floor
              in captured["chat_models"]}
    assert floors == {"v": 9000}


def test_solve_matrix_squeeze_adopted(profiles):
    from llama_packer.tasks import TaskCatalogue
    tasks = TaskCatalogue.from_cfg({
        "embeddings": {"endpoint": "openai.embeddings",
                       "max_context": 32768},
        "rerank": {"endpoint": "openai.rerank", "max_context": 32768},
    })
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=32768,
                 tasks=["embeddings"])
    rerank = make("r", role="rerank", context_length=32768, tasks=["rerank"])
    _fake_vram(chat, (4000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.5, 100))
    baseline = _solve_matrix_context(
        [chat, embed, rerank], embed, rerank,
        fit_bin="unused", vram_total=64 * 1024, spare=None,
        profiles=profiles, tasks=tasks,
        knobs=MatrixKnobs(ctx_gain_min=10 ** 9),
        routing_sets={"rag": "chat & embeddings & rerank"},
    )
    result = _solve_matrix_context(
        [chat, embed, rerank], embed, rerank,
        fit_bin="unused", vram_total=64 * 1024, spare=None,
        profiles=profiles, tasks=tasks,
        routing_sets={"rag": "chat & embeddings & rerank"},
    )
    assert baseline is not None and result is not None
    assert baseline.squeeze is False
    assert result.squeeze is True
    assert (result.embed_ctx, result.rerank_ctx) == (20480, 20480)
    assert result.chat_ctx > baseline.chat_ctx


def make(stem, **fm):
    """Minimal Model-like stub for the solve tests (no filesystem)."""
    from llama_packer.model import Model
    from pathlib import Path
    import tempfile
    d = tempfile.mkdtemp()
    (Path(d) / f"{stem}.gguf").write_bytes(b"x")
    fm = {"name": stem, "context_length": 32768, **fm}
    m = Model(Path(d) / f"{stem}.md", fm)
    m.tasks_resolved = fake_tasks(fm)
    return m


# ── plan(): demotion, squeeze clamp, coload flags ─────────────────────────


def test_plan_threads_matrix_result_into_variants(profiles, monkeypatch):
    # Validates: SPEC.md, 'Output Format': the matrix result is threaded into the emitted variants
    chat = make("chat", role="chat", capabilities=["tools"])
    embed = make("e", role="embeddings", context_length=32768)
    rerank = make("r", role="rerank", context_length=32768)
    s2t = make("s2t", role="s2t", vram_mb=640, context_length=8192)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.5, 100))
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None

    # solve_context: echo the design_ctx the planner proposes, else the model's own.
    def fake_solve_context(*a, **kw):
        return kw.get("design_ctx") or 32768

    for m in (chat, embed, rerank, s2t):
        m.vram.solve_context = fake_solve_context

    planner = Planner([chat, embed, rerank, s2t], profiles, fit_bin="unused",
                      vram_total=64 * 1024,
                      matrix_cfg={"min_chat_ctx": 8192,
                                  "embed_context": 32768,
                                  "rerank_context": 32768},
                      embed_model=embed, rerank_model=rerank)
    plan = planner.plan()

    assert planner.matrix_result is not None
    # squeeze adopted (chat 22528 -> 32768), RAG entries served at 20480.
    assert planner.matrix_result.squeeze
    assert plan["e"][0].ctx_size == 20480
    assert plan["r"][0].ctx_size == 20480
    # chat solved below tools_min_ctx (131072) -> demoted.
    assert plan["chat"][0].tools_demoted is True

    config = emit_config([chat, embed, rerank, s2t], plan, profiles, TVARS)
    chat_entry = entry_for(config, "chat")
    assert chat_entry["capabilities"]["tools"] is False
    assert chat_entry["metadata"]["tools_demoted"] is True
    assert entry_for(config, "s2t")["metadata"]["ctx_size"] == 8192


def test_plan_demotes_tools_below_the_threshold_without_a_matrix(profiles):
    """Served at 32768 against a 131072 tools threshold, it must be demoted.

    This asserted the opposite. Demotion keyed off ``self.chat_ctx``, which
    is None whenever no ``matrix:`` section exists -- the default -- so
    demotion could never fire, and an entry advertised tool calling an order
    of magnitude below the window it needs. The verdict now comes from the
    context this entry actually serves.

    Validates: docs/plans/requirement-trace.md #30: tools is demoted on the entry's
    own served context.
    """
    chat = make("chat", role="chat", capabilities=["tools"])
    chat.vram.solve_context = lambda *a, **k: 32768
    planner = Planner([chat], profiles, fit_bin="unused", vram_total=64 * 1024)
    plan = planner.plan()
    assert plan["chat"][0].tools_demoted is True
    config = emit_config([chat], plan, profiles, TVARS)
    entry = only_entry(config)
    assert entry["capabilities"]["tools"] is False
    assert entry["metadata"]["tools_demoted"] is True


def test_plan_keeps_tools_when_the_context_clears_the_threshold(profiles):
    """The other direction, so the fix is not simply "always demote".

    Validates: docs/plans/requirement-trace.md #30: the other direction, so the fix
    cannot read as 'always demote'.
    """
    # make() seeds context_length 32768, which would cap the served ctx
    # below the threshold and demote regardless of what solve_context returns.
    chat = make("chat", role="chat", capabilities=["tools"],
                context_length=262144)
    chat.vram.solve_context = lambda *a, **k: 131072
    planner = Planner([chat], profiles, fit_bin="unused", vram_total=64 * 1024)
    plan = planner.plan()
    assert plan["chat"][0].tools_demoted is False
    config = emit_config([chat], plan, profiles, TVARS)
    assert only_entry(config)["capabilities"]["tools"] is True
    assert "tools_demoted" not in only_entry(config)["metadata"]


def test_set_resident_estimate_gets_headroom(profiles, caplog):
    # No pin, no fit-params: the file-size + buffer estimate is padded by
    # estimate_headroom (1.25x default) so a bad guess errs toward reserving.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", backend="whisper-server",
               context_length=8192)
    s2t.gguf_path.write_bytes(b"x" * (1024 * 1024))  # 1 MB file
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    with caplog.at_level("INFO"):
        result = _solve_matrix_context(
            [chat, embed, rerank, s2t], embed, rerank,
            fit_bin="unused", vram_total=48 * 1024, spare=None,
            profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192),
            routing_sets={"voice": "chat & s2t"},
        )
    assert result is not None
    # s2t overhead = (1 + 100) * 1.25 = 126 (the set names no RAG pair).
    assert any("reserves 126 MB" in r.message for r in caplog.records)


# ── unestimable models: conservative bound, never a crash ─────────────────


def test_plan_unestimable_chat_floor_and_metadata(make_model, profiles):
    # Validates: SPEC.md, 'Model Metadata': a model whose VRAM cannot be estimated is marked estimated: false with the reason, and falls back to the floor
    from llama_packer.consts import ESTIMATE_ERROR_REASON

    # No sidecar context_length: the floor cascade (no pin, no tools) is
    # half the design context.
    m = make("ue")
    del m.frontmatter["context_length"]
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["ue"]
    assert len(variants) == 1
    v = variants[0]
    assert v.ctx_size == 16384
    assert v.estimate_error == ESTIMATE_ERROR_REASON
    config = emit_config([m], {"ue": variants}, profiles, TVARS)
    meta = only_entry(config)["metadata"]
    assert meta["estimated"] is False
    assert meta["estimate_error"] == ESTIMATE_ERROR_REASON
    assert meta["ctx_size"] == 16384


def test_matrix_unestimable_chat_rides_free(profiles, monkeypatch):
    # Validates: SPEC.md, 'Remainder': the chat budget is what is left after the residents are charged
    from llama_packer import writer

    good = make("good", role="chat")
    bad = make("bad", role="chat")
    embed = make("e", role="embeddings")
    for m in (good, embed):
        _fake_vram(m, (1000, 0.5, 100))
    bad.vram.effective_static = lambda *a, **k: None
    bad.vram.fit_params_static = lambda *a, **k: None

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 8192

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    planner = Planner([good, bad, embed], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=embed)
    plan = planner.plan()
    # The unestimable chat model stays in the shared solve but rides free:
    # zero quad — no expansion of the set's measured allocation.
    assert len(captured["chat_models"]) == 2
    assert captured["chat_models"][1][1:5] == (0, 0.0, 0.0, 0)
    assert planner.synthetic_quads["bad"] == (0, 0.0, 0.0, 0)
    assert bad.vram.unestimated_reason is not None
    # It serves at the shared solved context and is flagged.
    v = plan["bad"][0]
    assert v.ctx_size == 8192
    assert v.estimate_error is not None
    # The measurable chat model plans normally at the same shared ctx.
    assert plan["good"][0].ctx_size == 8192
    assert plan["good"][0].estimate_error is None


def test_matrix_unestimable_embed_clamped_and_flagged(profiles, monkeypatch):
    # Validates: SPEC.md, 'Remainder': a resident that cannot be estimated is clamped to its fixed context and flagged
    from llama_packer import writer

    from llama_packer.endpoints import EMBEDDINGS, RERANK
    from llama_packer.profiles import Profiles
    profiles = Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
        "tasks": {"embeddings": {"endpoint": EMBEDDINGS,
                                 "max_context": 4096},
                  "rerank": {"endpoint": RERANK, "max_context": 4096}}})
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=32768,
                 tasks=["embeddings"])
    rerank = make("r", role="rerank", context_length=32768, tasks=["rerank"])
    _fake_vram(chat, (8000, 0.5, 500))
    for m in (embed, rerank):
        m.vram.effective_static = lambda *a, **k: None
        m.vram.fit_params_static = lambda *a, **k: None

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 12000

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    planner = Planner([chat, embed, rerank], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=rerank)
    plan = planner.plan()
    # Riders are charged nothing extra (zero quads) and serve at the
    # configured RAG cap (4096), not their design context.
    assert captured["embed_params"] == (0, 0.0, 0.0, 0)
    assert captured["rerank_params"] == (0, 0.0, 0.0, 0)
    assert captured["embed_ctx"] == 4096
    assert captured["rerank_ctx"] == 4096
    assert "e" in planner.synthetic_quads and "r" in planner.synthetic_quads
    for stem in ("e", "r"):
        v = plan[stem][0]
        assert v.ctx_size == 4096
        assert v.estimate_error is not None


def test_solve_matrix_aborts_when_nothing_measurable(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings")
    chat.vram.effective_static = lambda *a, **k: None
    embed.vram.fit_params_static = lambda *a, **k: None
    result = _solve_matrix_context(
        [chat], embed, embed,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles,
        routing_sets={"rag": "chat & embeddings & rerank"},
    )
    assert result is None


def test_planner_skips_text_variant_that_gains_nothing(make_model, profiles,
                                                       tmp_path):
    # A model already at its design cap cannot reclaim context by shedding
    # the projection, so the text rung earns no entry. Regression: ten
    # models shipped a -text twin at identical context (2026-09-26).
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    (tmp_path / "vis-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("vis", mmproj={"file": "vis-mmproj.gguf",
                                  "capabilities": ["image"]},
                   context_length=262144)
    # Same context with and without the projection: nothing to gain.
    _scripted_ctx(m, {True: 200000, False: 200000})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["vis"]
    assert len(variants) == 1
    assert variants[0].include_mmproj is True


def _quiet_logger():
    import logging
    log = logging.getLogger("test-vars")
    log.addHandler(logging.NullHandler())
    return log


# ── solve_rung: most context that fits, vision kept only when free ───────


def _affording(model, by_shed):
    """_bounded_ctx keyed on include_mmproj, in MiB-equivalent ctx terms."""
    def fake(view, *, parallel, include_mmproj, **_kw):
        return by_shed[bool(include_mmproj)]
    planner = _single_planner(model)
    planner._bounded_ctx = fake
    return planner


def _single_planner(model):
    from llama_packer.writer import Planner
    p = Planner.__new__(Planner)
    p.vram_total = 48 * 1024
    p.fit_bin = "unused"
    p.baseline_mb = 0
    p.memory_margin = 0.0
    p.profiles = SimpleNamespace(default_cache_type="q8_0",
                                 shed_order=("vision", "draft"))
    p.llama_args = ""
    p.max_context = None
    p.min_context = 65536
    p.min_context_explicit = False
    p.knobs = MatrixKnobs()
    return p


def _demand(name="coding", **kw):
    from llama_packer.endpoints import CHAT
    from llama_packer.tasks import Task
    return Task(name, CHAT, "chat", **kw)


def test_rung_drops_vision_when_it_buys_context(make_model):
    """Both levels qualify here, which is what makes this a test of the
    *ranking* and not of the shed order: with only one level legal, any
    implementation that simply takes the first hit passes too.

    Validates: SPEC.md, 'Shedding is described per category': most context
    that fits wins.
    """
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 140000, False: 200000})
    rung = p.solve_rung(m, _demand(min_context=131072), parallel=1,
                        cache_type="q8_0", spare_mb=0)
    assert rung.shed == frozenset({"vision"})
    assert rung.ctx == 200000


def test_rung_keeps_vision_when_it_is_free(make_model):
    """Plentiful VRAM: text-only buys nothing, so the projector stays.

    "coding ... only adds vision if VRAM is sufficient for MAX DESIGN
    CONTEXT + VISION." Both levels afford the same context, so the more
    capable one wins instead of being discarded for no gain.

    Validates: SPEC.md, 'Shedding is described per category': 'only adds vision if VRAM
    is sufficient for MAX DESIGN CONTEXT + VISION'.
    """
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 200000, False: 200000})
    rung = p.solve_rung(m, _demand(min_context=131072), parallel=1,
                        cache_type="q8_0", spare_mb=0)
    assert rung.shed == frozenset()
    assert rung.ctx == 200000


def test_rung_is_none_when_the_category_cannot_be_served(make_model):
    """No id may claim a category the fleet cannot deliver.

    Validates: SPEC.md: 'A category that cannot meet min_context at any rung emits
    nothing.'
    """
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 40960, False: 40960})
    assert p.solve_rung(m, _demand(min_context=131072), parallel=1,
                        cache_type="q8_0", spare_mb=0) is None


def test_rung_honours_keep_then_falls_back_to_vision(make_model):
    """keep: [draft] stops the ladder shedding MTP, so it moves to vision.

    Validates: SPEC.md: 'keep only forbids shedding'.
    """
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 90000, False: 200000})
    rung = p.solve_rung(
        m, _demand(min_context=131072, keep=frozenset({"draft"})),
        parallel=1, cache_type="q8_0", spare_mb=0)
    assert "draft" not in rung.shed
    assert rung.shed == frozenset({"vision"})


def test_fallback_keeps_capability_until_the_minimum_forces_it(make_model):
    """Capability first; context only when there is none left to protect.

    Taking the bare 4096 and keeping the projector is the point: the
    fallback has no goals, so giving up a capability for context the
    category never asked for would be a pointless trade. It sheds only when
    holding the capability would leave no legal serving at all.

    Validates: docs/plans/requirement-trace.md, requirement #11.
    """
    from llama_packer.writer import fallback_task
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 4096, False: 262144})
    fallback = fallback_task(m)
    rung = p.solve_rung(m, fallback, parallel=1, cache_type="q8_0",
                        spare_mb=0, prefer_capability=True)
    assert rung.shed == frozenset(), "shed while a legal serving existed"
    assert rung.ctx == 4096, "took the minimum, keeping the projector"

    # Plentiful: the projector costs nothing, so the cap is served outright.
    generous = _affording(m, {True: 262144, False: 262144})
    assert generous.solve_rung(m, fallback, parallel=1, cache_type="q8_0",
                               spare_mb=0,
                               prefer_capability=True).ctx == 262144


def test_fallback_sheds_when_nothing_else_fits(make_model):
    """The other direction: no legal serving with the capability attached.

    Validates: docs/plans/requirement-trace.md, requirement #12.
    """
    from llama_packer.writer import fallback_task
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 0, False: 4096})
    rung = p.solve_rung(m, fallback_task(m), parallel=1,
                        cache_type="q8_0",
                        spare_mb=0, prefer_capability=True)
    assert rung is not None, "the fallback must still emit something"
    assert "vision" in rung.shed, "kept a capability it could not afford"
    assert rung.ctx >= 4096


def test_keep_draft_is_enforced_now_that_the_axis_exists(make_model):
    """Dropping the draft must buy context, and `keep` must prevent it.

    Until the draft axis existed this was unfalsifiable: MTP was always
    costed and never shed, so `keep: [draft]` was correct by accident rather
    than by design. With the axis real, a category without `keep` has to be
    able to shed it -- otherwise there is nothing for `keep` to protect.

    Validates: SPEC.md: 'require and keep are different and both are honoured'.
    """
    m = make_model("c", backend="llama-server", context_length=262144)
    seen = []

    def fake(view, *, parallel, include_mmproj, include_draft=True, **_kw):
        seen.append((bool(include_mmproj), bool(include_draft)))
        return 90000 if (include_mmproj and include_draft) else 200000

    p = _single_planner(m)
    p._bounded_ctx = fake
    p.solve_rung(m, _demand(min_context=131072), parallel=1,
                 cache_type="q8_0", spare_mb=0)
    # Draft offered up: (vision, draft) -> (no vision, draft) -> ... -> neither
    assert (False, False) in seen, seen

    seen.clear()
    rung = p.solve_rung(m, _demand(min_context=131072,
                                   keep=frozenset({"draft"})),
                        parallel=1, cache_type="q8_0", spare_mb=0)
    assert all(draft for _, draft in seen), "keep: [draft] still shed it"
    assert "draft" not in rung.shed
    assert rung.shed == frozenset({"vision"})


# ── one entry per category; the fallback is the floor on that ─────────────


def test_every_category_gets_an_entry(make_model, profiles):
    """Each satisfiable category emits its own id.

    Validates: docs/plans/requirement-trace.md #3 -- `usage: [coding, chat]`
    on a model that can serve both produces a `.coding` and a `.chat`, so a
    client can ask for the one it wants.
    """
    m = make_model("multi", role="chat")
    m.frontmatter["tasks"] = ["coding", "chat"]
    profs = Profiles({"defaults": {"cache_type": "q8_0"},
                      "profiles": {"default": {}},
                      "tasks": tasks_section(
                          {"coding": {"min_context": "4k"},
                           "chat": {"min_context": "4k"}})})
    planner = Planner([m], profs, fit_bin="unused", vram_total=48 * 1024,
                      min_context=4096)
    planner._bounded_ctx = lambda *a, **k: 131072
    planner._auto_parallel = lambda *a, **k: (1, 131072)
    cats = {v.category.id for v in planner.plan()["multi"] if v.category}
    assert cats == {"coding", "chat", ""}, cats
    ids = list(emit_config([m], planner.plan(), profs, TVARS)["models"])
    assert {"multi.coding", "multi.chat", "multi"} <= set(ids), ids


def test_every_model_gets_a_fallback(make_model, profiles):
    """A category cannot satisfy every model, but the fallback always can.

    Validates: docs/plans/requirement-trace.md #5 -- the fallback is the
    floor on "a category that cannot be met emits nothing", so an untagged
    model still reaches clients.
    """
    m = make_model("lonely", role="chat")
    m.frontmatter["tasks"] = ["coding"]
    # coding asks for 1M, which nothing can serve; the fallback asks 4096.
    profs = Profiles({"defaults": {"cache_type": "q8_0"},
                      "profiles": {"default": {}},
                      "tasks": tasks_section(
                          {"coding": {"min_context": 1000000}})})
    planner = Planner([m], profs, fit_bin="unused", vram_total=48 * 1024,
                      min_context=4096)
    planner._bounded_ctx = lambda *a, **k: 131072
    planner._auto_parallel = lambda *a, **k: (1, 131072)
    variants = planner.plan()["lonely"]
    assert [v.category.id if v.category else "" for v in variants] == [""]
    assert list(emit_config([m], {"lonely": variants},
                            profs, TVARS)["models"]) == ["lonely"]


def test_model_with_no_rung_at_all_emits_nothing(make_model, profiles):
    """Even the fallback can fail. Then the model is absent, not emitted at
    a context it cannot hold.

    Validates: docs/plans/requirement-trace.md #6 -- the fallback can itself
    fail and emit nothing.
    """
    m = make_model("huge", role="chat")
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024)
    planner._bounded_ctx = lambda *a, **k: 0
    assert planner.plan().get("huge", []) == []


def test_category_prefers_context_over_capability(make_model):
    """A category takes the rung buying the MOST context.

    The mirror of the fallback's rule, and the reason the two differ: coding
    would rather have 262144 tokens text-only than 131072 with a projector.

    Validates: docs/plans/requirement-trace.md #10.
    """
    coding = _demand(min_context=131072)
    m = make_model("c", backend="llama-server", context_length=262144)
    p = _affording(m, {True: 131072, False: 262144})
    rung = p.solve_rung(m, coding, parallel=1, cache_type="q8_0", spare_mb=0)
    assert rung.shed == frozenset({"vision"}), "kept the projector"
    assert rung.ctx == 262144, "took the text-only rung for the context"


def test_require_prevents_the_entry_emitted(make_model, profiles, tmp_path):
    """`require` gates existence end to end, not just the helper.

    `Planner.can_meet` is the unit; this is the behaviour a client sees: a
    model tagged with a category requiring vision, having no projector, gets
    **no id at all** for that category -- and the fallback carries on.

    Validates: docs/plans/requirement-trace.md #16.
    """
    (tmp_path / "t-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("blind", role="chat")          # text only, no mmproj
    m.frontmatter["tasks"] = ["qa"]
    profs = Profiles({"defaults": {"cache_type": "q8_0"},
                      "profiles": {"default": {}},
                      "tasks": tasks_section(
                          {"qa": {"min_context": "4k",
                                  "require": ["vision"]}})})
    planner = Planner([m], profs, fit_bin="unused", vram_total=48 * 1024,
                      min_context=4096)
    planner._bounded_ctx = lambda *a, **k: 131072
    planner._auto_parallel = lambda *a, **k: (1, 131072)
    names = {v.category.id if v.category else "" for v in planner.plan()["blind"]}
    assert "qa" not in names, "emitted an id claiming a capability it lacks"
    assert "" in names, "the fallback must still serve it"


def test_shed_order_is_configurable(make_model):
    """What a category omits is shed in the `shed:` order, not a constant.

    Validates: docs/plans/requirement-trace.md #14 -- capabilities a
    category omits are shed last, in the order from `shed:`.
    """
    from types import SimpleNamespace

    from llama_packer.writer import Planner
    p = Planner.__new__(Planner)
    p.profiles = SimpleNamespace(shed_order=("draft", "vision"))
    from llama_packer.writer import shed_levels
    levels = shed_levels((), default_order=p.profiles.shed_order)
    assert levels == (frozenset(), frozenset({"draft"}),
                      frozenset({"draft", "vision"}))


def test_fallback_slot_exponent_is_zero(make_model):
    """The fallback runs the slot solve with slots worth nothing.

    Exponent 0 is not a special case for "parallel 1": it is the ordinary
    solve with a bias that says a slot is free to take *if the context still
    holds*. That is why it is worth expressing as a power rather than a
    branch.

    Validates: docs/plans/requirement-trace.md #35.
    """
    m = make_model("c", backend="llama-server", role="chat")
    seen = []
    from llama_packer.profiles import Profiles as _P
    profs = _P({"defaults": {"cache_type": "q8_0"}, "profiles": {"default": {}}})
    planner = Planner([m], profs, fit_bin="unused", vram_total=48 * 1024)
    real = planner._auto_parallel

    def spy(*a, **kw):
        seen.append(kw.get("power"))
        return real(*a, **kw)

    planner._auto_parallel = spy
    planner._bounded_ctx = lambda *a, **k: 131072
    planner.plan()
    # The fallback's call passes power=0.0 explicitly.
    assert 0.0 in seen, seen


def test_ties_go_to_more_slots(make_model):
    """Equal score resolves upward, not to the first candidate.

    Validates: docs/plans/requirement-trace.md #36 -- ties in the slot
    score go to more slots, which is what lets a plentiful-VRAM model keep
    its context *and* take concurrency.
    """
    from test_auto_parallel import _planner as _ap, _scripted_by_parallel
    m = make_model("c", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384, 4: 8192})
    variants = _ap(m).plan()["c"]
    assert variants[0].parallel == 2, "tie must not fall to parallel 1"


# ── residents: fixed context, single slot, charged up front ──────────────


def test_resident_rag_is_reserved_at_its_task_context(make_model, profiles):
    """A resident serves its task's `max_context`, capped by its design
    context. The number is computable before any chat model is solved.

    Validates: SPEC.md, 'VRAM Allocation': each resident serves the context
    its task declares.
    """
    from llama_packer.endpoints import EMBEDDINGS, RERANK
    from llama_packer.tasks import TaskCatalogue
    from llama_packer.writer import resident_ctx
    tasks = TaskCatalogue.from_cfg({
        "embeddings": {"endpoint": EMBEDDINGS, "max_context": 4096},
        "rerank": {"endpoint": RERANK, "max_context": 6144}})
    emb = make_model("e", role="embeddings", context_length=131072,
                     tasks=["embeddings"])
    rnk = make_model("r", role="rerank", context_length=8192,
                     tasks=["rerank"])
    assert resident_ctx(emb, tasks) == 4096
    assert resident_ctx(rnk, tasks) == 6144
    tight = make_model("t", role="rerank", context_length=2048,
                       tasks=["rerank"])
    assert resident_ctx(tight, tasks) == 2048


def test_resident_parallelism_is_ignored(make_model, profiles, caplog):
    """A resident's declared `parallel:` is ignored with a note.

    Resident slots would come straight out of the chat context, which is the
    opposite of what a resident is for.

    Validates: SPEC.md, 'VRAM Allocation': residents always serve
    single-slot.
    """
    import logging
    from llama_packer.writer import Planner
    emb = make_model("e", role="embeddings", context_length=131072)
    emb.frontmatter["parallel"] = 8
    planner = Planner([emb], profiles, fit_bin="unused", vram_total=48 * 1024)
    planner._bounded_ctx = lambda *a, **k: 4096
    planner._auto_parallel = lambda *a, **k: (1, 4096)
    with caplog.at_level(logging.INFO, logger="llama_packer.writer"):
        variants = planner.plan()["e"]
    assert [v.parallel for v in variants] == [1], "resident took 8 slots"


# ── min_image_megapixels: resolution budget keeps vision ─────────────────

def _mp_profiles():
    return Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                     "tasks": tasks_section(
                         {"coding": {"min_context": 4096,
                                     "sacrifice": ["vision"]}})})


def test_min_image_megapixels_keeps_vision(make_model, tmp_path):
    """A resolution budget must be served, so vision cannot be shed.

    The category sacrifices vision; the model's declared budget overrides
    that, because a budget that is never served is not a budget.

    Validates: docs/models/vision-context.md.
    """
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    m = make_model("v", tasks=["coding"],
                   mmproj={"file": "v-mmproj.gguf", "capabilities": ["image"]},
                   min_image_megapixels=2)
    demands = Planner([m], _mp_profiles(), fit_bin="unused",
                      vram_total=64 * 1024).demands_for(m)
    assert demands and all("vision" in d.keep for d in demands)
    assert all("vision" not in d.require for d in demands)


def test_no_budget_leaves_vision_shed_able(make_model, tmp_path):
    """Without a budget the category's sacrifice order stands.

    Validates: docs/models/vision-context.md.
    """
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    m = make_model("v", tasks=["coding"],
                   mmproj={"file": "v-mmproj.gguf", "capabilities": ["image"]})
    demands = Planner([m], _mp_profiles(), fit_bin="unused",
                      vram_total=64 * 1024).demands_for(m)
    assert demands and all("vision" not in d.keep for d in demands)


def test_zero_budget_drops_the_vision_requirement(make_model, tmp_path):
    """min_image_megapixels: 0 means vision need not be served at all.

    Validates: docs/models/vision-context.md.
    """
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    m = make_model("v", tasks=["coding"],
                   mmproj={"file": "v-mmproj.gguf", "capabilities": ["image"]},
                   min_image_megapixels=0)
    demands = Planner([m], _mp_profiles(), fit_bin="unused",
                      vram_total=64 * 1024).demands_for(m)
    assert demands and all("vision" not in d.keep for d in demands)


# ── host budget (`device: cpu`) ───────────────────────────────────────────


def test_cpu_resident_charges_host_not_vram(profiles, monkeypatch):
    # Validates: docs/plans/requirement-trace.md, requirement #74
    # the host ledger (VRAM charge stays 0), and two committed quads for one
    # role are a planner bug.
    from llama_packer import writer
    from llama_packer.writer import HostLedger

    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", device="cpu", context_length=32768)
    _fake_vram(chat, (1000, 0.5, 100))
    _fake_vram(embed, (466, 0.057, 346))

    monkeypatch.setattr(writer, "solve_matrix_ctx", lambda **kw: 8192)
    planner = Planner([chat, embed], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=embed,
                      host_pool_mb=4096)
    planner.plan()
    # Charge is the measured CPU quad at the resolved resident context.
    # Both slots embed+rerank are filled by the same model: two server
    # processes, two host charges ("one per role", not one per model).
    ctx = planner.matrix_result.embed_ctx
    per_slot = 466 + 346 + int(0.057 * ctx)   # residents serve single-slot
    assert planner.host_ledger.resident_mb() == 2 * per_slot
    assert planner.host_ledger.available() == 4096 - 2 * per_slot
    # The VRAM ledger carries nothing for it (decision 5: no double count).
    assert planner.ledger.residents == []

    # One co-resident per role: committing two quads for one role is a
    # planner bug, not a pool the packer would overrun.
    ledger = HostLedger(4096)
    ledger.add_resident("embed:e", "embeddings", 100)
    with pytest.raises(AssertionError):
        ledger.add_resident("embed:e2", "embeddings", 100)


def test_cpu_variant_refused_on_small_pool(profiles, caplog):
    # Validates: docs/plans/requirement-trace.md, requirement #70, #75
    # quad exceeds the host pool emits no id and logs the shortfall.
    import logging
    m = make("cpu", role="chat", device="cpu")
    _fake_vram(m, (466, 0.057, 346))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                      host_pool_mb=1000)
    with caplog.at_level(logging.INFO):
        plan = planner.plan()
    assert "cpu" not in plan
    assert any("host pool" in r.message for r in caplog.records)


def test_cpu_parallel_charge_flips_refusal(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #75
    # multiplies D and the KV pool and can flip the refusal.
    m = make("cpu", role="chat", device="cpu", parallel=2)
    _fake_vram(m, (466, 0.057, 346))
    # parallel 2 charges two KV pools and two D terms: p=2 charge ≈ 4572 MiB
    refused = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                      host_pool_mb=1800).plan()
    assert "cpu" not in refused
    emitted = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                      host_pool_mb=8192).plan()
    assert [v.parallel for v in emitted["cpu"]] == [2]


def test_no_host_pool_never_gates(profiles, caplog):
    # Validates: docs/plans/requirement-trace.md, requirement #76
    # pool the planner never refuses on host grounds.
    import logging
    m = make("cpu", role="chat", device="cpu")
    _fake_vram(m, (466, 0.057, 346))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                      host_pool_mb=None)
    with caplog.at_level(logging.INFO):
        plan = planner.plan()
    assert "cpu" in plan


def test_host_resident_sizing_lifts_cpu_gate(profiles, monkeypatch):
    # Validates: docs/plans/requirement-trace.md, requirement #74
    # sizes CPU-resident residents through allow_cpu=True (fit_params_static
    # otherwise returns None for a device: cpu model and the charge reads 0).
    m = make("e", role="embeddings", device="cpu", context_length=8192)
    seen = {}

    def fake_static(fit_bin, cache_type="q8_0", llama_args="",
                    allow_cpu=False):
        seen["allow_cpu"] = allow_cpu
        if allow_cpu:
            return SimpleNamespace(model_mib=100, kv_per_token_mib=0.1,
                                   slot_mib=0.0, compute_mib=50,
                                   source="fit-params")
        return None

    m.vram.fit_params_static = fake_static
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                      host_pool_mb=4096)
    mb = planner._host_resident_mb(m, 4096)
    assert seen["allow_cpu"] is True
    assert mb == 100 + 50 + int(0.1 * 4096)


# ── metadata.packing ──────────────────────────────────────────────────────


def test_metadata_packing_cpu_entry(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #77
    m = make("cpk", role="chat", device="cpu")
    _fake_vram(m, (466, 0.057, 346))
    variants = Planner([m], profiles, fit_bin="unused",
                       vram_total=64 * 1024,
                       host_pool_mb=8192).plan()["cpk"]
    v = variants[0]
    assert v.mem_vram_mib is None and v.mem_ram_mib > 0
    config = emit_config([m], {"cpk": variants}, profiles, TVARS)
    p = only_entry(config)["metadata"]["packing"]
    assert p["device"] == "cpu"
    assert p["ctx_size"] == v.ctx_size == 32768
    assert p["parallel"] == v.parallel
    assert p["ram_mib"] == int(round(v.mem_ram_mib))
    assert p["kv_per_token"] == 59769      # 0.057 MiB expressed in bytes
    assert "vram_mib" not in p


def test_metadata_packing_gpu_entry(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #77
    m = make("gpk", role="chat")
    _fake_vram(m, (466, 0.057, 346))
    variants = Planner([m], profiles, fit_bin="unused",
                       vram_total=64 * 1024).plan()["gpk"]
    v = variants[0]
    assert v.mem_vram_mib is not None
    config = emit_config([m], {"gpk": variants}, profiles, TVARS)
    p = only_entry(config)["metadata"]["packing"]
    assert p["device"] == "gpu"
    assert p["vram_mib"] == int(round(v.mem_vram_mib))
    assert p["ram_mib"] == int(round(v.mem_ram_mib))
    assert p["ctx_size"] == v.ctx_size
    assert p["parallel"] == v.parallel
    assert p["kv_per_token"] == 59769


def test_cpu_chat_entry_inherits_the_global_ttl(profiles):
    # Validates: docs/allocation.md, 'TTL' -- an undeclared serving inherits
    # the top-level ttl, so no per-entry key is emitted.
    m = make("ttlcpu", role="chat", device="cpu")
    _fake_vram(m, (466, 0.057, 346))
    variants = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024,
                       host_pool_mb=8192).plan()["ttlcpu"]
    entry = only_entry(emit_config([m], {"ttlcpu": variants}, profiles, TVARS))
    assert "ttl" not in entry


def test_gpu_entry_has_no_ttl(profiles):
    # Validates: docs/plans/requirement-trace.md, requirement #80
    # the inherited/global TTL (no per-model ttl key emitted).
    m = make("ttlgpu", role="chat")
    _fake_vram(m, (466, 0.057, 346))
    variants = Planner([m], profiles, fit_bin="unused",
                       vram_total=64 * 1024).plan()["ttlgpu"]
    entry = only_entry(emit_config([m], {"ttlgpu": variants}, profiles, TVARS))
    assert "ttl" not in entry


def test_cpu_fixed_category_resident_charged_host(profiles, monkeypatch):
    # Validates: docs/plans/requirement-trace.md, requirement #74
    # category (rnk2/stt/…) on device: cpu is a resident too and charges the
    # host pool, one per category.
    from llama_packer import writer

    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", device="cpu", context_length=32768)
    rnk = make("r", role="rerank", device="cpu", context_length=32768)
    extra = make("z", role="rerank", device="cpu", context_length=32768)
    _fake_vram(chat, (1000, 0.5, 100))
    for m in (embed, rnk, extra):
        _fake_vram(m, (100, 0.01, 50))

    monkeypatch.setattr(writer, "solve_matrix_ctx", lambda **kw: 8192)
    planner = Planner([chat, embed, rnk, extra], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=rnk,
                      fixed_categories=[("rnk2", extra)],
                      host_pool_mb=10000)
    planner.plan()
    roles = [role for _, role, _ in planner.host_ledger.residents]
    assert sorted(roles) == ["embed", "rerank", "rnk2"]


def test_task_max_context_caps_a_non_chat_entry(make_model):
    # Validates: SPEC.md, 'Tasks': max_context is the served ceiling, so a
    # resident serves its task's RAG window, not its design context.
    from llama_packer.endpoints import EMBEDDINGS
    from llama_packer.profiles import Profiles
    profiles = Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
        "tasks": {"embeddings": {"endpoint": EMBEDDINGS,
                                 "max_context": 4096}}})
    emb = make_model("e", role="embeddings", context_length=32768,
                     tasks=["embeddings"])
    planner = Planner([emb], profiles, fit_bin="unused",
                      vram_total=48 * 1024)
    planner._bounded_ctx = lambda *a, **kw: kw.get("context_length") or 32768
    variants = planner.plan()["e"]
    assert [v.ctx_size for v in variants] == [4096]


def test_task_max_context_is_per_entry(make_model):
    # Each task's ceiling applies to its own entry, not to every sibling.
    from llama_packer.endpoints import CHAT
    from llama_packer.profiles import Profiles
    profiles = Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
        "tasks": {"coding": {"endpoint": CHAT, "max_context": 131072},
                  "qa": {"endpoint": CHAT, "max_context": 32768}}})
    m = make_model("m", role="chat", context_length=262144,
                   tasks=["coding", "qa"])
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024, matrix_cfg={"auto_parallel": False})
    planner._bounded_ctx = lambda *a, **kw: kw.get("context_length") or 262144
    plan = planner.plan()["m"]
    by_task = {v.task_demands[0].id: v.ctx_size
               for v in plan if v.task_demands}
    assert by_task == {"coding": 131072, "qa": 32768}


def test_non_chat_entry_gets_its_task_ttl(make_model):
    from llama_packer.writer import _entry_task_ttl
    m = make_model("e", role="embeddings", tasks=["embeddings"])
    assert _entry_task_ttl(m, None, {"embeddings": 0}) == 0
    assert _entry_task_ttl(m, None, {"embeddings": 3600}) == 3600


def test_model_id_in_a_set_is_a_chat_slot():
    from llama_packer.writer import _has_chat_slot
    assert _has_chat_slot("Qwen3-27B & embeddings", chat_roles={"chat"},
                          chat_ids={"qwen3-27b"})
    assert not _has_chat_slot("s2t & embeddings", chat_roles={"chat"},
                              chat_ids={"qwen3-27b"})


def test_unestimated_chat_respects_task_max_context(make_model):
    # The ceiling holds for an unestimated serving too.
    from llama_packer.endpoints import CHAT
    from llama_packer.profiles import Profiles
    profiles = Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
        "tasks": {"spot": {"endpoint": CHAT, "max_context": 8192}}})
    m = make_model("u", role="chat", context_length=32768, tasks=["spot"])
    m.vram.unestimated_reason = "no estimate"
    planner = Planner([m], profiles, fit_bin="unused",
                      vram_total=48 * 1024)
    variants = planner.plan()["u"]
    assert [v.ctx_size for v in variants] == [8192]
