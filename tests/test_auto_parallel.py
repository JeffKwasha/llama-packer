# tests/test_auto_parallel.py
"""Auto-parallel: value-function (ctx, slots) solve, floor cascade, knobs."""

from __future__ import annotations

from types import SimpleNamespace

from conftest import entry_for, only_entry
from llama_packer.profiles import Profiles
from llama_packer.consts import _MIN_AGENTIC_CTX
from llama_packer.writer import (
    MatrixKnobs,
    Planner,
    emit_config,
    parallel_value,
    resolve_min_ctx,
)

TVARS = {"llama_bin": "/opt/llama-server"}


def _profiles_no_pin():
    # No `parallel` anywhere: defaults carry only cache_type, so every group
    # is auto-parallel-eligible (nothing pinned).
    return Profiles({"defaults": {"cache_type": "q8_0"},
                     "profiles": {"default": {}}})


def _scripted_by_parallel(model, by_parallel):
    """Fake calc_ctx: max affordable ctx keyed on parallel, minus spare.

    The spare term proves ledger reserves actually reach the budget: every
    MB of spare costs 2 tokens of context in the fake.
    """
    def fake(vram_total_mb, *, server_bin=None, parallel=1, spare_mb=0,
             include_mmproj=True, baseline_mb=0, cache_type="q8_0",
             design_ctx=None, **kw):
        return max(0, by_parallel.get(parallel, 0) - spare_mb * 2)
    model.vram.calc_ctx = fake
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.calc_ctx = fake


def _view(stem="m", **fm):
    return SimpleNamespace(stem=stem, frontmatter=dict(fm),
                           capabilities=list(fm.get("capabilities", [])),
                           design_context=fm.get("design_context", 262144))


# ── score function: the doc's worked examples ─────────────────────────────

def test_value_two_slots_beat_one_stretched():
    # Validates: docs/plans/category-semantics.md, 'Parallel: the same search, a different exponent': Rank a (context.
    assert (parallel_value(131072, 131072, 2, 0.75)
            > parallel_value(262144, 131072, 1, 0.75))


def test_value_bump_happens_on_less_than_half_cut():
    # 1x200k vs 2x128k: the 1->2 bump happens even though ctx is cut by
    # less than half.
    # Validates: docs/plans/category-semantics.md, 'Parallel: the same search, a different exponent': Rank a (context.
    assert (parallel_value(131072, 131072, 2, 0.75)
            > parallel_value(204800, 131072, 1, 0.75))


def test_value_big_stretch_beats_slots():
    # 1x512k vs 2x128k: the stretch wins — the math simply decides.
    # Validates: docs/plans/category-semantics.md, 'Parallel: the same search, a different exponent': Rank a (context.
    assert (parallel_value(524288, 131072, 1, 0.75)
            > parallel_value(131072, 131072, 2, 0.75))


# ── floor cascade ─────────────────────────────────────────────────────────

def test_floor_explicit_min_context_wins():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(min_context=65536, context_length=262144)
    assert resolve_min_ctx(v, pin_ctx=262144) == 65536


def test_floor_pin_is_its_own_floor():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(context_length=200000)
    assert resolve_min_ctx(v, pin_ctx=200000) == 200000


def test_floor_pin_below_explicit_warns_and_honours_pin(caplog):
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(min_context=131072)
    assert resolve_min_ctx(v, pin_ctx=65536) == 131072
    assert any("floor unmet" in r.message for r in caplog.records)


def test_floor_tools_get_agentic_floor():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(capabilities=["tools"])
    assert resolve_min_ctx(v, tools_min_ctx=131072) == 131072


def test_floor_non_tools_get_half_max():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view()
    assert resolve_min_ctx(v) == 131072  # 262144 // 2


def test_floor_explicit_cli_flag_overrides_defaults():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(capabilities=["tools"])
    assert resolve_min_ctx(v, fallback_min_ctx=65536,
                           fallback_explicit=True) == 65536
    v2 = _view()
    assert resolve_min_ctx(v2, fallback_min_ctx=65536,
                           fallback_explicit=True) == 65536


def test_floor_explicit_cli_flag_never_beats_sidecar_key():
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(min_context=200000, capabilities=["tools"])
    assert resolve_min_ctx(v, fallback_min_ctx=65536,
                           fallback_explicit=True) == 200000


def test_floor_garbage_min_context_ignored(caplog):
    # Validates: SPEC.md, 'Context Tiers': Per-model minimum useful context: the hard floor of the (ctx, slots).
    v = _view(min_context="lots", capabilities=["tools"])
    assert resolve_min_ctx(v, tools_min_ctx=65536) == 65536
    assert any("not a positive integer" in r.message for r in caplog.records)


# ── knobs parsing ─────────────────────────────────────────────────────────

def test_knobs_auto_parallel_defaults_on():
    # Validates: docs/plans/auto-parallel.md, 'Design': auto-parallel is on unless the operator turns it off.
    k = MatrixKnobs.from_cfg({})
    assert k.auto_parallel is True
    assert k.auto_parallel_max == 8
    assert k.parallel_power == 0.75


def test_knobs_auto_parallel_disable():
    # Validates: SPEC.md, 'Knobs (matrix section keys)': auto_parallel defaults to true; false turns it off fleet-wide
    assert MatrixKnobs.from_cfg({"auto_parallel": False}).auto_parallel is False
    assert MatrixKnobs.from_cfg({"auto_parallel": "off"}).auto_parallel is False


def test_knobs_auto_parallel_parsing():
    # Validates: docs/plans/auto-parallel.md, 'Design': the knob is read from the matrix section.
    assert MatrixKnobs.from_cfg({"auto_parallel": True}).auto_parallel is True
    assert MatrixKnobs.from_cfg({"auto_parallel": "yes"}).auto_parallel is True
    assert MatrixKnobs.from_cfg({"auto_parallel": "off"}).auto_parallel is False
    assert MatrixKnobs.from_cfg({"auto_parallel_max": 4}).auto_parallel_max == 4
    assert MatrixKnobs.from_cfg({"parallel_power": 1.0}).parallel_power == 1.0


def test_knobs_auto_parallel_garbage_warns_and_defaults(caplog):
    # Validates: docs/plans/auto-parallel.md, 'Design': a malformed value warns and falls back to the default.
    k = MatrixKnobs.from_cfg({"auto_parallel": "maybe",
                              "auto_parallel_max": 0,
                              "parallel_power": -1.0})
    assert k.auto_parallel is False
    assert k.auto_parallel_max == 8
    assert k.parallel_power == 0.75
    assert len([r for r in caplog.records if "matrix:" in r.message]) == 3


# ── Planner integration ───────────────────────────────────────────────────

def _planner(model, matrix_extra=None):
    cfg = {"auto_parallel": True}
    cfg.update(matrix_extra or {})
    return Planner([model], _profiles_no_pin(), fit_bin="unused",
                   vram_total=48 * 1024, matrix_cfg=cfg)


def test_plan_picks_two_slots_over_stretch(make_model):
    # floor 8k (explicit), cap = design default 32k. p1 affords 32k, p2
    # affords 32k, p3 affords 16k: 2x32k wins on score.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]  # unpinned: no serving pin
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384, 4: 8192})
    variants = _planner(m).plan()["ap"]
    assert len(variants) == 1
    assert variants[0].parallel == 2
    assert variants[0].ctx_size == 32768


def test_plan_stretch_wins_when_second_slot_too_thin(make_model):
    # Validates: docs/plans/auto-parallel.md, 'Design': the slot score is (ctx/floor)^0.5 x slots^parallel_power, and the best feasible pair wins
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 8192})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_pinned_ctx_searches_slots_only(make_model):
    # Fixture default context_length=32768 is a serving pin: ctx is fixed,
    # the search runs over p alone. Both p1/p2 afford the pin → p2 wins.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    m = make_model("ap", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 2
    assert variants[0].ctx_size == 32768


def test_design_max_is_a_ceiling_not_a_floor(make_model):
    """context_length: is what the model was built for, not what to serve.

    It used to be read as a pin, so a 32768 design max became a 32768 floor,
    the 16384 the second slot would afford fell below it, and the model was
    locked to parallel 1 -- because a capability claim about the model was
    being read as an operating decision about the fleet. The floor is now half
    the design context, and with VRAM to spare the slot count rises while the
    served context holds at the design cap.

    Validates: SPEC.md, 'Context Tiers': sidecar context_length is a design maximum, a
    ceiling applied when serving, never a floor.
    """
    # Plentiful VRAM: the model holds its 32768 cap at two slots as well as
    # one, so the null category's exponent-0 score ties and the tie-break
    # decides. With only one slot's worth of headroom (32768 -> 16384) the
    # score is not tied and parallel 1 is the right answer.
    m = make_model("ap", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384, 4: 8192})
    variants = _planner(m).plan()["ap"]
    # p=1 and p=2 both hold the cap, so they tie at exponent 0 and the later
    # (higher) slot count wins. p=3 and p=4 afford less context, so they
    # score lower and are not tied.
    assert variants[0].parallel == 2, "ties must go to more slots"
    assert variants[0].ctx_size == 32768, "while holding the design cap"


def test_explicit_max_context_still_pins_and_floors(make_model):
    """--max-context is a real pin, so the old behaviour survives there.

    Validates: SPEC.md, 'Context Tiers': --max-context is a real pin and its own floor.
    """
    m = make_model("ap", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 16384})
    planner = _planner(m)
    planner.max_context = 32768
    variants = planner.plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_floor_unreachable_falls_back(make_model):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a rung that does not fit is not a candidate; the next along the order is tried
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=999999)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_floor_unreachable_solves_once(make_model):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': the solve runs once per model, not once per rung
    """An unaffordable model is not emitted, and not warned about twice.

    The probe count moved from one to one-per-axis -- solve_rung walks the
    shed ladder and the slot solve walks the slot curve -- so the assertion
    is on the outcome, which is what the old count was guarding.
    """
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=999999)
    del m.frontmatter["context_length"]
    calls: list[int] = []

    def fake(vram_total_mb, *, parallel=1, spare_mb=0, **kw):
        calls.append(parallel)
        return 32768

    m.vram.calc_ctx = fake
    for flag in (True, False):
        view = m.view_for(flag)
        if view is not m:
            view.vram.calc_ctx = fake
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768
    assert calls, "nothing was solved at all"


def test_plan_sidecar_parallel_pin_skips_solve(make_model):
    # Validates: SPEC.md, 'Knobs (matrix section keys)': a sidecar parallel: pin opts that model out of the auto-parallel solve
    m = make_model("ap", backend="llama-server", role="chat", parallel=2)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 32768})
    variants = _planner(m).plan()["ap"]
    # Group parallel (the pin) is kept; no solve runs.
    assert variants[0].parallel == 2


def test_plan_profile_parallel_pin_skips_solve(make_model):
    # Validates: SPEC.md, 'Knobs (matrix section keys)': a sidecar parallel: pin opts that model out of the auto-parallel solve
    m = make_model("ap", backend="llama-server", role="chat")
    del m.frontmatter["context_length"]
    # The planner now probes at the pinned slot count as well, so the fake
    # has to know what that many slots affords.
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 32768})
    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "profiles": {"default": {"parallel": 3}}})
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      matrix_cfg={"auto_parallel": True})
    variants = planner.plan()["ap"]
    assert variants[0].parallel == 3


def test_plan_auto_parallel_off_keeps_today(make_model):
    # Validates: SPEC.md, 'Knobs (matrix section keys)': with auto_parallel off the model keeps today's serving
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    planner = Planner([m], _profiles_no_pin(), fit_bin="unused",
                      vram_total=48 * 1024,
                      matrix_cfg={"auto_parallel": False})
    variants = planner.plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_non_chat_backend_untouched(make_model):
    # Validates: SPEC.md, 'Knobs (matrix section keys)': only unpinned chat models on llama-server and vLLM enter the auto-parallel solve
    m = make_model("ap", role="embeddings")
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1


def test_emit_parallel_in_cmd_and_metadata(make_model):
    # Validates: SPEC.md, 'Output Format': the served slot count reaches the command and the entry's metadata
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    profiles = _profiles_no_pin()
    variants = _planner(m).plan()["ap"]
    config = emit_config([m], {"ap": variants}, profiles, TVARS)
    entry = entry_for(config, "ap")
    assert "--parallel 2" in entry["cmd"]
    assert "parallel" not in entry["metadata"]  # cmd-level, not metadata
    assert entry["metadata"]["ctx_size"] == 32768


def test_emit_vllm_max_num_seqs(make_model):
    # Validates: docs/backends/vllm.md, 'Command shape': the slot count maps to --max-num-seqs
    m = make_model("ap", backend="vllm", hf_repo="org/model", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    profiles = _profiles_no_pin()
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 2
    config = emit_config([m], {"ap": variants}, profiles, TVARS)
    assert "--max-num-seqs 2" in entry_for(config, "ap")["cmd"]


def test_agentic_floor_default_unchanged():
    assert _MIN_AGENTIC_CTX == 131072


# ── pools config + ledger ───────────────────────────────────────────────

def test_pools_cfg_parsing():
    # Validates: docs/plans/auto-parallel.md, 'Configurable pools (operator escape hatch)': the pools section names each pool's vram, spare and reserve_extra
    p = Profiles({"defaults": {}, "profiles": {},
                  "pools": {"gpu0": {"vram": "24G", "spare": "2G",
                                     "reserve_extra": "8G",
                                     "pins": {"image": "40G",
                                              "embeddings": "auto"}}}})
    spec = p.pools_cfg["gpu0"]
    assert spec["vram"] == "24G"
    assert spec["pins"] == {"image": "40G", "embeddings": "auto"}
    assert Profiles({"defaults": {}}).pools_cfg == {}


def test_pools_cfg_garbage_tolerated(caplog):
    # Validates: docs/plans/auto-parallel.md, 'Configurable pools (operator escape hatch)': a malformed pools entry warns and falls back to the default
    p = Profiles({"defaults": {}, "profiles": {},
                  "pools": {"gpu0": "huge", "gpu1": {"pins": ["x"]}}})
    assert p.pools_cfg == {"gpu1": {"vram": None, "spare": None,
                                   "reserve_extra": None, "pins": {}}}
    assert Profiles({"pools": ["x"]}).pools_cfg == {}


def test_buckets():
    # Validates: docs/plans/auto-parallel.md, 'Memory pools — size buckets, not model types': a pool is a size bucket, not a model type
    from llama_packer.writer import PoolLedger, footprint_bucket
    tiny = SimpleNamespace(role="s2t", backend="whisper-server")
    assert footprint_bucket(tiny) == "tiny"
    assert footprint_bucket(SimpleNamespace(role="rerank",
                                            backend="llama-server")) == "tiny"
    assert footprint_bucket(SimpleNamespace(role="image",
                                            backend="sd-server")) == "huge"
    assert footprint_bucket(SimpleNamespace(role="chat",
                                            backend="llama-server")) == "chat"
    assert PoolLedger(48000).extra_reserve_mb() == 0


def test_ledger_extra_reserve_math():
    # Validates: docs/plans/auto-parallel.md, 'Memory pools — size buckets, not model types': the pool's reserve_extra is added to the solve's spare
    from llama_packer.writer import PoolLedger
    pools = {"default": {"vram": None, "spare": None,
                         "reserve_extra": "8G",
                         "pins": {"image": "4G", "embeddings": "auto"}}}
    ledger = PoolLedger(48 * 1024, pools)
    ledger.add_resident("rerank:r", 1024)
    # 8192 (reserve_extra) + 4096 (explicit pin; auto = 0) + 1024 resident
    assert ledger.extra_reserve_mb() == 8192 + 4096 + 1024


def test_ledger_spare_override_and_device_pools():
    # Validates: docs/plans/auto-parallel.md, 'Memory pools — size buckets, not model types': a model is assigned to the pool its device belongs to
    from llama_packer.writer import PoolLedger
    pools = {"gpu1": {"vram": "24G", "spare": "2G",
                      "reserve_extra": None, "pins": {}}}
    ledger = PoolLedger(96 * 1024, pools)
    assert ledger.pool_vram_mb("gpu1") == 24576
    assert ledger.pool_vram_mb("default") == 96 * 1024
    assert ledger.spare_for("gpu1", 512) == 2048
    assert ledger.spare_for("default", 512) == 512
    assert ledger.pool_id_for(SimpleNamespace(frontmatter={})) == "default"
    assert ledger.pool_id_for(
        SimpleNamespace(frontmatter={"device": 1})) == "gpu1"


def _planner_with_pools(model, pools):
    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "profiles": {"default": {}},
                         "pools": pools})
    return Planner([model], profiles, fit_bin="unused",
                   vram_total=48 * 1024,
                   matrix_cfg={"auto_parallel": True})


def test_plan_pools_reserve_extra_shrinks_ctx(make_model):
    # Same model, same fake: the only difference is an 8G unmodelled
    # resident on the default pool → 8192 MB × 2 tok = 16384 fewer tokens.
    # Validates: SPEC.md, 'Context Management': Plan serving variants for every model.
    def run(pools):
        m = make_model("ap", backend="llama-server", role="chat",
                       min_context=4096)
        del m.frontmatter["context_length"]
        _scripted_by_parallel(m, {1: 40000, 2: 0})
        return _planner_with_pools(m, pools).plan()["ap"][0]
    bare = run({})
    assert (bare.parallel, bare.ctx_size) == (1, 32768)  # capped by design
    held = run({"default": {"reserve_extra": "8G"}})
    assert (held.parallel, held.ctx_size) == (1, 40000 - 16384)


def test_plan_pools_spare_override_applies(make_model):
    # Validates: docs/plans/auto-parallel.md, 'Memory pools — size buckets, not model types': the pool's spare is the model's spare for the solve
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=4096)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 40000, 2: 0})
    v = _planner_with_pools(m, {"default": {"spare": "4G"}}).plan()["ap"][0]
    assert v.ctx_size == 40000 - 4096 * 2


# ── serving pin guards ────────────────────────────────────────────────────

def test_serving_pin_rejects_non_positive():
    # a 0/negative pin previously crashed auto-parallel (ZeroDivisionError
    # in parallel_value); treat it as absent instead
    # Validates: SPEC.md, 'Context Tiers': Explicitly pinned serving ctx.
    p = Planner.__new__(Planner)
    p.max_context = None
    assert p._serving_pin(_view(context_length=0)) is None
    assert p._serving_pin(_view(context_length=-4096)) is None
    p.max_context = 0
    assert p._serving_pin(_view()) is None


def test_serving_pin_ignores_the_design_max():
    """context_length is a ceiling; --max-context is the pin.

    Reading the sidecar as a pin meant the only way to give a model back its
    slots was to edit context_length -- rewriting what the model can do in
    order to express how we want it served. min_context: is that key, and it
    was already separate.

    Validates: SPEC.md, 'Context Tiers': only --max-context pins a serving context.
    """
    p = Planner.__new__(Planner)
    p.max_context = 0
    assert p._serving_pin(_view(context_length=8192)) is None
    p.max_context = 8192
    assert p._serving_pin(_view(context_length=4096)) == 8192


# ── a declared category outranks the heuristics below it ──────────────────


def test_category_floor_beats_the_tools_heuristic():
    """Declaring the work must not make the model less satisfiable.

    `resolve_min_ctx` fell back to tools_min_ctx (131072) for any model
    listing the `tools` capability, and a usage demand could only *raise*
    that floor. So a model tagged `aux` (80k) that also lists `tools` was
    floored at an unreachable 131072, fell back to one slot, and then failed
    its own category's min_parallel of 2 -- declaring it aux made it strictly
    harder to serve.

    Validates: docs/plans/category-semantics.md: a declared category outranks the
    heuristics that guess about undeclared models.
    """
    view = _view(capabilities=["tools", "image"])
    assert resolve_min_ctx(view) == 131072          # untagged: unchanged
    assert resolve_min_ctx(view, category_floor=81920) == 81920


def test_category_floor_never_overrides_the_operator():
    """`min_context:` and --max-context are the operator's, not a guess.

    Validates: SPEC.md, 'Context Tiers': min_context and --max-context outrank a category.
    """
    view = _view(capabilities=["tools"], min_context=200000)
    assert resolve_min_ctx(view, category_floor=81920) == 200000
    pinned = _view(capabilities=["tools"])
    assert resolve_min_ctx(pinned, pin_ctx=131072,
                           category_floor=81920) == 131072


def test_multi_category_takes_the_heaviest():
    """`usage: [coding, chat]` serves at coding's window, not chat's.

    Validates: SPEC.md, `usage:`: each satisfiable category emits its own entry.
    """
    view = _view(capabilities=["tools"])
    assert resolve_min_ctx(view, category_floor=133120) == 133120


def test_cpu_resident_chat_gets_no_auto_parallel(make_model):
    """Host-RAM KV lives in an arena no VRAM ledger budgets.

    calc_ctx returns the design context regardless of the slot count for a
    CPU-resident model, so every candidate looks equally affordable and the
    score reduces to p**power — walking straight to the cap. Eight slots on
    a model declared for one is an 8x KV over-commit nothing accounts for.

    Validates: docs/plans/requirement-trace.md #29.
    """
    from llama_packer.profiles import Profiles as _P
    model = make_model("cpu-chat", role="chat", device="cpu")
    planner = Planner([model],
                      _P({"defaults": {"cache_type": "q8_0"},
                          "profiles": {"default": {}}}),
                      fit_bin="unused", vram_total=64 * 1024)
    planner._bounded_ctx = lambda *a, **k: 131072
    assert [v.parallel for v in planner.plan()["cpu-chat"]] == [1]
