# tests/test_auto_parallel.py
"""Auto-parallel: value-function (ctx, slots) solve, floor cascade, knobs."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    def fake(vram_total_mb, *, fit_bin=None, parallel=1, spare_mb=0,
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
    assert (parallel_value(131072, 131072, 2, 0.75)
            > parallel_value(262144, 131072, 1, 0.75))


def test_value_bump_happens_on_less_than_half_cut():
    # 1x200k vs 2x128k: the 1->2 bump happens even though ctx is cut by
    # less than half.
    assert (parallel_value(131072, 131072, 2, 0.75)
            > parallel_value(204800, 131072, 1, 0.75))


def test_value_big_stretch_beats_slots():
    # 1x512k vs 2x128k: the stretch wins — the math simply decides.
    assert (parallel_value(524288, 131072, 1, 0.75)
            > parallel_value(131072, 131072, 2, 0.75))


# ── floor cascade ─────────────────────────────────────────────────────────

def test_floor_explicit_min_context_wins():
    v = _view(min_context=65536, context_length=262144)
    assert resolve_min_ctx(v, pin_ctx=262144) == 65536


def test_floor_pin_is_its_own_floor():
    v = _view(context_length=200000)
    assert resolve_min_ctx(v, pin_ctx=200000) == 200000


def test_floor_pin_below_explicit_warns_and_honours_pin(caplog):
    v = _view(min_context=131072)
    assert resolve_min_ctx(v, pin_ctx=65536) == 131072
    assert any("floor unmet" in r.message for r in caplog.records)


def test_floor_tools_get_agentic_floor():
    v = _view(capabilities=["tools"])
    assert resolve_min_ctx(v, tools_min_ctx=131072) == 131072


def test_floor_non_tools_get_half_max():
    v = _view()
    assert resolve_min_ctx(v) == 131072  # 262144 // 2


def test_floor_explicit_cli_flag_overrides_defaults():
    v = _view(capabilities=["tools"])
    assert resolve_min_ctx(v, fallback_min_ctx=65536,
                           fallback_explicit=True) == 65536
    v2 = _view()
    assert resolve_min_ctx(v2, fallback_min_ctx=65536,
                           fallback_explicit=True) == 65536


def test_floor_explicit_cli_flag_never_beats_sidecar_key():
    v = _view(min_context=200000, capabilities=["tools"])
    assert resolve_min_ctx(v, fallback_min_ctx=65536,
                           fallback_explicit=True) == 200000


def test_floor_garbage_min_context_ignored(caplog):
    v = _view(min_context="lots", capabilities=["tools"])
    assert resolve_min_ctx(v, tools_min_ctx=65536) == 65536
    assert any("not a positive integer" in r.message for r in caplog.records)


# ── knobs parsing ─────────────────────────────────────────────────────────

def test_knobs_auto_parallel_defaults_on():
    k = MatrixKnobs.from_cfg({})
    assert k.auto_parallel is True
    assert k.auto_parallel_max == 8
    assert k.parallel_power == 0.75


def test_knobs_auto_parallel_disable():
    assert MatrixKnobs.from_cfg({"auto_parallel": False}).auto_parallel is False
    assert MatrixKnobs.from_cfg({"auto_parallel": "off"}).auto_parallel is False


def test_knobs_auto_parallel_parsing():
    assert MatrixKnobs.from_cfg({"auto_parallel": True}).auto_parallel is True
    assert MatrixKnobs.from_cfg({"auto_parallel": "yes"}).auto_parallel is True
    assert MatrixKnobs.from_cfg({"auto_parallel": "off"}).auto_parallel is False
    assert MatrixKnobs.from_cfg({"auto_parallel_max": 4}).auto_parallel_max == 4
    assert MatrixKnobs.from_cfg({"parallel_power": 1.0}).parallel_power == 1.0


def test_knobs_auto_parallel_garbage_warns_and_defaults(caplog):
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
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]  # unpinned: no serving pin
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384, 4: 8192})
    variants = _planner(m).plan()["ap"]
    assert len(variants) == 1
    assert variants[0].parallel == 2
    assert variants[0].ctx_size == 32768


def test_plan_stretch_wins_when_second_slot_too_thin(make_model):
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
    m = make_model("ap", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 16384})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 2
    assert variants[0].ctx_size == 32768


def test_plan_pinned_ctx_unaffordable_at_two_stays_one(make_model):
    m = make_model("ap", backend="llama-server", role="chat")
    _scripted_by_parallel(m, {1: 32768, 2: 16384})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_floor_unreachable_falls_back(make_model):
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=999999)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1
    assert variants[0].ctx_size == 32768


def test_plan_sidecar_parallel_pin_skips_solve(make_model):
    m = make_model("ap", backend="llama-server", role="chat", parallel=2)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768, 3: 32768})
    variants = _planner(m).plan()["ap"]
    # Group parallel (the pin) is kept; no solve runs.
    assert variants[0].parallel == 2


def test_plan_profile_parallel_pin_skips_solve(make_model):
    m = make_model("ap", backend="llama-server", role="chat")
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "profiles": {"default": {"parallel": 3}}})
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      matrix_cfg={"auto_parallel": True})
    variants = planner.plan()["ap"]
    assert variants[0].parallel == 3


def test_plan_auto_parallel_off_keeps_today(make_model):
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
    m = make_model("ap", role="embeddings")
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 1


def test_emit_parallel_in_cmd_and_metadata(make_model):
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    profiles = _profiles_no_pin()
    variants = _planner(m).plan()["ap"]
    config = emit_config([m], {"ap": variants}, profiles, TVARS)
    entry = config["models"]["ap"]
    assert "--parallel 2" in entry["cmd"]
    assert "parallel" not in entry["metadata"]  # cmd-level, not metadata
    assert entry["metadata"]["ctx_size"] == 32768


def test_emit_vllm_max_num_seqs(make_model):
    m = make_model("ap", backend="vllm", hf_repo="org/model", role="chat",
                   min_context=8192)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 32768, 2: 32768})
    profiles = _profiles_no_pin()
    variants = _planner(m).plan()["ap"]
    assert variants[0].parallel == 2
    config = emit_config([m], {"ap": variants}, profiles, TVARS)
    assert "--max-num-seqs 2" in config["models"]["ap"]["cmd"]


def test_agentic_floor_default_unchanged():
    assert _MIN_AGENTIC_CTX == 131072


# ── pools config + ledger ───────────────────────────────────────────────

def test_pools_cfg_parsing():
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
    p = Profiles({"defaults": {}, "profiles": {},
                  "pools": {"gpu0": "huge", "gpu1": {"pins": ["x"]}}})
    assert p.pools_cfg == {"gpu1": {"vram": None, "spare": None,
                                   "reserve_extra": None, "pins": {}}}
    assert Profiles({"pools": ["x"]}).pools_cfg == {}


def test_buckets():
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
    from llama_packer.writer import PoolLedger
    pools = {"default": {"vram": None, "spare": None,
                         "reserve_extra": "8G",
                         "pins": {"image": "4G", "embeddings": "auto"}}}
    ledger = PoolLedger(48 * 1024, pools)
    ledger.add_resident("rerank:r", 1024)
    # 8192 (reserve_extra) + 4096 (explicit pin; auto = 0) + 1024 resident
    assert ledger.extra_reserve_mb() == 8192 + 4096 + 1024


def test_ledger_spare_override_and_device_pools():
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
    m = make_model("ap", backend="llama-server", role="chat",
                   min_context=4096)
    del m.frontmatter["context_length"]
    _scripted_by_parallel(m, {1: 40000, 2: 0})
    v = _planner_with_pools(m, {"default": {"spare": "4G"}}).plan()["ap"][0]
    assert v.ctx_size == 40000 - 4096 * 2
