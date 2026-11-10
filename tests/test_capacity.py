"""Capacity routing: pools, per-entry capacities, and derived eviction cost."""

from __future__ import annotations

from types import SimpleNamespace

from llama_packer.profiles import Profiles
from llama_packer.writer import Planner, _capacity_fields


def _model(**kw):
    from conftest import fake_tasks
    base = dict(on_cpu=False, size_mb=20000, role="chat",
                sidecar_keys=frozenset(), frontmatter={})
    base.update(kw)
    m = SimpleNamespace(**base)
    m.tasks_resolved = fake_tasks({"role": m.role})
    m.serves_endpoint = lambda ep: any(t.endpoint == ep
                                       for t in m.tasks_resolved)
    return m


def _variant(**kw):
    base = dict(mem_vram_mib=27000.0, mem_ram_mib=1300.0,
                ctx_size=200000, parallel=1,
                task_demands=(SimpleNamespace(role="chat"),))
    base.update(kw)
    return SimpleNamespace(**base)


def test_capacity_pools_are_gpu_budget_host_pool_and_role_singletons():
    # Validates: SPEC.md, "Capacity routing": pools are the packer's
    # available GPU budget, the detected host pool, and one size-1 pool per
    # scheduling role.
    p = Planner([], Profiles({}), "fit", 32768, baseline_mb=4096,
                host_pool_mb=50000)
    assert p.capacity_pools["default"] == 32768 - 1024 - 4096
    assert p.capacity_pools["ram"] == 50000
    for role in ("chat", "embeddings", "rerank", "image", "s2t", "t2s"):
        assert p.capacity_pools[role] == 1


def test_pools_ram_spare_shrinks_the_host_pool():
    # Validates: docs/plans/requirement-trace.md, requirement #108
    p = Planner([], Profiles({"pools": {"ram": {"spare": "2G"}}}), "fit",
                32768, baseline_mb=4096, host_pool_mb=50000)
    assert p.capacity_pools["ram"] == 50000 - 2048
    assert p.host_ledger.pool_mb == 50000 - 2048


def test_custom_role_gets_its_own_size_one_pool():
    # Validates: docs/plans/requirement-trace.md, requirement #107
    from llama_packer import roles

    try:
        roles.configure({"roles": {"cat": {"kind": "chat"},
                                   "dog": {"kind": "rerank"}}})
        p = Planner([], Profiles({}), "fit", 32768, baseline_mb=4096,
                    host_pool_mb=50000)
        assert p.capacity_pools["cat"] == 1
        assert p.capacity_pools["dog"] == 1
    finally:
        roles.configure(None)


def test_custom_role_entry_charges_its_own_pool():
    # Validates: docs/plans/requirement-trace.md, requirement #107
    caps = _capacity_fields(
        _model(role="cat"),
        _variant(task_demands=(SimpleNamespace(role="cat"),)),
        {"default": 30000, "ram": 50000, "cat": 1},
        100, 1000, 0.04)
    assert caps["capacities"]["cat"] == 1


def test_capacity_fields_charge_gpu_ram_and_role():
    # Validates: SPEC.md, "Capacity routing": every entry declares the pools
    # it consumes; chat entries also take their role's singleton slot.
    caps = _capacity_fields(_model(), _variant(), {"default": 30000,
                                                   "ram": 50000, "chat": 1},
                            100, 1000, 0.04)
    assert caps["capacities"] == {"default": 28080, "ram": 1352, "chat": 1}
    assert caps["evict_cost"] == 400  # 200 s load + 200 s prefill


def test_capacity_fields_cpu_helper_charges_host_only():
    # Validates: SPEC.md, "Capacity routing": a device: cpu serving consumes
    # the host pool, never a GPU pool.
    caps = _capacity_fields(
        _model(on_cpu=True, size_mb=700, role="embeddings"),
        _variant(mem_vram_mib=None, mem_ram_mib=1200, ctx_size=4096,
                 task_demands=(SimpleNamespace(role="embeddings"),)),
        {"default": 30000, "ram": 50000, "embeddings": 1},
        100, 1000, 0.04)
    assert caps["capacities"] == {"ram": 1248, "embeddings": 1}
    assert caps["evict_cost"] == 7


def test_capacity_evict_cost_ignores_prefill_for_stateless_roles():
    # Validates: SPEC.md, "Capacity routing": only a stateful chat serving
    # pays the lost-prefill term; helpers pay reload time.
    caps = _capacity_fields(
        _model(role="rerank", size_mb=4000),
        _variant(task_demands=(SimpleNamespace(role="rerank"),), ctx_size=4096),
        {"default": 30000, "rerank": 1}, 100, 1000, 0.04)
    assert caps["evict_cost"] == 40


def test_capacity_evict_cost_uses_sidecar_prompt_rate():
    # Validates: SPEC.md, "Capacity routing": a sidecar pp_tps: overrides the
    # fleet default when pricing lost prefill.
    caps = _capacity_fields(
        _model(size_mb=20000, frontmatter={"pp_tps": 3000},
               sidecar_keys=frozenset({"pp_tps"})),
        _variant(ctx_size=200000, parallel=1),
        {"default": 30000, "chat": 1}, 100, 1000, 0.04)
    assert caps["evict_cost"] == 267  # 200 s load + 67 s prefill


def test_capacity_unestimated_serving_reserves_the_whole_pool():
    # Validates: SPEC.md, "Capacity routing": an unmeasured entry is never
    # declared as a guessed floor; it reserves the pool and runs alone.
    caps = _capacity_fields(
        _model(size_mb=20000),
        _variant(mem_vram_mib=None),
        {"default": 30000, "chat": 1}, 100, 1000, 0.04)
    assert caps["capacities"]["default"] == 30000


def test_int_spare_is_mib_not_gigabytes():
    # Regression: a resolved spare passed as an int was re-parsed as GB.
    p = Planner([], Profiles({}), "fit", 24576, spare=64)
    assert p.capacity_pools["default"] == 24576 - 2048 - 64
