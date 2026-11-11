"""The role registry: operator data, not hardcoded names."""

from __future__ import annotations

from llama_packer import roles
from llama_packer.backends import BACKENDS
from llama_packer.roles import Share


def test_share_is_a_request_weight():
    # Validates: docs/allocation.md, 'The auction': a role's share is its
    # request weight in each set, not a fraction of the fleet.
    table, problems = roles.parse_roles({"roles": {
        "chat": {"share": "fill"},
        "dog": {"share": "40%"},
        "cat": {"share": 0.2},
    }})
    assert problems == []
    assert table["dog"].share == Share("weight", 40.0)
    assert table["cat"].share == Share("weight", 0.2)
    assert table["chat"].share == Share("fill", 100.0)


def test_share_values_are_uncapped():
    # Validates: docs/allocation.md, 'The auction': weights are relative and
    # may exceed 100; they are normalised per set.
    table, problems = roles.parse_roles({"roles": {
        "a": {"share": "80%"},
        "b": {"share": "40%"},
        "c": {"share": "123"},
        "d": {"share": "min"},
        "e": {"share": 0},
    }})
    assert problems == []
    assert table["c"].share == Share("weight", 123.0)
    assert table["d"].share == Share("min", 0.0)
    assert table["e"].share == Share("zero", 0.0)


def test_bad_share_is_reported():
    _table, problems = roles.parse_roles({"roles": {
        "a": {"share": "banana"},
        "b": {"share": -1},
    }})
    assert sum("is not a weight" in p for p in problems) == 2


def test_role_registry_derives_its_sets():
    # Validates: docs/plans/requirement-trace.md, requirement #90
    assert roles.names() == ("chat", "embeddings", "rerank",
                             "image", "s2t", "t2s")
    assert all(r.device is None for r in roles.ROLES.values())
    assert roles.get("embedder") is None


def test_custom_role_is_data_and_gets_its_own_pool():
    # Validates: docs/plans/requirement-trace.md, requirement #107
    try:
        roles.configure({"roles": {
            "cat": {},
            "dog": {"device": 1},
            "hamster": {"device": "cpu"},
        }})
        assert roles.names() == ("cat", "dog", "hamster")
        assert roles.get("dog").device == 1
        assert roles.get("hamster").device == "cpu"
    finally:
        roles.configure(None)


def test_dir_is_one_path_or_a_list_and_is_normalised():
    # Validates: SPEC.md, Roles: dir is the tree a role owns.
    table, problems = roles.parse_roles({"roles": {
        "chat": {"dir": ["./chat/", "T2T"]},
        "pics": {"dir": "img"},
    }})
    assert problems == []
    assert table["chat"].dir == ("chat", "t2t")
    assert table["pics"].dir == ("img",)


def test_overlapping_role_trees_are_rejected():
    # Validates: SPEC.md, Roles: a tree belongs to exactly one role.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "chat"},
        "b": {"dir": "chat/qwen"},
    }})
    assert any("both own" in p for p in problems)
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "."},
        "b": {"dir": "chat"},
    }})
    assert any("both own" in p for p in problems)


def test_role_dir_rejects_paths_outside_the_models_root():
    # Validates: SPEC.md, Roles: dir is relative to the models root.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "/srv/models"},
        "b": {"dir": "../escape"},
    }})
    assert sum("must be relative" in p for p in problems) == 2


def test_custom_role_parse_reports_every_problem():
    # Validates: docs/plans/requirement-trace.md, requirement #107
    _roles, problems = roles.parse_roles({"roles": {
        "cat": {"kind": "unicorn", "weight": 0},
        "dog": {"unknown": 1},
    }})
    text = "\n".join(problems)
    assert "'kind' is retired" in text
    assert "'weight' is retired" in text
    assert "unknown key 'unknown'" in text


def test_every_engine_endpoint_is_closed_and_served():
    # Validates: docs/plans/requirement-trace.md, requirement #90
    from llama_packer import endpoints

    served = {e for b in BACKENDS.values() for e in b.endpoints}
    assert served <= endpoints.ENDPOINTS
    assert all(b.endpoints for b in BACKENDS.values())


def test_ctx_cap_is_retired():
    # Validates: SPEC.md, 'Roles': the context cap is a task max_context.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "x", "ctx_cap": 8192}}})
    assert any("'ctx_cap' is retired" in p for p in problems)


def test_persistent_is_retired():
    # Validates: SPEC.md, Roles: there is no persistent; residency is ttl: 0.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "x", "persistent": True}}})
    assert any("'persistent' is retired" in p for p in problems)


def test_role_device_uses_the_sidecar_vocabulary():
    # Validates: SPEC.md, Roles: device is cpu or a GPU index, exactly like a
    # sidecar; there is no gpu/any spelling.
    table, problems = roles.parse_roles({"roles": {
        "a": {"dir": "x", "device": "cpu"},
        "b": {"dir": "y", "device": 1},
        "c": {"dir": "z", "device": "1"},
        "d": {"dir": "w", "device": "gpu"},
        "e": {"dir": "v", "device": "any"},
    }})
    assert table["a"].device == "cpu"
    assert table["b"].device == 1
    assert table["c"].device == 1
    assert table["d"].device is None and table["e"].device is None
    assert sum("neither 'cpu' nor a GPU index" in p for p in problems) == 2


def test_reserved_pool_names_are_rejected():
    # A role pool is named after the role; ram/default/gpuN are pool ids.
    _table, problems = roles.parse_roles({"roles": {
        "ram": {"dir": "a"}, "default": {"dir": "b"}, "gpu0": {"dir": "c"}}})
    assert sum("reserved pool name" in p for p in problems) == 3


def test_one_role_may_list_overlapping_trees():
    # A role listing x and x/y is redundant, not two roles owning a tree.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"dir": ["x", "x/y"]}}})
    assert problems == []


def test_share_forms():
    # Validates: docs/allocation.md, 'The auction': a number is a weight,
    # `fill` is 100, `min` is 0 with no accept, and `0` accepts leftovers.
    def share(value):
        table, problems = roles.parse_roles(
            {"roles": {"a": {"share": value}}})
        assert problems == []
        return table["a"].share

    assert share("40%") == Share("weight", 40.0)
    assert share(40) == Share("weight", 40.0)
    assert share(0.4) == Share("weight", 0.4)
    assert share("fill") == Share("fill", 100.0)
    assert share("min") == Share("min", 0.0)
    assert share(0) == Share("zero", 0.0)


def test_fill_and_live_are_not_keys():
    # Validates: docs/roles-tasks-routing.md, Role: `share` carries the
    # request; there is no fill or per-role count key.
    _table, problems = roles.parse_roles({"roles": {
        "a": {"fill": True},
        "b": {"live": 1},
    }})
    text = "\n".join(problems)
    assert "unknown key 'fill'" in text
    assert "unknown key 'live'" in text
