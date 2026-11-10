"""The role registry: operator data, not hardcoded names."""

from __future__ import annotations

from llama_packer import roles
from llama_packer.backends import BACKENDS


def test_share_is_a_percentage_of_slack():
    # Validates: docs/plans/requirement-trace.md, requirement #110
    table, problems = roles.parse_roles({"roles": {
        "chat": {"fill": True},
        "dog": {"share": "40%"},
        "cat": {"share": 0.2},
    }})
    assert problems == []
    assert table["dog"].share == 0.4
    assert table["cat"].share == 0.2
    assert table["chat"].fill is True


def test_share_and_fill_are_validated():
    # Validates: docs/plans/requirement-trace.md, requirement #110
    _table, problems = roles.parse_roles({"roles": {
        "a": {"share": "80%"},
        "b": {"share": "40%"},
        "c": {"fill": True},
        "d": {"fill": True},
        "e": {"share": "120%"},
        "f": {"share": "10%", "fill": True},
    }})
    text = "\n".join(problems)
    assert "shares sum to" in text
    assert "all fill" in text
    assert "is not a percentage" in text
    assert "mutually exclusive" in text


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
            "hamster": {"live": "per-device", "device": "cpu"},
        }})
        assert roles.names() == ("cat", "dog", "hamster")
        assert roles.get("hamster").live == "per-device"
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
    # A percent with a suffix, a bare percent, a fraction, and 1 = 100%.
    def share(value):
        table, problems = roles.parse_roles(
            {"roles": {"a": {"share": value}}})
        assert problems == []
        return table["a"].share

    assert [share("40%"), share(40), share(0.4), share(1)] == [
        0.4, 0.4, 0.4, 1.0]
