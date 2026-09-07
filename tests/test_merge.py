# tests/test_merge.py
"""The single merge rule: dicts merge, lists append, None deletes.

Covers utils.merge_layer, its wiring through ScopeStack/profiles, sidecar
mode layering (the partial-mode inheritance fix), and mmproj companion
blocks with serving views.
"""

from __future__ import annotations

from llama_packer import utils
from llama_packer.writer import _build_mode_params


# ── merge_layer ───────────────────────────────────────────────────────────

def test_merge_scalars_upper_wins():
    assert utils.merge_layer({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}


def test_merge_dicts_recurse():
    below = {"modes": {"coding": {"temperature": 0.5, "top_p": 0.9}}}
    above = {"modes": {"coding": {"temperature": 0.2}}}
    assert utils.merge_layer(below, above) == {
        "modes": {"coding": {"temperature": 0.2, "top_p": 0.9}}}


def test_merge_lists_union_append_dedup():
    assert utils.merge_layer({"capabilities": ["tools"]},
                             {"capabilities": ["image", "tools"]}) == {
        "capabilities": ["tools", "image"]}


def test_merge_list_removal_syntax():
    assert utils.merge_layer({"capabilities": ["tools", "image"]},
                             {"capabilities": ["-image"]}) == {
        "capabilities": ["tools"]}
    # Removing an absent item is a no-op.
    assert utils.merge_layer({"capabilities": ["tools"]},
                             {"capabilities": ["-image"]}) == {
        "capabilities": ["tools"]}


def test_merge_none_deletes():
    assert utils.merge_layer({"top_p": 0.9, "temperature": 0.5},
                             {"top_p": None}) == {"temperature": 0.5}
    # Deleting a missing key is a no-op.
    assert utils.merge_layer({"a": 1}, {"b": None}) == {"a": 1}


def test_merge_base_expression_against_below():
    assert utils.merge_layer({"temperature": 1.0},
                             {"temperature": "base * 0.8"}) == {
        "temperature": 0.8}


def test_merge_base_expression_missing_below_skipped(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="llama_packer.utils"):
        assert utils.merge_layer({}, {"temperature": "base * 0.8"}) == {}
    assert any("no numeric value below" in r.message
               for r in caplog.records)


def test_merge_does_not_alias_inputs():
    below = {"d": {"x": 1}, "l": ["a"]}
    above = {"d": {"y": 2}, "l": ["b"], "s": "v"}
    out = utils.merge_layer(below, above)
    out["d"]["x"] = 99
    out["l"].append("zzz")
    assert below == {"d": {"x": 1}, "l": ["a"]}
    assert above == {"d": {"y": 2}, "l": ["b"], "s": "v"}


def test_resolve_params_none_deletes():
    # profiles: coding: {top_p: None} drops the inherited top_p outright.
    assert utils.resolve_params({"top_p": None, "temperature": 0.2},
                                {"top_p": 0.9, "temperature": 0.5}) == {
        "temperature": 0.2}


# ── ScopeStack wiring ─────────────────────────────────────────────────────

def test_scope_defaults_lists_append():
    from llama_packer.scope import ScopeStack
    stack = ScopeStack()
    stack.push({"defaults": {"capabilities": ["tools"]}}, origin="t")
    stack.push({"defaults": {"capabilities": ["image"]}}, origin="t")
    assert stack.defaults == {"capabilities": ["tools", "image"]}
    stack.pop()
    stack.pop()


def test_scope_rules_append_and_delete(make_model):
    from llama_packer.scope import ScopeStack
    m = make_model("m", loras=["a.gguf"], cache_type="q8_0")
    stack = ScopeStack()
    stack.push({"overrides": [
        {"when": {"name": "m"},
         "loras": ["b.gguf", "-a.gguf"],
         "cache_type": None},
    ]}, origin="t")
    changed = stack.apply_rules(m)
    assert m.frontmatter["loras"] == ["b.gguf"]
    assert "cache_type" not in m.frontmatter
    assert changed == {"loras", "cache_type"}
    stack.pop()


# ── sidecar mode layering (partial-mode inheritance fix) ──────────────────

def test_sidecar_mode_inherits_profile_keys(make_model):
    # The bug: a partial sidecar mode dropped every uninherited profile key.
    # (Single declared mode becomes the default → bare ${MODEL_ID} key.)
    m = make_model("m", modes={"coding": {"temperature": 0.2}})
    group = [("coding", {"temperature": 0.5, "top_p": 0.9, "min_p": 0.05})]
    params = _build_mode_params(m, group, {})
    assert params["${MODEL_ID}"]["temperature"] == 0.2
    assert params["${MODEL_ID}"]["top_p"] == 0.9
    assert params["${MODEL_ID}"]["min_p"] == 0.05


def test_sidecar_mode_falls_back_to_defaults(make_model):
    # No same-named profile: layer over the fleet defaults.
    m = make_model("m", modes={"custom": {"temperature": 0.3},
                               "other": {"temperature": 0.4}})
    group = [("other", {"temperature": 0.7})]
    params = _build_mode_params(
        m, group, {"temperature": 0.6, "top_p": 0.8})
    assert params["${MODEL_ID}"]["temperature"] == 0.3
    assert params["${MODEL_ID}"]["top_p"] == 0.8
    assert params["${MODEL_ID}:other"]["temperature"] == 0.4


def test_sidecar_mode_none_deletes_profile_key(make_model):
    m = make_model("m", modes={"coding": {"min_p": None}})
    group = [("coding", {"temperature": 0.5, "min_p": 0.05})]
    params = _build_mode_params(m, group, {})
    assert params["${MODEL_ID}"]["temperature"] == 0.5
    assert "min_p" not in params["${MODEL_ID}"]


# ── intrinsics warnings ───────────────────────────────────────────────────

def test_scope_defaults_intrinsic_warns(caplog):
    import logging
    from llama_packer.scope import ScopeStack
    stack = ScopeStack()
    with caplog.at_level(logging.WARNING, logger="llama_packer.scope"):
        stack.push({"defaults": {"capabilities": ["tools"],
                                 "cache_type": "q8_0"}}, origin="dir/x")
    assert any("intrinsic" in r.message and "capabilities" in r.message
               for r in caplog.records)
    assert not any("cache_type" in r.message for r in caplog.records)
    stack.pop()


def test_sidecar_capability_removal_warns(tmp_path, caplog):
    import logging
    from llama_packer import discover
    from llama_packer.scope import ScopeStack
    (tmp_path / "m.gguf").write_bytes(b"dummy")
    out = []
    with caplog.at_level(logging.WARNING, logger="llama_packer.discover"):
        discover._build(tmp_path / "m.md",
                        {"name": "m", "capabilities": ["-tools"]},
                        None, ScopeStack(), None, out)
    assert any("add capabilities" in r.message for r in caplog.records)


# ── companion blocks ──────────────────────────────────────────────────────

def test_block_resolves_and_views_split(make_model, tmp_path):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                "capabilities": ["image"]})
    assert m.mmproj is not None
    assert m.mmproj_overlay == {"capabilities": ["image"]}
    assert m.capabilities == []
    on = m.view_for(True)
    assert on is not m
    assert on.capabilities == ["image"]
    assert m.view_for(False) is m


def test_bare_string_mmproj_errors(make_model, tmp_path):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", mmproj="v-mmproj.gguf")
    assert m.mmproj is None
    assert getattr(m, "_override_error", None) is not None
    assert "mapping" in m._override_error


def test_block_denied_keys_error(make_model, tmp_path):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", mmproj={"file": "v-mmproj.gguf", "role": "chat"})
    assert m.mmproj is None
    assert getattr(m, "_override_error", None) is not None


def test_block_missing_file_warns_without_overlay(make_model, tmp_path):
    m = make_model("v", mmproj={"file": "absent.gguf",
                                "capabilities": ["image"]})
    assert m.mmproj is None
    assert m.mmproj_overlay == {}
    assert m.view_for(True) is m


def test_block_list_appends_to_base(make_model, tmp_path):
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", capabilities=["tools"],
                   mmproj={"file": "v-mmproj.gguf",
                           "capabilities": ["image"]})
    assert m.capabilities == ["tools"]
    assert m.view_for(True).capabilities == ["tools", "image"]
