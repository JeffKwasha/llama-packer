# tests/test_file_refs.py
"""File refs: dict ``{hf_repo:, file:[, pick:]}`` form for every file key.

``{file: x}`` ≡ bare string ``x``; ``{hf_repo:, file:}`` names a hub file;
``pick`` scopes snapshot selection (``newest``/``oldest``) and in-snapshot
depth (``top`` = snapshot root only). Ambiguity still fails loud.
"""

from __future__ import annotations

import logging
import os

from llama_packer.model import Model, default_finder
from llama_packer.overrides import resolve_setting_paths
from llama_packer.utils import parse_pick

from conftest import _hf_tree


def _stamp(hf_home, repo, rev, mtime):
    snap = (hf_home / "hub" / f"models--{repo.replace('/', '--')}"
            / "snapshots" / rev)
    os.utime(snap, (mtime, mtime))


def test_parse_pick_defaults():
    assert parse_pick(None) == ("main", False)
    assert parse_pick("newest,top") == ("newest", True)
    assert parse_pick(["oldest", "top"]) == ("oldest", True)
    assert parse_pick("  NEWEST  top ") == ("newest", True)


def test_parse_pick_unknown_token_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert parse_pick("newest,bogus") == ("newest", False)
    assert any("unknown pick token" in r.message for r in caplog.records)


def test_dict_template_resolves_from_hub(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1",), files=("chat_template.jinja",))
    m = make_model("q", chat_template={"hf_repo": "org/repo",
                                       "file": "chat_template.jinja"})
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    assert m._resolved_chat_template.name == "chat_template.jinja"
    assert "r1" in str(m._resolved_chat_template)


def test_file_only_dict_equals_bare_string(make_model, tmp_path):
    (tmp_path / "local.jinja").write_text("x")
    m1 = make_model("a", chat_template="local.jinja")
    m2 = make_model("b", chat_template={"file": "local.jinja"})
    assert resolve_setting_paths(m1) == []
    assert resolve_setting_paths(m2) == []
    assert m1._resolved_chat_template == m2._resolved_chat_template


def test_dict_template_tracks_refs_main(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1", "r2"),
                       files=("chat_template.jinja",),
                       contents={("r1", "chat_template.jinja"): b"one",
                                 ("r2", "chat_template.jinja"): b"two"})
    m = make_model("q", chat_template={"hf_repo": "org/repo",
                                       "file": "chat_template.jinja"})
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    assert "r1" in str(m._resolved_chat_template)
    (hf_home / "hub" / "models--org--repo" / "refs" / "main").write_text("r2")
    del m._resolved_chat_template
    assert resolve_setting_paths(m) == []
    assert "r2" in str(m._resolved_chat_template)


def test_pick_newest_oldest_without_ref(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1", "r2"), ref=None,
                       files=("chat_template.jinja",))
    _stamp(hf_home, "org/repo", "r1", 1000)
    _stamp(hf_home, "org/repo", "r2", 2000)
    ref = {"hf_repo": "org/repo", "file": "chat_template.jinja"}
    m = make_model("q", chat_template={**ref, "pick": "newest"})
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    assert "r2" in str(m._resolved_chat_template)
    m2 = make_model("w", chat_template={**ref, "pick": "oldest"})
    m2._hf_home = hf_home
    assert resolve_setting_paths(m2) == []
    assert "r1" in str(m2._resolved_chat_template)


def test_pick_top_prefers_snapshot_root(make_model, tmp_path, caplog):
    hf_home = _hf_tree(tmp_path, files=("chat_template.jinja",
                                        "archive/v1/chat_template.jinja"))
    ref = {"hf_repo": "org/repo", "file": "chat_template.jinja"}
    m = make_model("q", chat_template=ref)
    m._hf_home = hf_home
    with caplog.at_level(logging.WARNING):
        errors = resolve_setting_paths(m)
    assert errors  # ambiguous across depths
    assert getattr(m, "_resolved_chat_template", None) is None
    m2 = make_model("w", chat_template={**ref, "pick": "top"})
    m2._hf_home = hf_home
    assert resolve_setting_paths(m2) == []
    hit = str(m2._resolved_chat_template)
    assert hit.endswith("/chat_template.jinja") and "archive" not in hit


def test_hub_string_form_for_template(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1",), files=("chat_template.jinja",))
    m = make_model("q", chat_template="hub:org/repo:chat_template.jinja")
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    assert "r1" in str(m._resolved_chat_template)


def test_dict_loras_mixed_local_and_hub(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, files=("hub-lora.gguf",))
    (tmp_path / "local-lora.gguf").write_bytes(b"x")
    m = make_model("q", loras=["local-lora.gguf",
                               {"hf_repo": "org/repo", "file": "hub-lora.gguf"}])
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    names = [p.name for p in m._resolved_loras]
    assert names == ["local-lora.gguf", "hub-lora.gguf"]


def test_malformed_refs_are_errors_not_crashes(make_model, tmp_path):
    m = make_model("q", chat_template={"hf_repo": "org/repo"})
    errors = resolve_setting_paths(m)
    assert any("chat_template" in e for e in errors)
    m2 = make_model("w", chat_template=123)
    errors = resolve_setting_paths(m2)
    assert any("chat_template" in e for e in errors)
    m3 = make_model("e", loras=[{"file": "nope.gguf"}])
    errors = resolve_setting_paths(m3)
    assert any("lora" in e for e in errors)


def test_missing_hub_file_hints_hf_download(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path)
    m = make_model("q", chat_template={"hf_repo": "org/repo",
                                       "file": "nope.jinja"})
    m._hf_home = hf_home
    errors = resolve_setting_paths(m)
    assert any("hf download org/repo" in e for e in errors)


def test_model_dict_form_resolves_weight_from_hub(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("Qwen-x.Q4_K_M.gguf",))
    md_path = tmp_path / "qwen.md"
    fm = {"name": "qwen",
          "model": {"hf_repo": "org/repo", "file": "Qwen-x.Q4_K_M.gguf"}}
    m = Model(md_path, fm, hf_home=hf_home)
    assert m.gguf_path is not None
    assert m.gguf_path.name == "Qwen-x.Q4_K_M.gguf"


def test_from_ref_dict_with_pick_newest(tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1", "r2"),
                       files=("m.Q4_K_M.gguf",), ref=None)
    _stamp(hf_home, "org/repo", "r1", 1000)
    _stamp(hf_home, "org/repo", "r2", 2000)
    hit = Model.from_ref({"hf_repo": "org/repo", "file": "m.Q4_K_M.gguf",
                          "pick": "newest"},
                         anchors=[], hf_home=hf_home,
                         finder=default_finder())
    assert hit is not None and "r2" in str(hit.gguf_path)


def test_macros_dict_template(tmp_path):
    from llama_packer.macros import _flags_for_settings
    hf_home = _hf_tree(tmp_path, revs=("r1",), files=("chat_template.jinja",))
    d = tmp_path / "models"
    d.mkdir()
    flags = _flags_for_settings(
        {"chat_template": {"hf_repo": "org/repo",
                           "file": "chat_template.jinja"}}, d, None,
        hf_home=hf_home)
    assert flags["--jinja"] == ""
    assert flags["--chat-template-file"].endswith("chat_template.jinja")
    assert "r1" in flags["--chat-template-file"]


def test_macros_unresolvable_ref_skipped(tmp_path, caplog):
    from llama_packer.macros import _flags_for_settings
    d = tmp_path / "models"
    d.mkdir()
    with caplog.at_level(logging.WARNING):
        flags = _flags_for_settings(
            {"chat_template": {"hf_repo": "org/nope",
                               "file": "nope.jinja"}}, d, None,
            hf_home=tmp_path / "hf")
    assert "--chat-template-file" not in flags


def test_weights_only_ignores_non_weight_namesakes(tmp_path):
    from llama_packer.model import WeightFinder
    hf_home = _hf_tree(tmp_path, files=("m.gguf", "m.md"))
    f = WeightFinder()
    hit = Model.from_ref("m.gguf", hf_repo="org/repo",
                         hf_home=hf_home, finder=f)
    assert hit is not None and hit.gguf_path is not None
    assert hit.gguf_path.name == "m.gguf"
    # The any-file index sees both; the weight path must not waver.
    any_hit = f.resolve_hf_file("org/repo", "m.md", hf_home)
    assert any_hit is not None and any_hit.name == "m.md"


def test_from_ref_dict_equals_string_for_weights(tmp_path):
    hf_home = _hf_tree(tmp_path, files=("w.Q4_K_M.gguf",))
    kw = {"hf_home": hf_home, "finder": default_finder()}
    a = Model.from_ref("w.Q4_K_M.gguf", hf_repo="org/repo", **kw)
    b = Model.from_ref({"hf_repo": "org/repo", "file": "w.Q4_K_M.gguf"},
                       **kw)
    assert a is not None and b is not None
    assert a.gguf_path == b.gguf_path


def test_pick_list_form_resolves(make_model, tmp_path):
    hf_home = _hf_tree(tmp_path, revs=("r1", "r2"),
                       files=("chat_template.jinja",), ref=None)
    _stamp(hf_home, "org/repo", "r1", 1000)
    _stamp(hf_home, "org/repo", "r2", 2000)
    m = make_model("q", chat_template={"hf_repo": "org/repo",
                                       "file": "chat_template.jinja",
                                       "pick": ["oldest", "top"]})
    m._hf_home = hf_home
    assert resolve_setting_paths(m) == []
    assert "r1" in str(m._resolved_chat_template)


def test_local_ref_with_bad_pick_warns_nothing(make_model, tmp_path, caplog):
    (tmp_path / "local.jinja").write_text("x")
    m = make_model("q", chat_template={"file": "local.jinja",
                                       "pick": "bogus"})
    with caplog.at_level(logging.WARNING):
        assert resolve_setting_paths(m) == []
    assert not [r for r in caplog.records if "pick" in r.message]


def test_absolute_file_with_repo_warns_ignored(make_model, tmp_path, caplog):
    tpl = tmp_path / "abs.jinja"
    tpl.write_text("x")
    m = make_model("q", chat_template={"hf_repo": "org/repo",
                                       "file": str(tpl)})
    with caplog.at_level(logging.WARNING):
        assert resolve_setting_paths(m) == []
    assert any("ignoring hf_repo" in r.message for r in caplog.records)
    assert m._resolved_chat_template == tpl.resolve()
