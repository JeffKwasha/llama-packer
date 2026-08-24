# tests/test_dirconfig.py
"""Directory-scoped models.yaml: frontmatter defaults, scoped overrides,
precedence (inner > outer > global), and validation."""

from __future__ import annotations

import logging

import pytest

from llama_packer.model import Model
from llama_packer.overrides import apply_overrides, compile_scoped_rules
from llama_packer.utils import collect_dir_configs, load_dir_config, validate_dir_roles

from conftest import make_model  # noqa: F401  (fixture via pytest, direct import for emit test)


def _sidecar(name: str, extra: str = "") -> str:
    return f"---\nname: {name}\n{extra}---\n"


@pytest.fixture(autouse=True)
def _clear_dir_config_cache():
    from llama_packer import utils
    utils._dir_config_cache.clear()
    yield
    utils._dir_config_cache.clear()


# ── Frontmatter defaults inheritance ──────────────────────────────────────

def test_defaults_inherit_inner_beats_outer_beats_sidecar(tmp_path):
    root = tmp_path / "models"
    sub = root / "chat" / "qwen3"
    sub.mkdir(parents=True)
    (root / "chat" / "models.yaml").write_text(
        "defaults:\n  context_length: 8192\n  backend: llama-server\n")
    (sub / "models.yaml").write_text(
        "defaults:\n  context_length: 16384\n")
    # Sidecar wins over both; the other key is inherited from chat/ level.
    (sub / "m.gguf").write_bytes(b"x")
    (sub / "m.md").write_text(_sidecar("M", "context_length: 32768\n"))
    (root / "chat" / "plain.gguf").write_bytes(b"x")  # stub gets defaults too

    models = Model.from_dir(root, generate_stubs=True)
    by_stem = {m.stem: m for m in models}
    assert by_stem["m"].frontmatter["context_length"] == 32768
    assert by_stem["m"].frontmatter["backend"] == "llama-server"
    assert by_stem["plain"].frontmatter["context_length"] == 8192


def test_forbidden_default_keys_rejected(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "models.yaml").write_text("defaults:\n  name: evil\n")
    with pytest.raises(SystemExit, match="per-model"):
        load_dir_config(d)


# ── Scoped override rules ──────────────────────────────────────────────────

def _model_in(tmp_path, rel: str):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.with_suffix(".gguf")).write_bytes(b"x")
    from llama_packer.utils import parse_frontmatter
    fm = parse_frontmatter(p.with_suffix(".md")) if p.with_suffix(".md").is_file() else {}
    if not fm.get("name"):
        fm = {"name": p.stem, "context_length": 32768}
        p.with_suffix(".md").write_text(_sidecar(p.stem))
    return Model(p.with_suffix(".md"), fm)


def test_scoped_rule_applies_only_within_subtree(tmp_path):
    root = tmp_path / "models"
    qwen = root / "chat" / "qwen3"
    qwen.mkdir(parents=True)
    (qwen / "models.yaml").write_text(
        "overrides:\n  - when: true\n    chat_template: qwen.jinja\n")

    inside = _model_in(qwen, "chat/qwen3/a.md")
    outside = _model_in(root / "chat", "chat/b.md")

    dir_cfgs = collect_dir_configs([root])
    scoped = compile_scoped_rules(dir_cfgs)
    apply_overrides([inside, outside], {}, scoped_rules=scoped)

    assert inside.frontmatter["chat_template"] == "qwen.jinja"
    assert "chat_template" not in outside.frontmatter


def test_inner_scope_and_global_precedence(tmp_path):
    root = tmp_path / "models"
    qwen = root / "chat" / "qwen3"
    qwen.mkdir(parents=True)
    (qwen / "models.yaml").write_text(
        "overrides:\n  - when: true\n    chat_template: inner.jinja\n")
    global_profiles = {"overrides": [
        {"when": True, "chat_template": "global.jinja"},
        {"when": True, "cli_args": "--seed 7"},
    ]}

    m = _model_in(qwen, "chat/qwen3/a.md")
    scoped = compile_scoped_rules(collect_dir_configs([root]))
    apply_overrides([m], global_profiles, scoped_rules=scoped)

    # Inner scope beats global; untouched global keys still land.
    assert m.frontmatter["chat_template"] == "inner.jinja"
    assert m.frontmatter["cli_args"] == "--seed 7"


def test_outer_scope_between_global_and_inner(tmp_path):
    root = tmp_path / "models"
    outer = root / "chat"
    inner = root / "chat" / "qwen3"
    inner.mkdir(parents=True)
    (outer / "models.yaml").write_text(
        "overrides:\n  - when: true\n    chat_template: outer.jinja\n")
    (inner / "models.yaml").write_text(
        "overrides:\n  - when: true\n    chat_template: inner.jinja\n")

    m = _model_in(inner, "chat/qwen3/a.md")
    scoped = compile_scoped_rules(collect_dir_configs([root]))
    apply_overrides([m], {}, scoped_rules=scoped)
    assert m.frontmatter["chat_template"] == "inner.jinja"

    sibling = _model_in(outer, "chat/b.md")
    apply_overrides([sibling], {}, scoped_rules=scoped)
    assert sibling.frontmatter["chat_template"] == "outer.jinja"


def test_malformed_scoped_rule_names_models_yaml():
    cfgs = {"/x/chat": {"overrides": [{"chat_template": "a.jinja"}]}}
    with pytest.raises(SystemExit, match="models.yaml"):
        compile_scoped_rules(cfgs)


# ── validate_dir_roles ─────────────────────────────────────────────────────

def test_validate_dir_roles():
    assert validate_dir_roles({"ocr": "chat"}) is None
    err = validate_dir_roles({"ocr": "vision"})
    assert err and "unknown role" in err


# ── Entry-id collision detection ───────────────────────────────────────────

def test_emit_config_duplicate_id_raises(make_model):
    from llama_packer.writer import emit_config, Planner, TEXT_SUFFIX

    a = make_model("dup", name="My Model")
    b = make_model("dup2", name="my-model!")  # slugs to the same entry id

    class P:
        pass

    plan = Planner.__new__(Planner)  # not used: empty plans per model
    plan.chat_ctx = None
    plans = {m.stem: [] for m in (a, b)}
    # Build entries directly through emit_config with one variant each.
    from llama_packer.writer import Variant
    v = Variant(parallel=1, cache_type="q8_0", spare_mb=0,
                profiles_group=[("default", {})], ctx_size=4096,
                include_mmproj=True, vision_ctx=None)
    plans = {m.stem: [v] for m in (a, b)}

    profiles = P()
    profiles.defaults = {}
    with pytest.raises(ValueError, match="duplicate entry id"):
        emit_config([a, b], plans, profiles, {})

    assert TEXT_SUFFIX == "-text"
