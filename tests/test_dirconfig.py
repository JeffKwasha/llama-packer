# tests/test_dirconfig.py
"""Directory-scoped models.yaml: frontmatter defaults, scoped overrides,
precedence (inner > outer > global), and validation."""

from __future__ import annotations

import logging

import pytest

from llama_packer.model import Model
from llama_packer.scope import ScopeStack
from llama_packer.utils import load_dir_config, validate_dir_roles

from conftest import make_model  # noqa: F401  (fixture via pytest, direct import for emit test)


def _sidecar(name: str, extra: str = "") -> str:
    return f"---\nname: {name}\n{extra}---\n"


@pytest.fixture(autouse=True)
def _clear_dir_config_cache():
    from llama_packer import utils
    utils._dir_config_cache.clear()
    yield
    utils._dir_config_cache.clear()


def _discover(root, profiles_cfg=None):
    """Run full discovery with optional global (profiles.yaml) rules."""
    stack = ScopeStack()
    stack.push({"overrides": (profiles_cfg or {}).get("overrides")},
               origin="profiles.yaml")
    return Model.from_dir(root, generate_stubs=False, stack=stack)


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

def _pair_in(tmp_path, rel: str):
    """Write an <rel>.gguf + authored sidecar; return the sidecar path."""
    md = tmp_path / rel
    md.parent.mkdir(parents=True, exist_ok=True)
    md.with_suffix(".gguf").write_bytes(b"x")
    md.write_text(_sidecar(md.stem))
    return md


def test_scoped_rule_applies_only_within_subtree(tmp_path):
    root = tmp_path / "models"
    qwen = root / "chat" / "qwen3"
    qwen.mkdir(parents=True)
    (qwen / "models.yaml").write_text(
        "overrides:\n  - when: true\n    chat_template: qwen.jinja\n")

    inside = _pair_in(qwen, "chat/qwen3/a.md")
    outside = _pair_in(root / "chat", "chat/b.md")

    by_stem = {m.stem: m for m in _discover(root)}
    assert by_stem["a"].frontmatter["chat_template"] == "qwen.jinja"
    assert "chat_template" not in by_stem[outside.with_suffix("").stem].frontmatter
    del inside


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
    _pair_in(qwen, "chat/qwen3/a.md")

    m = next(m for m in _discover(root, global_profiles) if m.stem == "a")
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

    _pair_in(inner, "chat/qwen3/a.md")
    _pair_in(outer, "chat/b.md")

    by_stem = {m.stem: m for m in _discover(root)}
    assert by_stem["a"].frontmatter["chat_template"] == "inner.jinja"
    assert by_stem["b"].frontmatter["chat_template"] == "outer.jinja"


def test_malformed_scoped_rule_names_models_yaml(tmp_path):
    d = tmp_path / "chat"
    d.mkdir()
    (d / "models.yaml").write_text(
        "overrides:\n  - chat_template: a.jinja\n")  # missing 'when'
    with pytest.raises(SystemExit, match="models.yaml"):
        _discover(tmp_path)


def test_defaults_and_rules_both_support_mmproj_false(tmp_path):
    # The same key works at both precedence levels via the same engine:
    # directory defaults and pattern rules both disable the vision companion.
    root = tmp_path / "models"
    for sub in ("d", "r"):
        (root / "vision" / sub).mkdir(parents=True)
        gguf = root / "vision" / sub / f"{sub}.gguf"
        gguf.write_bytes(b"x")
        mm = root / "vision" / sub / f"{sub}-mmproj.gguf"
        mm.write_bytes(b"x")
    (root / "vision" / "d" / "models.yaml").write_text("defaults:\n  mmproj: false\n")
    (root / "vision" / "r" / "models.yaml").write_text(
        "overrides:\n  - when: true\n    mmproj: false\n")

    by_stem = {m.stem: m for m in _discover(root)}
    assert by_stem["d"].mmproj is None
    assert by_stem["r"].mmproj is None


# ── validate_dir_roles ─────────────────────────────────────────────────────

def test_validate_dir_roles():
    assert validate_dir_roles({"ocr": "chat"}) is None
    err = validate_dir_roles({"ocr": "vision"})
    assert err and "unknown role" in err


# ── Entry-id collision detection ───────────────────────────────────────────

def test_emit_config_duplicate_id_raises(make_model):
    from llama_packer.writer import emit_config, Planner, TEXT_SUFFIX

    # template_id is slug of sidecar stem (not name). Two different stems that
    # slug to the same id are "my model" and "my-model" -> both "my-model".
    a = make_model("my model", name="My Model")
    b = make_model("my-model", name="my-model!")  # same slug as "my model"

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
