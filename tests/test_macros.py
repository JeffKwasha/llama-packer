# tests/test_macros.py
"""Contract tests for the flag-macro engine.

Macros are an *observable* output: ``Macro.definitions()`` lands in the emitted
config's ``macros:`` block and ``Macro.apply`` rewrites command strings.  These
tests pin that observable contract — matching semantics, idempotency, provenance
naming and source precedence — rather than the private matching loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from llama_packer.macros import Macro, Macros, _flags_from_cache_parallel
from llama_packer.profiles import Profiles


@pytest.fixture(autouse=True)
def _clean_registry():
    Macro.clear()
    yield
    Macro.clear()


def _profiles(defaults=None, profiles=None) -> Profiles:
    return Profiles({"defaults": defaults or {}, "profiles": profiles or {}})


# ── registry contract ─────────────────────────────────────────────────────

def test_lookup_by_name_and_missing_returns_none():
    m = Macro("X", "src", {"--a": "1"})
    assert Macro["X"] is m
    assert Macro.get("X") is m
    assert Macro.get("missing") is None


def test_duplicate_name_replaces_previous(caplog):
    Macro("X", "first", {"--a": "1"})
    with caplog.at_level(logging.WARNING):
        Macro("X", "second", {"--b": "2"})
    # Last registration wins and the collision is surfaced.
    assert Macro["X"].source == "second"
    assert Macro["X"].flags == {"--b": "2"}
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_all_is_ordered_longest_first_then_name():
    Macro("small", "s", {"--a": "1"})
    Macro("mid", "s", {"--a": "1", "--b": "2"})
    Macro("big", "s", {"--a": "1", "--b": "2", "--c": "3"})
    assert [m.name for m in Macro.all()] == ["big", "mid", "small"]


def test_definitions_render_flags_and_preserve_order():
    Macro("X", "s", {"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"})
    Macro("Y", "s", {"--jinja": ""})
    Macro("E", "s", {})
    defs = Macro.definitions()
    assert defs["X"] == "--cache-type-k q8_0 --cache-type-v q8_0"
    assert defs["Y"] == "--jinja"           # valueless -> bare flag
    assert defs["E"] == ""


# ── apply(): command rewriting contract ───────────────────────────────────

def test_apply_replaces_matching_subset_at_first_occurrence():
    Macro("CACHE", "s", {"--cache-type-k": "q8_0", "--cache-type-v": "q8_0"})
    out = Macro.apply(
        "llama-server --parallel 4 --cache-type-k q8_0 --cache-type-v q8_0 --seed 1")
    assert out == "llama-server --parallel 4 ${CACHE} --seed 1"


def test_apply_requires_exact_value_match():
    Macro("CACHE", "s", {"--cache-type-k": "q8_0"})
    cmd = "llama-server --cache-type-k f16"
    assert Macro.apply(cmd) == cmd


def test_apply_leaves_unmatched_flags_and_head_untouched():
    Macro("A", "s", {"--a": "1"})
    assert Macro.apply("bin --a 1 --flag val --bool") == "bin ${A} --flag val --bool"


def test_apply_emits_each_macro_once():
    Macro("A", "s", {"--a": "1"})
    out = Macro.apply("bin --a 1 --b 2 --a 1")
    # Duplicate flags collapse in the flag map; the macro ref is emitted once.
    assert out.count("${A}") == 1
    assert "--a" not in out


def test_apply_greedy_longest_first_claims_overlapping_flags():
    Macro("BIG", "s", {"--a": "1", "--b": "2"})
    Macro("SMALL", "s", {"--b": "2"})
    out = Macro.apply("bin --a 1 --b 2 --c 3")
    assert out == "bin ${BIG} --c 3"
    assert "${SMALL}" not in out


def test_apply_multiple_disjoint_macros():
    Macro("A", "s", {"--a": "1"})
    Macro("B", "s", {"--b": "2"})
    assert Macro.apply("bin --a 1 --b 2") == "bin ${A} ${B}"


def test_apply_is_idempotent():
    Macro("A", "s", {"--a": "1"})
    once = Macro.apply("bin --a 1 --b 2")
    assert Macro.apply(once) == once


def test_apply_no_match_empty_and_no_registry_are_identity():
    Macro("A", "s", {"--a": "1"})
    assert Macro.apply("bin --b 2") == "bin --b 2"
    assert Macro.apply("") == ""
    Macro.clear()
    assert Macro.apply("bin --a 1") == "bin --a 1"


def test_apply_unparsable_command_is_unchanged(caplog):
    Macro("A", "s", {"--a": "1"})
    bad = "bin --x 'unterminated"
    with caplog.at_level(logging.WARNING):
        assert Macro.apply(bad) == bad


def test_apply_empty_flag_macro_never_matches():
    Macro("EMPTY", "s", {})
    assert Macro.apply("bin --a 1") == "bin --a 1"


# ── apply_to_dict() contract ──────────────────────────────────────────────

def test_apply_to_dict_reports_remaining_and_matched():
    Macro("A", "s", {"--a": "1"})
    remaining, matched = Macro.apply_to_dict({"--a": "1", "--b": "2"})
    assert remaining == {"--b": "2"}
    assert matched == ["A"]


def test_apply_to_dict_requires_exact_value():
    Macro("A", "s", {"--a": "1"})
    remaining, matched = Macro.apply_to_dict({"--a": "2"})
    assert remaining == {"--a": "2"}
    assert matched == []


# ── Macros builder contract ───────────────────────────────────────────────

def test_builder_always_registers_common_gpu():
    Macros({})
    assert Macro.get("COMMON_GPU").flags == {"--n-gpu-layers": "999"}


def test_builder_parses_explicit_macro_strings():
    Macros({"macros": {"FAST": "--flash-attn on -b 512"}})
    assert Macro.get("FAST").flags == {"--flash-attn": "on", "-b": "512"}


def test_builder_ignores_non_string_and_unparsable_macros(caplog):
    with caplog.at_level(logging.WARNING):
        Macros({"macros": {"BADNUM": 123, "BADSH": "--a 'oops",
                           "GOOD": "--x 1"}})
    assert Macro.get("BADNUM") is None
    assert Macro.get("BADSH") is None
    assert Macro.get("GOOD") is not None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_builder_profile_defaults_macro():
    Macros({}, _profiles(defaults={"cache_type": "q8_0", "parallel": 4}))
    assert Macro.get("PROFILE_DEFAULTS").flags == {
        "--cache-type-k": "q8_0", "--cache-type-v": "q8_0", "--parallel": "4"}


def test_builder_per_profile_names_are_normalised():
    Macros({}, _profiles(profiles={
        "fast-tools": {"parallel": 8},
        "plain": {"cache_type": "f16"},
        "noop": {},
    }))
    assert Macro.get("PROFILE_FAST_TOOLS").flags == {"--parallel": "8"}
    assert Macro.get("PROFILE_PLAIN").flags == {
        "--cache-type-k": "f16", "--cache-type-v": "f16"}
    assert Macro.get("PROFILE_NOOP") is None


def test_builder_last_registration_wins_across_builds():
    Macros({"macros": {"X": "--a 1"}})
    Macros({"macros": {"X": "--b 2"}})
    assert Macro.get("X").flags == {"--b": "2"}


def test_builder_models_yaml_macro_named_from_directory(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "models.yaml").write_text("defaults:\n  cache_type: f16\n")
    Macros({}, None, [tmp_path / "models"])
    m = Macro.get("MODELS_CHAT")
    assert m is not None
    assert m.flags["--cache-type-k"] == "f16"


def test_builder_chat_template_implies_jinja_and_resolves_relative(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "tpl.jinja").write_text("{{ x }}")
    (d / "models.yaml").write_text("defaults:\n  chat_template: tpl.jinja\n")
    Macros({}, None, [tmp_path / "models"])
    flags = Macro.get("MODELS_CHAT").flags
    assert flags["--jinja"] == ""
    assert flags["--chat-template-file"] == str(d / "tpl.jinja")


def test_builder_chat_template_paths_pass_through_sub(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "tpl.jinja").write_text("{{ x }}")
    (d / "models.yaml").write_text("defaults:\n  chat_template: tpl.jinja\n")
    Macros({}, None, [tmp_path / "models"],
           sub=lambda p: "${MODELS_DIR}/" + Path(p).name)
    assert Macro.get("MODELS_CHAT").flags["--chat-template-file"] == \
        "${MODELS_DIR}/tpl.jinja"


def test_builder_loras_accepts_string_and_list(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "a.lora").write_text("a")
    (d / "b.lora").write_text("b")
    (d / "one.yaml").parent.mkdir(parents=True, exist_ok=True)
    # Single string -> one value.
    (d / "models.yaml").write_text("defaults:\n  loras: a.lora\n")
    Macros({}, None, [tmp_path / "models"])
    assert Macro.get("MODELS_CHAT").flags["--lora"] == str(d / "a.lora")
    Macro.clear()
    # List -> comma-joined, order preserved.
    (d / "models.yaml").write_text("defaults:\n  loras: [a.lora, b.lora]\n")
    Macros({}, None, [tmp_path / "models"])
    assert Macro.get("MODELS_CHAT").flags["--lora"] == \
        f"{d / 'a.lora'},{d / 'b.lora'}"


def test_builder_reasoning_flags(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "models.yaml").write_text(
        "defaults:\n  reasoning-format: deepseek\n  reasoning-preserve: true\n")
    Macros({}, None, [tmp_path / "models"])
    flags = Macro.get("MODELS_CHAT").flags
    assert flags["--reasoning-format"] == "deepseek"
    assert flags["--reasoning-preserve"] == ""


def test_builder_cli_args_win_over_other_settings(tmp_path):
    d = tmp_path / "models" / "chat"
    d.mkdir(parents=True)
    (d / "models.yaml").write_text(
        "defaults:\n  cache_type: q8_0\n  cli_args: --cache-type-k f16\n")
    Macros({}, None, [tmp_path / "models"])
    flags = Macro.get("MODELS_CHAT").flags
    assert flags["--cache-type-k"] == "f16"      # cli_args last-write-wins
    assert flags["--cache-type-v"] == "q8_0"


def test_flags_from_cache_parallel_ignores_non_mappings():
    assert _flags_from_cache_parallel(None) == {}
    assert _flags_from_cache_parallel("not-a-dict") == {}
    assert _flags_from_cache_parallel({"cache_type": None, "parallel": None}) == {}
