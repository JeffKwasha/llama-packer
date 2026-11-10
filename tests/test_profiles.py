"""profiles.yaml key validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from llama_packer.profiles import check_profiles, validate_profiles_keys


def test_unknown_profiles_key_warns_with_a_hint(caplog):
    # Validates: docs/plans/requirement-trace.md, requirement #89
    # nothing reads warns (with a near-miss hint) instead of doing nothing.
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        validate_profiles_keys({"override": [], "taks": {}}, log)
    msgs = [r.message for r in caplog.records]
    assert any("unknown key 'override'" in m
               and "did you mean 'overrides'" in m for m in msgs)
    assert any("unknown key 'taks'" in m and "did you mean 'tasks'" in m
               for m in msgs)


def test_check_profiles_names_every_nested_error():
    # Validates: docs/plans/requirement-trace.md, requirement #106
    problems = check_profiles({
        "profiles": {"default": {"temperatur": 0.5}},
        "tasks": {"chat": {"role": "chat"}},
        "matrix": {"auto_paralel": False, "sets": ["x"]},
        "pools": {"default": {"resserve_extra": "4G"}, "ram": "2G"},
        "hardware": {"basline_mb": "512m"},
        "dirs": {"ocr": "ocr"},
    })
    text = "\n".join(problems)
    assert "profiles.default: unknown key 'temperatur'" in text
    assert "tasks.chat: endpoint is required" in text
    assert "matrix: unknown key 'auto_paralel'" in text
    assert "matrix.sets: must be a mapping" in text
    assert "pools.default: unknown key 'resserve_extra'" in text
    assert "pools.ram: must be a mapping" in text
    assert "hardware: unknown key 'basline_mb'" in text
    assert "dirs: retired" in text


def test_check_profiles_validates_sets_and_retired_routing():
    # Validates: docs/plans/requirement-trace.md, requirement #109
    problems = check_profiles({
        "profiles": {"default": {}},
        "routing": {"use": "turbo"},
        "matrix": {"sets": {"rag": 1}},
    })
    text = "\n".join(problems)
    assert "routing: retired" in text
    assert "matrix.sets.rag: must be an expression string" in text


def test_matrix_sets_are_accepted_and_select_the_router():
    # Validates: docs/plans/requirement-trace.md, requirement #109
    problems = check_profiles({
        "profiles": {"default": {}},
        "matrix": {"sets": {"rag": "chat & embeddings"}},
    })
    assert problems == []


def test_matrix_sets_expand_roles_to_entry_ids():
    # Validates: docs/plans/requirement-trace.md, requirement #109
    from llama_packer.__main__ import _expand_matrix_sets

    got = _expand_matrix_sets(
        {"rag": "chat & embeddings", "solo": "chat"},
        {"c1": "chat", "c2": "chat", "e1": "embeddings"},
        {"c1": ["c1"], "c2": ["c2"], "e1": ["e1"]})
    assert got["rag"] == "(c1 | c2) & e1"
    assert got["solo"] == "(c1 | c2)"
    assert _expand_matrix_sets({}, {}, {}) == {}


def test_check_profiles_accepts_the_example():
    # Validates: docs/plans/requirement-trace.md, requirement #106
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "profiles.yaml.example").read_text())
    assert check_profiles(cfg) == []


def test_check_mode_exits_before_discovery(tmp_path):
    # Validates: docs/plans/requirement-trace.md, requirement #106
    from llama_packer.__main__ import main

    good = tmp_path / "good.yaml"
    good.write_text("profiles:\n  default:\n")
    main(["llama-packer", "--check", "--profiles", str(good)])

    bad = tmp_path / "bad.yaml"
    bad.write_text("profiles:\n  default:\n    temperatur: 0.5\n")
    with pytest.raises(SystemExit) as exc:
        main(["llama-packer", "--check", "--profiles", str(bad)])
    assert exc.value.code == 1


def test_retired_set_placeholders_error():
    # Validates: SPEC.md, 'VRAM Allocation': a set names roles; the old
    # __CHAT_VARS__/__COLOAD_VARS__ placeholders are retired.
    problems = check_profiles({
        "profiles": {"default": {}},
        "matrix": {"sets": {"rag": "__CHAT_VARS__ & emb",
                            "voice": "chat & __COLOAD_VARS__"}},
    })
    text = "\n".join(problems)
    assert text.count("are retired") == 2


def test_task_intent_is_accepted():
    # The parser reads intent; the validator must mirror it.
    problems = check_profiles({
        "profiles": {"default": {}},
        "tasks": {"chat": {"endpoint": "openai.chat.completions",
                           "intent": "converse"}},
    })
    assert problems == []


def test_retired_and_invalid_keys_are_reported_once():
    problems = check_profiles({
        "profiles": {"default": {}},
        "matrix": {"sets": "not-a-map", "embed_context": 4096},
        "routing": {"use": "turbo"},
    })
    text = "\n".join(problems)
    assert text.count("matrix.sets: must be a mapping") == 1
    assert text.count("routing: retired") == 1
    assert "unknown top-level key 'routing'" not in text
    assert "matrix.embed_context: retired" in text
    assert "routing.use" not in text


def test_task_device_is_validated():
    problems = check_profiles({
        "profiles": {"default": {}},
        "tasks": {"chat": {"endpoint": "openai.chat.completions",
                           "device": "gpu"}},
    })
    assert any("tasks.chat.device" in p for p in problems)
