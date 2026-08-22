# tests/test_overrides.py
"""Pattern-scoped override rules: matching, last-wins, path resolution."""

from __future__ import annotations

import pytest

from llama_packer.overrides import apply_overrides, rule_matches, resolve_setting_paths


def test_rule_matches_regex_on_base_model(make_model):
    m = make_model("q", base_model="qwen3.8")
    assert rule_matches({"base_model": "qwen3\\.[68]"}, m)
    assert not rule_matches({"base_model": "llama"}, m)


def test_rule_matches_requires_all_fields(make_model):
    m = make_model("q", base_model="qwen3.6", parameters="35B-A3B")
    # both fields must match
    assert rule_matches({"base_model": "qwen3\\.6", "parameters": "35B"}, m)
    # one misses
    assert not rule_matches({"base_model": "qwen3\\.6", "parameters": "70B"}, m)


def test_rule_matches_invalid_regex(make_model, caplog):
    import logging
    m = make_model("q", base_model="qwen3.6")
    with caplog.at_level(logging.WARNING):
        assert not rule_matches({"base_model": "[unclosed"}, m)
    assert any("invalid regex" in r.message for r in caplog.records)


def test_apply_overrides_last_wins_per_key(make_model, tmp_path):
    profiles = {
        "overrides": [
            {"when": {"base_model": "qwen3\\.6"}, "backend": "vllm-docker",
             "hf_repo": "org/old"},
            {"when": {"name": "qwen3-27b"}, "hf_repo": "org/new"},
        ],
    }
    m = make_model("qwen3-27b", base_model="qwen3.6", name="qwen3-27b")
    apply_overrides([m], profiles, tmp_path)
    assert m.backend == "vllm-docker"          # first rule set it
    assert m.hf_repo == "org/new"             # last-wins per key (name rule)


def test_apply_overrides_sidecar_seed(make_model, tmp_path):
    # A sidecar-declared setting survives when no rule overrides it.
    profiles = {"overrides": [{"when": {"base_model": "qwen"}, "backend": "vllm"}]}
    m = make_model("q", base_model="qwen3.8", name="qwen3-8", cli_args="--seed 1")
    apply_overrides([m], profiles, tmp_path)
    assert m.backend == "vllm"
    assert m.frontmatter["cli_args"] == "--seed 1"


def test_apply_overrides_unknown_backend_skips(make_model, tmp_path, caplog):
    import logging
    profiles = {"overrides": [{"when": {"base_model": "qwen"}, "backend": "bogus"}]}
    m = make_model("q", base_model="qwen3.8", name="qwen3-8")
    with caplog.at_level(logging.ERROR):
        apply_overrides([m], profiles, tmp_path)
    assert getattr(m, "_override_error", None) is not None
    assert any("unknown backend" in r.message for r in caplog.records)


def test_rule_matches_when_true_matches_all(make_model):
    assert rule_matches(True, make_model("a"))
    assert rule_matches(True, make_model("b", base_model="whatever"))


def test_missing_when_stops_run(make_model, tmp_path):
    # A rule without 'when' is a config error that must abort the run —
    # silently ignoring it would compound the misconfiguration.
    for bad in (
        {"backend": "vllm"},                       # no when at all
        {"when": None, "backend": "vllm"},         # null when
        {"when": {}, "backend": "vllm"},           # empty when
    ):
        with pytest.raises(SystemExit, match="missing 'when'"):
            apply_overrides([make_model("q")], {"overrides": [bad]}, tmp_path)


def test_non_mapping_when_stops_run(make_model, tmp_path):
    with pytest.raises(SystemExit, match="'when' must be a mapping"):
        apply_overrides(
            [make_model("q")],
            {"overrides": [{"when": "qwen", "backend": "vllm"}]},
            tmp_path,
        )


def test_infer_backend_gguf_defaults_llama_server(make_model, tmp_path):
    m = make_model("g")
    apply_overrides([m], {}, tmp_path, avail={"llama_bin": "/opt/llama-server"})
    assert m.backend == "llama-server"
    assert m.frontmatter["backend"] == "llama-server"


def test_infer_backend_safetensors_prefers_docker(make_model, tmp_path):
    m = make_model("s", hf_repo="org/model")
    m.gguf_path = m.gguf_path.with_suffix(".safetensors")
    apply_overrides([m], {}, tmp_path, avail={"vllm_image": "img:1"})
    assert m.backend == "vllm-docker"


def test_infer_backend_safetensors_falls_back_to_host(make_model, tmp_path):
    m = make_model("s", hf_repo="org/model")
    m.gguf_path = m.gguf_path.with_suffix(".safetensors")
    apply_overrides([m], {}, tmp_path, avail={"vllm_bin": "vllm"})
    assert m.backend == "vllm"


def test_infer_backend_no_available_backend_skips(make_model, tmp_path, caplog):
    import logging
    m = make_model("s", hf_repo="org/model")
    m.gguf_path = m.gguf_path.with_suffix(".onnx")
    with caplog.at_level(logging.ERROR):
        apply_overrides([m], {}, tmp_path)
    assert getattr(m, "_override_error", None) is not None
    assert any("no available backend" in r.message for r in caplog.records)


def test_declared_backend_beats_inference(make_model, tmp_path):
    profiles = {"overrides": [{"when": True, "backend": "vllm",
                               "hf_repo": "org/m"}]}
    m = make_model("g")  # local gguf would infer llama-server
    apply_overrides([m], profiles, tmp_path)
    assert m.backend == "vllm"


def test_resolve_setting_paths_missing_file(make_model, tmp_path):
    m = make_model("q", chat_template="does-not-exist.jinja")
    errors = resolve_setting_paths(m, tmp_path)
    assert any("chat_template" in e for e in errors)
    assert getattr(m, "_resolved_chat_template", None) is None


def test_resolve_setting_paths_finds_files(make_model, tmp_path):
    tpl = tmp_path / "qwen.jinja"
    tpl.write_text("x")
    lora = tmp_path / "lora.gguf"
    lora.write_bytes(b"x")
    m = make_model("q", chat_template="qwen.jinja", loras=["lora.gguf"])
    errors = resolve_setting_paths(m, tmp_path)
    assert errors == []
    assert m._resolved_chat_template == tpl.resolve()
    assert m._resolved_loras == [lora.resolve()]
