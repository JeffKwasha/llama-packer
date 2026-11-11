# tests/test_main.py
"""CLI-level helpers: health-check timeout, path-macro substitution."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from llama_packer.__main__ import (resolve_host_binary,
                                 _apply_env_subst,
                                 _health_check_timeout)
from llama_packer.utils import make_subst


def _args(**kw):
    return SimpleNamespace(drive_speed=kw.get("drive_speed", 500),
                           health_check_timeout=kw.get("health_check_timeout"))


def _model(tmp_path, size_mb=0, backend="llama-server"):
    gguf = tmp_path / f"m{size_mb}-{backend}.gguf"
    # Sparse file: correct stat().st_size without allocating the bytes.
    with gguf.open("wb") as f:
        f.seek(size_mb * 1024 * 1024)
        f.write(b"\0")
    return SimpleNamespace(gguf_path=gguf, backend=backend)


def test_health_check_floor_and_formula(tmp_path):
    # Validates: SPEC.md, 'Health-Check Timeout': Auto-calculated healthCheckTimeout when not set explicitly.
    models = [_model(tmp_path, 1000)]  # 1000 MiB at 500 MB/s -> 2.4s -> floor
    assert _health_check_timeout(models, _args()) == 120


def test_health_check_scales_with_model(tmp_path):
    # Validates: SPEC.md, 'Health-Check Timeout': Auto-calculated healthCheckTimeout when not set explicitly.
    models = [_model(tmp_path, 100000)]  # 100000/500*1.2 = 240s
    assert _health_check_timeout(models, _args()) == 240


def test_health_check_vllm_backend_raises_floor(tmp_path):
    # Validates: SPEC.md, 'Health-Check Timeout': Auto-calculated healthCheckTimeout when not set explicitly.
    models = [_model(tmp_path, 1000), _model(tmp_path, 10, backend="vllm")]
    assert _health_check_timeout(models, _args()) == 300


def test_health_check_vllm_size_floor(tmp_path):
    # vLLM loads at an assumed 100 MB/s: floor = largest vLLM model size / 100.
    # 60000 MiB -> 600 s (a ~60 GB NVFP4 model loads in ~10 min on DGX Spark);
    # the 1.2 * size / drive_speed term (144 s here) must not undercut it.
    # Validates: SPEC.md, 'Health-Check Timeout': Auto-calculated healthCheckTimeout when not set explicitly.
    models = [_model(tmp_path, 60000, backend="vllm")]
    assert _health_check_timeout(models, _args()) == 600
    # Non-vLLM models are unaffected by the vLLM size floor.
    models = [_model(tmp_path, 60000)]
    assert _health_check_timeout(models, _args()) == 144


def test_health_check_explicit_speed_and_env(tmp_path, monkeypatch):
    # Validates: SPEC.md, 'Health-Check Timeout': Auto-calculated healthCheckTimeout when not set explicitly.
    models = [_model(tmp_path, 100000)]
    assert _health_check_timeout(models, _args(drive_speed=1000)) == 120
    monkeypatch.setenv("GEN_CONFIG_DRIVE_SPEED", "1000")
    # 100000/1000*1.2 = 120 -> floor
    assert _health_check_timeout(models, _args(drive_speed=None)) == 120


def test_apply_env_subst_longest_prefix_wins():
    # Manual prefix map: compute_env_prefixes groups by mount, which is
    # environment-dependent; this test targets substitution ordering only.
    # Validates: SPEC.md, 'Path Macros (`macros:` block and `config.env`)': Replace each raw emitted path in the generated cmds with its ${VAR} macro form.
    sub = make_subp = make_subst({"/opt/bin": "LLAMA_DIR",
                                  "/data/models": "MODELS_DIR"})
    config = {"models": {
        "m": {"cmd": "/opt/bin/llama-server -m /data/models/a.gguf"},
    }}
    out = _apply_env_subst(config, sub,
                           ["/opt/bin/llama-server", "/data/models/a.gguf"])
    cmd = out["models"]["m"]["cmd"]
    assert cmd == "${LLAMA_DIR}/llama-server -m ${MODELS_DIR}/a.gguf"


def test_apply_env_subst_leaves_unmatched_paths_alone():
    # Validates: SPEC.md, 'Path Macros (`macros:` block and `config.env`)': Replace each raw emitted path in the generated cmds with its ${VAR} macro form.
    sub = make_subst({})
    config = {"models": {"m": {"cmd": "run /elsewhere/m.gguf"}}}
    out = _apply_env_subst(config, sub, [])
    assert out["models"]["m"]["cmd"] == "run /elsewhere/m.gguf"


def test_backend_args_validation(caplog):
    # Validates: SPEC.md, 'Global backend args': Validate and return a backend section's free-form ``args`` string. Performance/tuning flags shared by every command that backend renders.
    import logging
    from llama_packer.__main__ import backend_args

    # Missing/empty section or key -> empty string, never None.
    assert backend_args(None, "llama_server") == ""
    assert backend_args({}, "llama_server") == ""
    assert backend_args({"args": None}, "llama_server") == ""
    assert backend_args({"args": "   "}, "llama_server") == ""
    # Valid free-form flags pass through stripped.
    assert backend_args({"args": " --flash-attn on -b 512 "},
                        "llama_server") == "--flash-attn on -b 512"
    # Bad quoting aborts the run (fail fast, not per-command).
    with pytest.raises(SystemExit):
        backend_args({"args": "--foo 'unclosed"}, "llama_server")
    # Non-string YAML values are rejected, not stringified into garbage flags.
    with pytest.raises(SystemExit):
        backend_args({"args": 4096}, "llama_server")
    with pytest.raises(SystemExit):
        backend_args({"args": ["--flash-attn", "on"]}, "llama_server")
    # A stringified container still passes (verbatim) but warns.
    with caplog.at_level(logging.WARNING):
        assert backend_args({"args": "['--flash-attn']"}, "sd") == "['--flash-attn']"
    assert "stringified container" in caplog.text


def test_detect_matrix_returns_none_without_a_section():
    from llama_packer.__main__ import _detect_matrix

    assert _detect_matrix({}) is None


def test_detect_matrix_returns_the_section():
    # Validates: docs/allocation.md, 'Routers': sets are priced by roles and
    # model ids; there is no RAG pair or category pick.
    from llama_packer.__main__ import _detect_matrix

    matrix_cfg = {"sets": {"rag": "chat & embeddings & rerank"}}
    assert _detect_matrix({"matrix": matrix_cfg}) == matrix_cfg


def test_parse_args_spare_pool_is_repeatable():
    from llama_packer.__main__ import parse_args

    args = parse_args(["prog", "--spare-pool", "gpu0=1G",
                       "--spare-pool", "ram=2G"])
    assert args.spare_pool == ["gpu0=1G", "ram=2G"]


def test_parse_args_spare_and_baseline():
    from llama_packer.__main__ import parse_args

    args = parse_args(["prog", "--spare", "20G", "--baseline", "10G"])
    assert args.spare == "20G"
    assert args.baseline == "10G"



# --- host binary resolution ------------------------------------------------


def _bin(path, name="engine", text="#!/bin/sh\n"):
    path.write_text(text)
    path.chmod(0o755)
    return path


def test_host_binary_found_in_run_directory(tmp_path, monkeypatch):
    # The standard: binaries sit beside the config, so no profiles.yaml
    # entry, no env var, and no PATH edit is needed. llama.cpp has always
    # worked this way (a cwd scan for llama-b[0-9]*); every other backend
    # used to require its own *_BIN_DIR to match.
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may be found under *base_dir*, most specific first.
    monkeypatch.delenv("AUDIOCPP_BIN_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "audiocpp_server")
    got = resolve_host_binary(None, None, "AUDIOCPP_BIN_DIR", "audiocpp_server",
                              run_dir=tmp_path)
    assert got == str(tmp_path / "audiocpp_server")
    assert Path(got).is_file()


def test_host_binary_precedence_cli_then_profiles_then_env_then_run_dir(
        tmp_path, monkeypatch):
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may be found under *base_dir*, most specific first.
    monkeypatch.setenv("AUDIOCPP_BIN_DIR", str(tmp_path / "from-env"))
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "audiocpp_server")
    _bin(tmp_path / "from-env")
    _bin(tmp_path / "from-profiles")
    args = ("AUDIOCPP_BIN_DIR", "audiocpp_server")
    # CLI wins over everything
    assert resolve_host_binary(str(tmp_path / "from-cli"), None, *args,
                               run_dir=tmp_path) == str(tmp_path / "from-cli")
    # then profiles.yaml
    assert resolve_host_binary(None, str(tmp_path / "from-profiles"), *args,
                               run_dir=tmp_path) == str(tmp_path / "from-profiles")
    # then the env var
    assert resolve_host_binary(None, None, *args,
                               run_dir=tmp_path) == str(tmp_path / "from-env")
    # and only then the run directory
    monkeypatch.delenv("AUDIOCPP_BIN_DIR")
    assert resolve_host_binary(None, None, *args,
                               run_dir=tmp_path) == str(tmp_path / "audiocpp_server")


def test_host_binary_env_var_may_name_a_directory(tmp_path, monkeypatch):
    # Mirrors LLAMA_BIN_DIR: a directory expands to <dir>/<name>.
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may be found under *base_dir*, most specific first.
    monkeypatch.setenv("WHISPER_BIN_DIR", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "whisper-server")
    assert resolve_host_binary(None, None, "WHISPER_BIN_DIR", "whisper-server",
                               run_dir=Path("/nonexistent")) == str(
        tmp_path / "whisper-server")


def test_host_binary_path_used_when_nothing_resolves(tmp_path, monkeypatch):
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may be found under *base_dir*, most specific first.
    monkeypatch.setattr(shutil, "which", lambda n: None)
    got = resolve_host_binary(None, None, "SD_BIN_DIR", "sd-server",
                              run_dir=tmp_path)
    assert got is None
    # ...unless a fallback was supplied (vLLM keeps its built-in default)
    assert resolve_host_binary(None, None, "SD_BIN_DIR", "sd-server",
                              fallback="sd", run_dir=tmp_path) == "sd"


def test_host_binary_stale_explicit_path_is_not_silently_replaced(
        tmp_path, monkeypatch):
    # A stale `bin:` must keep pointing there: is_available then reports
    # False and the operator gets a message naming what is wrong, rather
    # than a different binary being used without their knowledge.
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may be found under *base_dir*, most specific first.
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "sd-server")
    got = resolve_host_binary(None, str(tmp_path / "gone"), "SD_BIN_DIR",
                              "sd-server", run_dir=tmp_path)
    assert got == str(tmp_path / "gone")


def test_every_backend_uses_the_same_chain():
    # The point of the helper: one rule, not three copies of it.
    # Validates: SPEC.md, 'Host binary resolution': Resolve one host-only server binary.
    import inspect
    src = inspect.getsource(resolve_host_binary)
    # the source of the *body*, not the docstring: the docstring names the
    # tiers, so a search over the whole source would pass even with a tier
    # gone from the code.
    body = src.split('"""', 2)[-1]
    for tier in ("run_dir", "shutil.which", "env_var"):
        assert tier in body


# ── role-wide placement ──────────────────────────────────────────────────


def test_role_device_places_every_model_of_the_role(tmp_path):
    # Validates: SPEC.md, 'Roles': a role's device: places every model it
    # owns, indistinguishable from a sidecar device: cpu.
    from llama_packer import roles
    from llama_packer.discover import discover
    root = tmp_path / "models"
    for sub, stem in (("s2t", "asr"), ("t2s", "tts")):
        (root / sub).mkdir(parents=True)
        (root / sub / f"{stem}.gguf").write_bytes(b"x")
    try:
        roles.configure({"roles": {
            "s2t": {"dir": "s2t", "kind": "s2t", "device": "cpu"},
            "t2s": {"dir": "t2s", "kind": "t2s", "device": 1},
        }})
        models = discover(root, generate_stubs=False)
    finally:
        roles.configure(None)
    by_stem = {m.stem: m for m in models}
    assert by_stem["asr"].on_cpu
    assert by_stem["tts"].device == 1


def test_role_device_does_not_override_a_sidecar(tmp_path):
    # Validates: SPEC.md, 'Roles': the sidecar is the more specific source.
    from llama_packer import roles
    from llama_packer.discover import discover
    root = tmp_path / "models"
    (root / "s2t").mkdir(parents=True)
    (root / "s2t" / "m.gguf").write_bytes(b"x")
    (root / "s2t" / "m.md").write_text("---\nrole: s2t\ndevice: 1\n---\n")
    try:
        roles.configure({"roles": {
            "s2t": {"dir": "s2t", "kind": "s2t", "device": "cpu"}}})
        (model,) = discover(root, generate_stubs=False)
    finally:
        roles.configure(None)
    assert not model.on_cpu


def test_matrix_sets_validation():
    from llama_packer.__main__ import validate_matrix_sets
    assert validate_matrix_sets({}) is None
    assert "must be a mapping" in validate_matrix_sets(
        {"matrix": {"sets": ["rag"]}})
    assert "__CHAT_VARS__" in validate_matrix_sets(
        {"matrix": {"sets": {"rag": "__CHAT_VARS__ & embeddings"}}})
    assert validate_matrix_sets(
        {"matrix": {"sets": {"rag": "chat & embeddings"}}}) is None


def test_expand_matrix_sets_replaces_only_role_leaves():
    from llama_packer.__main__ import _expand_matrix_sets
    out = _expand_matrix_sets(
        {"base": "chat", "rag": "+base & jina-embeddings-v3"},
        {"jina-embeddings-v3": "embeddings", "qwen": "chat"},
        {"jina-embeddings-v3": ["jina-embeddings-v3"],
         "qwen": ["qwen", "qwen.aux"]})
    assert out["rag"] == "(qwen | qwen.aux) & jina-embeddings-v3"


def test_expand_matrix_sets_rejects_an_unknown_literal():
    import pytest
    from llama_packer.__main__ import _expand_matrix_sets
    with pytest.raises(ValueError, match="not a role"):
        _expand_matrix_sets({"s": "chat & nope"}, {"qwen": "chat"},
                            {"qwen": ["qwen"]})


def test_expand_matrix_sets_rejects_orphans():
    import pytest
    from llama_packer.__main__ import _expand_matrix_sets
    with pytest.raises(ValueError, match="cannot be priced"):
        _expand_matrix_sets({"s": "chat & +auto:orphans"}, {"qwen": "chat"},
                            {"qwen": ["qwen"]})


def test_parse_device_auto():
    from llama_packer.utils import parse_device
    assert parse_device("auto") == (None, None)
    assert parse_device("cpu") == ("cpu", None)
    assert parse_device("gpu")[1] is not None


def test_empty_role_is_pruned_from_a_set(caplog):
    import logging
    from llama_packer.__main__ import _expand_matrix_sets
    with caplog.at_level(logging.WARNING):
        out = _expand_matrix_sets(
            {"voice": "chat & t2s", "talk": "chat & (t2s | s2t)"},
            {"c": "chat"}, {"c": ["c"]})
    assert out == {"voice": "c", "talk": "c"}
    assert "has no entries" in caplog.text
