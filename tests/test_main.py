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
                                 _health_check_timeout,
                                 _select_model,
                                 _selection_reason)
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


def test_build_matrix_vars_includes_text_variants():
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Auto-collect matrix vars.
    import logging
    from llama_packer.__main__ import _build_matrix_vars

    def m(role, tid):
        return SimpleNamespace(role=role, template_id=tid, stem=tid)

    models = [m("chat", "alpha"), m("chat", "beta"), m("embeddings", "emb-1"),
              m("rerank", "rnk-1")]
    # Only emitted entries become vars: alpha kept vision (has -text), beta
    # was auto-dropped (its main entry IS beta-text), ghost never emitted.
    entry_ids_by_stem = {"alpha": ["alpha", "alpha-text"],
                         "beta": ["beta-text"],
                         "emb-1": ["emb-1-4k"], "rnk-1": ["rnk-1-4k"]}
    vars_, coload_vars = _build_matrix_vars(
        models, m("embeddings", "emb-1"), m("rerank", "rnk-1"), {},
        [], entry_ids_by_stem, logging.getLogger("test"))
    assert vars_ == {"c1": "alpha", "c2": "alpha-text", "c3": "beta-text",
                     "emb": "emb-1-4k", "rnk": "rnk-1-4k"}
    assert coload_vars == []


def test_build_matrix_vars_includes_coloads():
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Auto-collect matrix vars.
    import logging
    from llama_packer.__main__ import _build_matrix_vars

    def m(role, tid):
        return SimpleNamespace(role=role, template_id=tid, stem=tid)

    models = [m("chat", "alpha"), m("s2t", "parakeet"), m("image", "flux"),
              m("image", "flux2")]
    # co-load vars resolve to emitted entry ids too
    ids = {"alpha": ["alpha-128k"], "parakeet": ["parakeet-32k"],
           "flux": ["flux-64k"], "flux2": ["flux2-64k"]}
    vars_, coload_vars = _build_matrix_vars(
        models, None, None, {}, ["parakeet", "flux", "flux2"],
        ids, logging.getLogger("test"))
    assert vars_["c1"] == "alpha-128k"
    assert vars_["s2t"] == "parakeet-32k"
    assert vars_["img"] == "flux-64k"
    assert vars_["img2"] == "flux2-64k"
    assert coload_vars == ["s2t", "img", "img2"]


def test_expand_matrix_sets_placeholders():
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Expand ``__CHAT_VARS__``/``__COLOAD_VARS__`` placeholders in set exprs. Placeholders expand to parenthesized OR-lists of VAR NAMES (not model.
    import logging
    from llama_packer.__main__ import _expand_matrix_sets

    log = logging.getLogger("test")
    out = _expand_matrix_sets(
        {"rag": "__CHAT_VARS__ & emb & rnk & __COLOAD_VARS__",
         "chat_only": "__CHAT_VARS__"},
        "(c1 | c2)", ["s2t", "img"], log)
    assert out["rag"] == "(c1 | c2) & emb & rnk & (s2t | img)"
    assert out["chat_only"] == "(c1 | c2)"


def test_expand_matrix_sets_no_coloads_drops_placeholder(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Expand ``__CHAT_VARS__``/``__COLOAD_VARS__`` placeholders in set exprs. Placeholders expand to parenthesized OR-lists of VAR NAMES (not model.
    import logging
    from llama_packer.__main__ import _expand_matrix_sets

    log = logging.getLogger("test")
    with caplog.at_level(logging.WARNING):
        out = _expand_matrix_sets(
            {"rag": "__CHAT_VARS__ & emb & rnk & __COLOAD_VARS__",
             "leading": "__COLOAD_VARS__ & __CHAT_VARS__"},
            "(c1)", [], log)
    assert out["rag"] == "(c1) & emb & rnk"
    assert out["leading"] == "(c1)"
    assert "__COLOAD_VARS__" in caplog.text


def _stub_model(stem, role):
    from types import SimpleNamespace
    return SimpleNamespace(stem=stem, role=role, frontmatter={},
                           name=stem, template_id=stem, vram_mb=100)


def test_detect_matrix_warns_when_rag_models_but_no_section(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Resolve the swap-matrix configuration before build_config. Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.
    import logging
    from types import SimpleNamespace

    from llama_packer.__main__ import _detect_matrix

    models = [_stub_model("emb1", "embeddings"), _stub_model("rnk1", "rerank"),
              _stub_model("c1", "chat")]
    args = SimpleNamespace(embed=None, rerank=None)
    with caplog.at_level(logging.WARNING):
        cfg, emb, rnk, cats, fixed = _detect_matrix({}, models, args,
                                                    logging.getLogger("test"))
    assert cfg is None and emb is None and rnk is None
    assert "matrix: disabled" in caplog.text
    assert "emb1" in caplog.text and "rnk1" in caplog.text


def test_detect_matrix_silent_without_rag_models(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Resolve the swap-matrix configuration before build_config. Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.
    import logging
    from types import SimpleNamespace

    from llama_packer.__main__ import _detect_matrix

    models = [_stub_model("c1", "chat")]
    args = SimpleNamespace(embed=None, rerank=None)
    with caplog.at_level(logging.WARNING):
        cfg, _, _, cats, fixed = _detect_matrix({}, models, args,
                                                logging.getLogger("test"))
    assert cfg is None
    assert "matrix: disabled" not in caplog.text


def test_detect_matrix_selects_when_section_present(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Resolve the swap-matrix configuration before build_config. Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.
    import logging
    from types import SimpleNamespace

    from llama_packer.__main__ import _detect_matrix

    models = [_stub_model("emb1", "embeddings"), _stub_model("rnk1", "rerank"),
              _stub_model("c1", "chat")]
    args = SimpleNamespace(embed=None, rerank=None)
    matrix_cfg = {"sets": {"rag": "__CHAT_VARS__ & emb & rnk"}}
    with caplog.at_level(logging.WARNING):
        cfg, emb, rnk, cats, fixed = _detect_matrix({"matrix": matrix_cfg},
                                                    models, args,
                                                    logging.getLogger("test"))
    assert cfg is matrix_cfg
    assert emb.stem == "emb1" and rnk.stem == "rnk1"
    assert cats == {"emb": emb, "rnk": rnk}
    assert fixed == []
    assert "matrix: disabled" not in caplog.text


def test_detect_matrix_custom_categories(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Resolve the swap-matrix configuration before build_config. Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.
    import logging
    from types import SimpleNamespace

    from llama_packer.__main__ import _detect_matrix

    models = [_stub_model("emb1", "embeddings"), _stub_model("rnk1", "rerank"),
              _stub_model("tts1", "t2s"), _stub_model("stt1", "s2t"),
              _stub_model("c1", "chat")]
    args = SimpleNamespace(embed=None, rerank=None)
    matrix_cfg = {
        "categories": {"emb": {"role": "embeddings"},
                       "rnk": {"role": "rerank"},
                       "tts": {"role": "t2s"},
                       "stt": {"role": "s2t"}},
        "evict_costs": {"emb": 100, "tts": 50},
        "sets": {"voice": "__CHAT_VARS__ & (tts | stt)"},
    }
    with caplog.at_level(logging.WARNING):
        cfg, emb, rnk, cats, fixed = _detect_matrix(
            {"matrix": matrix_cfg}, models, args, logging.getLogger("test"))
    assert cfg is matrix_cfg
    assert set(cats) == {"emb", "rnk", "tts", "stt"}
    assert cats["tts"].stem == "tts1" and cats["stt"].stem == "stt1"
    # Non-RAG categories are fixed-overhead residents, in declaration order.
    assert [n for n, _ in fixed] == ["tts", "stt"]
    assert "evict_costs key" not in caplog.text


def test_detect_matrix_warns_unknown_evict_cost(caplog):
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Resolve the swap-matrix configuration before build_config. Returns ``(matrix_cfg, embed_model, rerank_model, categories, fixed)``.
    import logging
    from types import SimpleNamespace

    from llama_packer.__main__ import _detect_matrix

    models = [_stub_model("emb1", "embeddings"), _stub_model("rnk1", "rerank")]
    args = SimpleNamespace(embed=None, rerank=None)
    with caplog.at_level(logging.WARNING):
        _detect_matrix({"matrix": {"evict_costs": {"nope": 5}}}, models, args,
                       logging.getLogger("test"))
    assert "evict_costs key" in caplog.text


def test_build_matrix_vars_includes_categories():
    # Validates: docs/plans/matrix-categories.md, 'Design sketch': Auto-collect matrix vars.
    import logging

    from llama_packer.__main__ import _build_matrix_vars

    def m(role, tid):
        return SimpleNamespace(role=role, template_id=tid, stem=tid)

    emb, rnk, tts = (m("embeddings", "emb-1"), m("rerank", "rnk-1"),
                     m("t2s", "tts-1"))
    models = [m("chat", "alpha"), emb, rnk, tts]
    cats = {"emb": emb, "rnk": rnk, "tts": tts}
    # Category vars resolve through the emitted entry ids, which carry the
    # served context -- template_id is not an entry name.
    ids = {"alpha": ["alpha-128k"], "emb-1": ["emb-1-4k"],
           "rnk-1": ["rnk-1-4k"], "tts-1": ["tts-1-32k"]}
    vars_, coload_vars = _build_matrix_vars(
        models, emb, rnk, cats, [], ids, logging.getLogger("test"))
    assert vars_ == {"c1": "alpha-128k", "emb": "emb-1-4k",
                     "rnk": "rnk-1-4k", "tts": "tts-1-32k"}
    assert coload_vars == []


def test_parse_args_idle_unload():
    from llama_packer.__main__ import parse_args

    args = parse_args(["prog", "--idle-unload", "600"])
    assert args.idle_unload == 600
    # Off by default: no globalTTL key is emitted.
    args = parse_args(["prog"])
    assert args.idle_unload is None


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
    # The standard: binaries live beside the config, so no profiles.yaml
    # entry, no env var, and no PATH edit is needed. llama.cpp has always
    # worked this way (a cwd scan for llama-b[0-9]*); every other backend
    # used to require its own *_BIN_DIR to match.
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may live under *base_dir*, most specific first.
    monkeypatch.delenv("AUDIOCPP_BIN_DIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "audiocpp_server")
    got = resolve_host_binary(None, None, "AUDIOCPP_BIN_DIR", "audiocpp_server",
                              run_dir=tmp_path)
    assert got == str(tmp_path / "audiocpp_server")
    assert Path(got).is_file()


def test_host_binary_precedence_cli_then_profiles_then_env_then_run_dir(
        tmp_path, monkeypatch):
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may live under *base_dir*, most specific first.
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
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may live under *base_dir*, most specific first.
    monkeypatch.setenv("WHISPER_BIN_DIR", str(tmp_path))
    monkeypatch.setattr(shutil, "which", lambda n: None)
    _bin(tmp_path / "whisper-server")
    assert resolve_host_binary(None, None, "WHISPER_BIN_DIR", "whisper-server",
                               run_dir=Path("/nonexistent")) == str(
        tmp_path / "whisper-server")


def test_host_binary_path_used_when_nothing_resolves(tmp_path, monkeypatch):
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may live under *base_dir*, most specific first.
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
    # Validates: SPEC.md, 'llama.cpp Build Selection': Where an engine binary may live under *base_dir*, most specific first.
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


# ── resident selection (`device: cpu` wins the role) ─────────────────────


def _role_model(stem, mb, role="embeddings", **fm):
    m = SimpleNamespace(stem=stem, role=role, vram_mb=mb,
                        template_id=stem,
                        frontmatter={"name": stem, **fm})
    m.on_cpu = str(m.frontmatter.get("device", "")).strip().lower() == "cpu"
    return m


def test_select_model_prefers_cpu_candidate():
    # Validates: docs/plans/cpu-residency-budget.md #17 -- a device: cpu
    # candidate wins the resident role over a smaller device model.
    gpu_small = _role_model("voyage", 355)
    cpu_big = _role_model("jina", 472, device="cpu")
    assert _select_model([gpu_small, cpu_big], "embeddings", None, None) is cpu_big


def test_select_model_smallest_among_cpu_candidates():
    # Validates: docs/plans/cpu-residency-budget.md #17 -- among CPU
    # candidates the smallest footprint still wins.
    cpu_big = _role_model("qwen3-embed", 640, device="cpu")
    cpu_small = _role_model("jina", 472, device="cpu")
    assert _select_model([cpu_big, cpu_small], "embeddings", None,
                         None) is cpu_small


def test_select_model_no_cpu_keeps_size_rule():
    # Validates: docs/plans/cpu-residency-budget.md #17 -- with no CPU
    # candidate the selection is unchanged (smallest footprint).
    big = _role_model("big", 900)
    small = _role_model("small", 355)
    assert _select_model([big, small], "embeddings", None, None) is small


def test_select_model_selector_overrides_cpu_preference():
    # Validates: docs/plans/cpu-residency-budget.md #17 -- an explicit
    # selector overrides the CPU preference.
    cpu = _role_model("jina", 472, device="cpu")
    gpu = _role_model("voyage", 355)
    assert _select_model([cpu, gpu], "embeddings", "voyage", None) is gpu


def test_selection_reason_names_the_arm():
    # Validates: docs/plans/cpu-residency-budget.md #17 -- the resident pick
    # logs why it won (explicit CPU placement vs smallest footprint).
    cpu = _role_model("jina", 472, device="cpu")
    gpu = _role_model("voyage", 355)
    assert _selection_reason(cpu) == "device: cpu — explicit placement"
    assert _selection_reason(gpu) == "smallest footprint 355 MiB"


# ── role-wide placement + CPU-helper pinning ─────────────────────────────


def test_device_roles_place_every_role_model(make_model, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #19 -- device_roles names
    # roles whose every model is CPU-placed, logged with the count.
    import logging
    from llama_packer.__main__ import apply_device_roles
    a = make_model("embed-a", role="embeddings")
    b = make_model("embed-b", role="embeddings")
    chat = make_model("chat", role="chat")
    with caplog.at_level(logging.INFO):
        apply_device_roles([a, b, chat], ["embeddings"], logging.getLogger("t"))
    assert a.on_cpu and b.on_cpu
    assert not chat.on_cpu
    assert any("device_roles: embeddings" in r.message for r in caplog.records)


def test_device_roles_absent_and_unmatched(make_model, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #19 -- absent key changes
    # nothing; a role matching no model warns.
    import logging
    from llama_packer.__main__ import apply_device_roles
    a = make_model("chat", role="chat")
    apply_device_roles([a], None, logging.getLogger("t"))
    assert not a.on_cpu
    with caplog.at_level(logging.WARNING):
        apply_device_roles([a], {"s2t": "cpu"}, logging.getLogger("t"))
    assert any("matched no model" in r.message for r in caplog.records)


def test_device_roles_rejects_unknown_role(make_model, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #19 -- unknown role names
    # warn and are ignored (placement stays an explicit, known-role act).
    import logging
    from llama_packer.__main__ import apply_device_roles
    a = make_model("chat", role="chat")
    with caplog.at_level(logging.WARNING):
        apply_device_roles([a], {"everything": "cpu"}, logging.getLogger("t"))
    assert not a.on_cpu
    assert any("not a placeable role" in r.message for r in caplog.records)


def test_cpu_helpers_pinned_into_every_set():
    # Validates: docs/plans/cpu-residency-budget.md #20 -- CPU helper terms
    # join every set unless already a leaf.
    from llama_packer.__main__ import _expand_matrix_sets
    out = _expand_matrix_sets(
        {"rag": "__CHAT_VARS__ & emb & rnk",
         "talk": "__CHAT_VARS__ & tts"},
        "(c1 | c2)", ["tts"], None, pinned=["emb", "rnk", "qwen3-embedding-0.6b"])
    leaves = lambda e: re.findall(r"(?<![\w.-])(?:emb|rnk)(?![\w.-])", e)
    assert leaves(out["rag"]) == ["emb", "rnk"]
    assert out["rag"].endswith("& qwen3-embedding-0.6b")
    assert out["talk"].endswith("& emb & rnk & qwen3-embedding-0.6b")
