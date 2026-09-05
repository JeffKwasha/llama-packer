# tests/test_main.py
"""CLI-level helpers: health-check timeout, path-macro substitution."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from llama_packer.__main__ import _apply_env_subst, _health_check_timeout
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
    models = [_model(tmp_path, 1000)]  # 1000 MiB at 500 MB/s -> 2.4s -> floor
    assert _health_check_timeout(models, _args()) == 120


def test_health_check_scales_with_model(tmp_path):
    models = [_model(tmp_path, 100000)]  # 100000/500*1.2 = 240s
    assert _health_check_timeout(models, _args()) == 240


def test_health_check_vllm_backend_raises_floor(tmp_path):
    models = [_model(tmp_path, 1000), _model(tmp_path, 10, backend="vllm")]
    assert _health_check_timeout(models, _args()) == 300


def test_health_check_explicit_speed_and_env(tmp_path, monkeypatch):
    models = [_model(tmp_path, 100000)]
    assert _health_check_timeout(models, _args(drive_speed=1000)) == 120
    monkeypatch.setenv("GEN_CONFIG_DRIVE_SPEED", "1000")
    # 100000/1000*1.2 = 120 -> floor
    assert _health_check_timeout(models, _args(drive_speed=None)) == 120


def test_apply_env_subst_longest_prefix_wins():
    # Manual prefix map: compute_env_prefixes groups by mount, which is
    # environment-dependent; this test targets substitution ordering only.
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
    sub = make_subst({})
    config = {"models": {"m": {"cmd": "run /elsewhere/m.gguf"}}}
    out = _apply_env_subst(config, sub, [])
    assert out["models"]["m"]["cmd"] == "run /elsewhere/m.gguf"


def test_backend_args_validation(caplog):
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
    import logging
    from llama_packer.__main__ import _build_matrix_vars

    def m(role, tid):
        return SimpleNamespace(role=role, template_id=tid, stem=tid)

    models = [m("chat", "alpha"), m("chat", "beta"), m("embeddings", "emb-1"),
              m("rerank", "rnk-1")]
    # Only emitted entries become vars: alpha kept vision (has -text), beta
    # was auto-dropped (its main entry IS beta-text), ghost never emitted.
    entry_ids_by_stem = {"alpha": ["alpha", "alpha-text"], "beta": ["beta-text"]}
    vars_, coload_vars = _build_matrix_vars(
        models, m("embeddings", "emb-1"), m("rerank", "rnk-1"), [],
        entry_ids_by_stem, logging.getLogger("test"))
    assert vars_ == {"c1": "alpha", "c2": "alpha-text", "c3": "beta-text",
                     "emb": "emb-1", "rnk": "rnk-1"}
    assert coload_vars == []


def test_build_matrix_vars_includes_coloads():
    import logging
    from llama_packer.__main__ import _build_matrix_vars

    def m(role, tid):
        return SimpleNamespace(role=role, template_id=tid, stem=tid)

    models = [m("chat", "alpha"), m("s2t", "parakeet"), m("image", "flux"),
              m("image", "flux2")]
    vars_, coload_vars = _build_matrix_vars(
        models, None, None, ["parakeet", "flux", "flux2"],
        {"alpha": ["alpha"]}, logging.getLogger("test"))
    assert vars_["c1"] == "alpha"
    assert vars_["s2t"] == "parakeet"
    assert vars_["img"] == "flux"
    assert vars_["img2"] == "flux2"
    assert coload_vars == ["s2t", "img", "img2"]


def test_expand_matrix_sets_placeholders():
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
