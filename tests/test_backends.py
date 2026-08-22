# tests/test_backends.py
"""Backend support matrix and command composition."""

from __future__ import annotations

import logging
from pathlib import Path

from llama_packer.backends import (
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
    SETTING_KEYS,
    VLLM_BACKENDS,
    BaseBackend,
    get_backend,
)
from llama_packer.backends.llama_server import LlamaServerBackend
from llama_packer.backends.vllm import VllmDockerBackend, VllmHostBackend


def _tvars():
    return {
        "llama_bin": "/opt/llama-server",
        "vllm_bin": "vllm",
        "vllm_image": "vllm/vllm-openai:latest",
        "gpu_mem_util": "0.9",
        "container_port": 8000,
        "docker_args": "--runtime=nvidia --gpus all --shm-size=16g",
        "models_dir": "/models",
    }


def test_registry_names():
    for name in ("llama-server", "vllm", "vllm-docker"):
        assert get_backend(name).name == name


def test_registry_unknown_raises():
    import pytest
    with pytest.raises(KeyError):
        get_backend("nope")


def test_llama_server_roles_and_formats():
    b = LlamaServerBackend()
    assert b.formats == {".gguf"}
    assert b.roles == {"chat", "embeddings", "rerank"}


def test_supports_role_and_format(make_model):
    b = LlamaServerBackend()
    assert b.unsupported_reason(make_model("m")) is None
    assert b.unsupported_reason(make_model("e", role="embeddings")) is None
    assert b.unsupported_reason(make_model("r", role="rerank")) is None


def test_llama_server_rejects_safetensors(make_model):
    b = LlamaServerBackend()
    # A model with no local .gguf and no hf_repo cannot be served by llama-server.
    m = make_model("x", hf_repo="org/some-safetensors-model")
    # simulate safetensors: gguf_path is a .safetensors file
    m.gguf_path = Path("/models/x.safetensors")
    assert "safetensors" in b.unsupported_reason(m)


def test_vllm_requires_safetensors_or_hf_repo(make_model):
    b = VllmHostBackend()
    # local .gguf only -> unsupported by vllm
    assert b.unsupported_reason(make_model("m")) is not None
    # hf_repo -> supported
    assert b.unsupported_reason(make_model("v", hf_repo="org/model")) is None


def test_vllm_rejects_embeddings_role(make_model):
    b = VllmHostBackend()
    assert b.unsupported_reason(make_model("e", role="embeddings")) is not None


def test_llama_server_mtp_args(make_model):
    m = make_model("m", mtp=True, mtp_draft_n_max=4)
    b = LlamaServerBackend()
    args, meta = b._mtp_args(m)
    assert "--spec-type" in args and "--spec-draft-n-max" in args
    assert meta == {"mtp_enabled": True, "mtp_draft_max": 4}


def test_llama_server_cmd_has_mmap_layers_and_cache(make_model):
    m = make_model("m")
    b = LlamaServerBackend()
    cmd, meta = b.build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert cmd.startswith("/opt/llama-server --port ${PORT} -m ")
    assert "--n-gpu-layers 999" in cmd
    assert "--cache-type-k q8_0 --cache-type-v q8_0" in cmd
    assert meta == {"mtp_enabled": False}


def test_vllm_host_cmd(make_model):
    m = make_model("v", hf_repo="org/model")
    cmd, meta = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert cmd.startswith("vllm serve")
    assert "--model org/model" in cmd
    assert "--max-model-len 65536" in cmd
    assert meta == {"mtp_enabled": False}


def test_vllm_baked_in_mtp_emits_speculative_config(make_model):
    m = make_model("v", hf_repo="org/model", mtp=True)
    cmd, meta = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert '--speculative-config {"method":"mtp","num_speculative_tokens":2}' in cmd
    # Same default depth as the llama-server path — one config, one meaning.
    assert meta == {"mtp_enabled": True, "mtp_draft_max": 2}


def test_vllm_mtp_depth_override(make_model):
    m = make_model("v", hf_repo="org/model", mtp=True, **{"mtp_draft_n_max": 3})
    cmd, meta = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert '"num_speculative_tokens":3' in cmd
    assert meta["mtp_draft_max"] == 3


def test_vllm_explicit_speculative_config_wins(make_model):
    cfg = {"method": "eagle3", "model": "org/eagle-head",
           "num_speculative_tokens": 4}
    m = make_model("v", hf_repo="org/model", mtp=True,
                   **{"speculative_config": cfg})
    cmd, meta = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--speculative-config" in cmd
    # JSON emitted verbatim (key order preserved), not the derived mtp config
    assert '"method":"eagle3"' in cmd and '"num_speculative_tokens":4' in cmd
    assert "mtp" not in cmd.split("--speculative-config")[1].split()
    assert meta == {"mtp_enabled": True, "mtp_draft_max": 4}


def test_vllm_gguf_speculative_companion_warned_and_skipped(make_model, caplog):
    m = make_model("v", hf_repo="org/model", speculative="v.mtp.gguf",
                   backend="vllm")
    with caplog.at_level(logging.WARNING):
        cmd, meta = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--speculative-config" not in cmd
    assert meta == {"mtp_enabled": False}
    assert any("cannot be loaded by vLLM" in r.message for r in caplog.records)


def test_vllm_docker_mtp_flag(make_model):
    m = make_model("v", hf_repo="org/model", mtp=True)
    cmd, meta = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--speculative-config" in cmd
    assert meta["mtp_enabled"] is True


def test_vllm_docker_cmd_and_mounts(make_model, tmp_path):
    tpl = tmp_path / "qwen.jinja"
    tpl.write_text("x")
    m = make_model("v", hf_repo="org/model")
    m._resolved_chat_template = tpl.resolve()
    cmd, meta = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert cmd.startswith("docker run")
    assert "--chat-template" in cmd
    assert "/models" in cmd  # models_dir bind mount


def test_warn_unhandled(caplog):
    import logging
    b = VllmHostBackend()
    with caplog.at_level(logging.WARNING):
        b.warn_unhandled({"loras", "cli_args", "chat_template"})
    # vllm handles cli_args/chat_template/hf_repo, not loras -> warning
    assert any("loras" in r.message for r in caplog.records)


def test_setting_keys_partition():
    # framework-consumed and metadata-only keys are disjoint from each other
    # and from what a backend would render.
    assert FRAMEWORK_CONSUMED == {"backend"}
    assert METADATA_ONLY == {"chat_template_kwargs"}
    assert VLLM_BACKENDS == {"vllm", "vllm-docker"}
    assert "backend" in SETTING_KEYS and "chat_template" in SETTING_KEYS


def test_basebackend_is_abstract():
    import pytest
    with pytest.raises(TypeError):
        BaseBackend()  # type: ignore[abstract]


def test_filter_supported_skips_bad_format(make_model):
    from llama_packer.writer import _filter_supported

    # a local .gguf under vLLM is unsupported -> skipped
    vllm_gguf = make_model("v", backend="vllm")
    llama = make_model("l")
    out = _filter_supported([vllm_gguf, llama])
    assert [m.stem for m in out] == ["l"]


def test_filter_supported_skips_flagged_override(make_model):
    from llama_packer.writer import _filter_supported

    m = make_model("q", backend="vllm", hf_repo="org/model")
    m._override_error = "unknown backend"  # simulate apply_overrides flag
    assert _filter_supported([m]) == []


def test_filter_supported_warn_unhandled(make_model, caplog):
    import logging
    from llama_packer.writer import _filter_supported

    m = make_model("v", backend="vllm", hf_repo="org/model", loras=["x.gguf"])
    with caplog.at_level(logging.WARNING):
        _filter_supported([m])
    assert any("loras" in r.message for r in caplog.records)


# ── backend inference (format + availability) ──────────────────────────────


def test_infer_backend_gguf_prefers_llama_server(make_model):
    from llama_packer.backends import infer_backend
    assert infer_backend(make_model("g"), {"llama_bin": "/opt/llama-server"}) == "llama-server"


def test_infer_backend_safetensors_prefers_docker_then_host(make_model):
    from llama_packer.backends import infer_backend
    m = make_model("s", hf_repo="org/model")
    m.gguf_path = Path("/models/s.safetensors")
    # both available -> docker wins (registration preference order)
    assert infer_backend(m, {"vllm_image": "img", "vllm_bin": "vllm"}) == "vllm-docker"
    # only host binary available
    assert infer_backend(m, {"vllm_bin": "vllm"}) == "vllm"
    # neither configured -> no backend
    assert infer_backend(m, {}) is None


def test_infer_backend_hf_repo_only(make_model):
    from llama_packer.backends import infer_backend
    m = make_model("h", hf_repo="org/model")
    m.gguf_path = None  # no local file
    assert infer_backend(m, {"vllm_image": "img"}) == "vllm-docker"


def test_docker_cmd_uses_per_model_image(make_model):
    m = make_model("d", hf_repo="org/model", vllm_image="custom/img:v2")
    cmd, _ = VllmDockerBackend().build_cmd(m, 8192, 1, "q8_0", _tvars())
    assert "custom/img:v2" in cmd
    assert "vllm/vllm-openai:latest" not in cmd


def test_docker_cmd_in_tree_template_maps_under_models(make_model, tmp_path):
    tpl = tmp_path / "qwen.jinja"
    tpl.write_text("x")
    tvars = {**_tvars(), "models_dir": str(tmp_path)}
    m = make_model("d", hf_repo="org/model")
    m._resolved_chat_template = tpl.resolve()
    cmd, _ = VllmDockerBackend().build_cmd(m, 8192, 1, "q8_0", tvars)
    assert "--chat-template /models/qwen.jinja" in cmd


# ── reasoning flags + duplicate-free command composition ───────────────────


def test_llama_server_reasoning_format(make_model):
    m = make_model("m", **{"reasoning-format": "deepseek"})
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert "--reasoning-format deepseek" in cmd


def test_llama_server_reasoning_preserve(make_model):
    m = make_model("m", **{"reasoning-preserve": True})
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert "--reasoning-preserve" in cmd


def test_llama_server_reasoning_flags_chat_only(make_model):
    # embeddings/rerank never get reasoning flags (gated to the chat role).
    m = make_model("e", role="embeddings", **{"reasoning-format": "deepseek"})
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert "--reasoning-format" not in cmd


def test_cli_args_duplicate_flag_collapses(make_model):
    # A flag set both structurally and in cli_args is emitted exactly once
    # (the flag dict refuses duplicate keys); cli_args wins on value.
    m = make_model("m", cli_args="--reasoning-format none",
                   **{"reasoning-format": "deepseek"})
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert cmd.count("--reasoning-format") == 1
    assert "--reasoning-format none" in cmd
    assert "deepseek" not in cmd


def test_llama_server_multiple_loras_comma_joined(make_model):
    m = make_model("m")
    m._resolved_loras = [Path("/a/x.gguf"), Path("/a/y.gguf")]
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert cmd.count("--lora") == 1
    assert "--lora /a/x.gguf,/a/y.gguf" in cmd


def test_vllm_warns_on_reasoning_format(make_model, caplog):
    import logging
    m = make_model("v", hf_repo="org/model", **{"reasoning-format": "deepseek"})
    with caplog.at_level(logging.WARNING):
        VllmHostBackend().warn_unhandled(
            {k for k in SETTING_KEYS if k in m.frontmatter} - FRAMEWORK_CONSUMED - METADATA_ONLY
        )
    assert any("reasoning-format" in r.message for r in caplog.records)
