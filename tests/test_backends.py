# tests/test_backends.py
"""Backend support matrix and command composition."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from llama_packer.profiles import Profiles
from llama_packer.writer import MatrixKnobs


def _tvars():
    return {
        "llama_bin": "/opt/llama-server",
        "vllm_bin": "vllm",
        "vllm_image": "vllm/vllm-openai:latest",
        "gpu_mem_util": "0.9",
        "container_port": 8000,
        "docker_args": "--runtime=nvidia --gpus all --shm-size=16g",
        "models_dir": "/models",
        "hf_cache": "",
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


def test_vllm_cache_type_maps_to_kv_cache_dtype(make_model):
    # q8_* -> fp8 (single configuration: same precision decision both backends)
    m = make_model("v", hf_repo="org/model")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--kv-cache-dtype fp8" in cmd
    # f16/bf16/f32 -> vLLM auto; no flag
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "f16", _tvars())
    assert "--kv-cache-dtype" not in cmd


def test_vllm_sub_byte_cache_type_warned_and_skipped(make_model, caplog):
    m = make_model("v", hf_repo="org/model")
    with caplog.at_level(logging.WARNING):
        cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q4_0", _tvars())
    assert "--kv-cache-dtype" not in cmd
    assert any("no --kv-cache-dtype equivalent" in r.message
               for r in caplog.records)


def test_vllm_nvfp4_flag_emitted_and_sized(make_model):
    # nvfp4 is a valid --kv-cache-dtype value; whether the serving build
    # supports it (experimental, hardware-gated) is the operator's call —
    # we translate and size, we don't police.

    from llama_packer.consts import _KV_CACHE_BYTES
    assert _KV_CACHE_BYTES["nvfp4"] == pytest.approx(0.5625)
    m = make_model("v", hf_repo="org/model")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "nvfp4", _tvars())
    assert "--kv-cache-dtype nvfp4" in cmd


def test_vllm_parallel_maps_to_max_num_seqs(make_model):
    m = make_model("v", hf_repo="org/model")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 4, "q8_0", _tvars())
    assert "--max-num-seqs 4" in cmd
    docker_cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 4, "q8_0", _tvars())
    assert "--max-num-seqs 4" in docker_cmd


def test_vllm_uncapped_omits_max_num_seqs(make_model):
    # parallel 0 = uncapped: vLLM owns admission (elastic queueing), so the
    # cap flag is omitted entirely.
    m = make_model("v", hf_repo="org/model", parallel=0)
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 0, "q8_0", _tvars())
    assert "--max-num-seqs" not in cmd
    assert "--max-model-len 65536" in cmd


def test_vllm_batch_flag_and_llama_only_strip(make_model):
    # --max-num-batched-tokens renders from the batch key; llama.cpp-only
    # flags leaking via sidecar cli_args are stripped with their values.
    m = make_model("v", hf_repo="org/model",
                   cli_args="--flash-attn on -b 2048 -ub 512 "
                            "--reasoning-format deepseek")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 4, "q8_0", _tvars(),
                                         batch=2048)
    assert "--max-num-batched-tokens 2048" in cmd
    assert "--max-num-seqs 4" in cmd
    toks = cmd.split()
    for gone in ("--flash-attn", "-b", "-ub", "--reasoning-format"):
        assert gone not in toks
    assert "deepseek" not in cmd
    assert " 512" not in cmd


def test_solve_matrix_uses_declared_embed_rerank_contexts(make_model,
                                                          monkeypatch):
    from llama_packer import writer

    chat = make_model("chat", context_length=32768)
    embed = make_model("e", role="embeddings", context_length=32768)
    rerank = make_model("r", role="rerank", context_length=16384)

    def fake_effective_static(fit_bin, cache_type="q8_0", **kw):
        return (1000.0, 0.05, 0.0, 50.0)

    def fake_fp(fit_bin, cache_type="q8_0", **kw):
        return SimpleNamespace(model_mib=1000.0, kv_per_token_mib=0.05,
                               slot_mib=0.0, compute_mib=50.0,
                               source="fit-params")

    for m in (chat, embed, rerank):
        m.vram.effective_static = fake_effective_static
        m.vram.fit_params_static = fake_fp

    calls: list[dict] = []
    monkeypatch.setattr(writer, "solve_matrix_ctx",
                        lambda **kw: calls.append(dict(kw)) or 8192)
    profiles = Profiles({"defaults": {}, "profiles": {"default": {}}})
    result = writer._solve_matrix_context(
        [chat], embed, rerank, "unused", 48000, None, profiles,
        knobs=MatrixKnobs(embed_context=32768, rerank_context=16384))
    assert result is not None
    assert calls[0]["embed_ctx"] == 32768   # configured, not hardcoded 8192
    assert calls[0]["rerank_ctx"] == 16384
    # The fake solver returns 8192 (< tools_min_ctx) so the squeeze fires,
    # but its gain is 0 (< ctx_gain_min) — configured contexts must survive.
    assert (result.embed_ctx, result.rerank_ctx) == (32768, 16384)
    assert result.squeeze is False


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
    assert FRAMEWORK_CONSUMED == {"backend", "hf_repo"}
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
    m._override_error = "unknown backend"  # simulate finalize() flag
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


# ── global backend args (profiles.yaml `<section>.args`) ─────────────────

def test_llama_server_global_args_chat(make_model):
    # Chat models get the fleet-wide performance flags as-is.
    tv = {**_tvars(), "llama_args": "--flash-attn on -b 512 -ub 512"}
    cmd, _ = LlamaServerBackend().build_cmd(make_model("m"), 32768, 1, "q8_0", tv)
    assert "--flash-attn on -b 512 -ub 512" in cmd


def test_llama_server_global_args_role_flags_win(make_model):
    # Global args render BEFORE the per-role slot, and the resolved batch
    # keys ride in that slot: named -b/-ub win over conflicting fleet args;
    # non-conflicting flags still apply.
    tv = {**_tvars(), "llama_args": "--flash-attn on -b 512 -ub 512"}
    cmd, _ = LlamaServerBackend().build_cmd(
        make_model("e", role="embeddings"), 32768, 1, "q8_0", tv,
        batch=4096, ubatch=512)
    assert cmd.count("-b ") == 1 and cmd.count("-ub ") == 1
    assert "-b 4096 -ub 512" in cmd
    assert "-b 512" not in cmd
    assert "--flash-attn on" in cmd


def test_llama_server_batch_ubatch_render_and_defaults(make_model):
    # The resolved batch keys render explicitly in the per-role slot:
    # they win over conflicting fleet args, lose to cli_args (warned).
    be = LlamaServerBackend()
    tv = {**_tvars(), "llama_args": "--flash-attn on -b 512 -ub 512"}
    cmd, _ = be.build_cmd(make_model("c"), 32768, 1, "q8_0", tv,
                          batch=2048, ubatch=512)
    assert "-b 2048 -ub 512" in cmd
    assert "-b 512" not in cmd and "-ub 512" in cmd
    assert "--flash-attn on" in cmd
    # Role defaults: embed/rerank keep their tuned depth when nothing set.
    assert be.default_batch_ubatch("embeddings") == (4096, 512)
    assert be.default_batch_ubatch("rerank") == (4096, 512)
    assert be.default_batch_ubatch("chat") == (2048, 512)
    # cli_args -b/-ub warn once (shadowed + invisible to measurement).
    cmd2, _ = be.build_cmd(make_model("w", cli_args="-b 4096"), 32768, 1,
                           "q8_0", tv, batch=2048, ubatch=512)
    assert "-b 4096 -ub 512" in cmd2  # named flags win per-flag


def test_llama_server_batch_ubatch_resolution_cascade(make_model):
    from llama_packer.profiles import Profiles
    from llama_packer.writer import measurement_args, resolve_batch_ubatch
    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "llama_server": {"args": "--flash-attn on",
                                          "batch": 2048, "ubatch": 1024}})
    m = make_model("c", role="chat")
    # fleet tier wins over the role default
    assert resolve_batch_ubatch(profiles, m) == (2048, 1024)
    # sidecar tier wins over the fleet tier
    m2 = make_model("c2", role="chat", batch=4096, ubatch=2048)
    assert resolve_batch_ubatch(profiles, m2) == (4096, 2048)
    # profile tier sits between sidecar and fleet
    m3 = make_model("c3", role="chat")
    assert resolve_batch_ubatch(profiles, m3, profile={"batch": 8192}) \
        == (8192, 1024)
    # measurement shape: global args + -ub only (batch has no VRAM effect)
    assert measurement_args(profiles, m, "--flash-attn on") == \
        "--flash-attn on -ub 1024"
    assert measurement_args(profiles, m2, "--flash-attn on") == \
        "--flash-attn on -ub 2048"


def test_llama_server_global_args_per_model_cli_args_win(make_model):
    # Per-model cli_args beat the global args per flag, emitted once.
    tv = {**_tvars(), "llama_args": "--flash-attn on -b 512"}
    m = make_model("m", cli_args="-b 2048")
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", tv)
    assert cmd.count("-b ") == 1
    assert "-b 2048" in cmd
    assert "--flash-attn on" in cmd


def test_llama_server_global_args_absent_no_leak(make_model):
    # No llama_args configured -> nothing changes.
    cmd, _ = LlamaServerBackend().build_cmd(make_model("m"), 32768, 1, "q8_0", _tvars())
    assert "--flash-attn" not in cmd


def test_vllm_global_args_host_and_docker(make_model):
    tv = {**_tvars(), "vllm_args": "--max-num-batched-tokens 512"}
    m = make_model("v", hf_repo="org/model")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert "--max-num-batched-tokens 512" in cmd
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert "--max-num-batched-tokens 512" in cmd


def test_vllm_global_args_per_model_cli_args_win(make_model):
    tv = {**_tvars(), "vllm_args": "--max-num-batched-tokens 512"}
    m = make_model("v", hf_repo="org/model", cli_args="--max-num-batched-tokens 1024")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert cmd.count("--max-num-batched-tokens") == 1
    assert "--max-num-batched-tokens 1024" in cmd


def test_whisper_server_global_args(make_model):
    tv = {"whisper_bin": "/opt/whisper-server", "whisper_args": "--flash-attn on"}
    m = make_model("w", role="s2t")
    m.gguf_path = Path("/models/s2t/ggml-large-v3.bin")
    cmd, _ = get_backend("whisper-server").build_cmd(m, 32768, 1, "q8_0", tv)
    assert "--flash-attn on" in cmd


def test_sd_server_global_args(make_model):
    tv = {"sd_bin": "/opt/sd-server", "sd_args": "--diffusion-fa"}
    m = make_model("i", role="image")
    m.gguf_path = Path("/models/img/flux-4b.gguf")
    cmd, _ = get_backend("sd-server").build_cmd(m, 4096, 1, "q8_0", tv)
    assert "--diffusion-fa" in cmd


def test_whisper_global_args_per_model_cli_args_win(make_model):
    # Parity with llama/vllm: per-model cli_args beat the global args per flag.
    tv = {"whisper_bin": "/opt/whisper-server", "whisper_args": "--flash-attn on"}
    m = make_model("w", role="s2t", cli_args="--flash-attn off")
    m.gguf_path = Path("/models/s2t/ggml-large-v3.bin")
    cmd, _ = get_backend("whisper-server").build_cmd(m, 32768, 1, "q8_0", tv)
    assert cmd.count("--flash-attn") == 1
    assert "--flash-attn off" in cmd


def test_sd_global_args_per_model_cli_args_win(make_model):
    tv = {"sd_bin": "/opt/sd-server", "sd_args": "--diffusion-fa --steps 20"}
    m = make_model("i", role="image", cli_args="--steps 30")
    m.gguf_path = Path("/models/img/flux-4b.gguf")
    cmd, _ = get_backend("sd-server").build_cmd(m, 4096, 1, "q8_0", tv)
    assert cmd.count("--steps") == 1
    assert "--steps 30" in cmd
    assert "--diffusion-fa" in cmd


# ── render_command precedence (builtin < global < role < cli) ─────────────

def test_render_command_precedence_order():
    # Later sources overwrite earlier ones per flag; each flag emitted once.
    from llama_packer.utils import render_command
    # Later sources overwrite earlier values per flag; each flag emitted once
    # (at its first-seen position).
    cmd = render_command(
        ["srv"], ["-c", "4096", "--parallel", "1"],
        global_args="-b 512 --flash-attn on",
        role_flags="-b 4096 -ub 4096",
        cli_args="-c 8192",
    )
    assert cmd == "srv -c 8192 --parallel 1 -b 4096 --flash-attn on -ub 4096"


def test_render_command_empty_sources():
    from llama_packer.utils import render_command
    assert render_command(["srv"], []) == "srv"
    assert render_command(["srv"], ["-x"], global_args="", role_flags="") == "srv -x"


def test_render_command_flag_order_preserved_within_source():
    from llama_packer.utils import render_command
    cmd = render_command(["srv"], ["--z-last", "1", "--a-first", "2"])
    assert cmd == "srv --z-last 1 --a-first 2"


def test_llama_server_multiple_loras_comma_joined(make_model):
    m = make_model("m")
    m._resolved_loras = [Path("/a/x.gguf"), Path("/a/y.gguf")]
    cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert cmd.count("--lora") == 1
    assert "--lora /a/x.gguf,/a/y.gguf" in cmd


# ── image token flags (vision sidecar declarations) ───────────────────────

def test_llama_server_image_token_flags(make_model, tmp_path):
    # Declared image token bounds on a vision model are emitted as CLI flags.
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                "capabilities": ["image"],
                                "image_min_tokens": 1024,
                                "image_max_tokens": 4096})
    cmd, _ = LlamaServerBackend().build_cmd(m.view_for(True), 32768, 1,
                                            "q8_0", _tvars())
    assert "--mmproj" in cmd
    assert "--image-min-tokens 1024" in cmd
    assert "--image-max-tokens 4096" in cmd


def test_llama_server_no_image_flags_without_declaration(make_model):
    # No sidecar declaration -> no flags (defaults come from the model).
    cmd, _ = LlamaServerBackend().build_cmd(make_model("m"), 32768, 1, "q8_0", _tvars())
    assert "--image-min-tokens" not in cmd
    assert "--image-max-tokens" not in cmd


def test_llama_server_image_tokens_without_mmproj_warned(make_model, caplog):
    m = make_model("m", **{"image_min_tokens": 1024})
    with caplog.at_level(logging.WARNING):
        cmd, _ = LlamaServerBackend().build_cmd(m, 32768, 1, "q8_0", _tvars())
    assert "--image-min-tokens" not in cmd
    assert any("no mmproj companion" in r.message for r in caplog.records)


def test_llama_server_image_tokens_static_arch_skipped(make_model, tmp_path,
                                                       monkeypatch, caplog):
    # Static-resolution vision archs (Gemma/SigLIP, ~256 tokens/image fixed)
    # ignore the flags — declared bounds are warned about and skipped.
    (tmp_path / "g-mmproj.gguf").write_bytes(b"mm")
    m = make_model("g", mmproj={"file": "g-mmproj.gguf",
                                "capabilities": ["image"],
                                "image_min_tokens": 1024})
    monkeypatch.setattr("llama_packer.utils.gguf_header_probe",
                        lambda p: ("gemma4", True))
    with caplog.at_level(logging.WARNING):
        cmd, _ = LlamaServerBackend().build_cmd(m.view_for(True), 32768, 1,
                                                "q8_0", _tvars())
    assert "--image-min-tokens" not in cmd
    assert any("static-resolution" in r.message for r in caplog.records)


def test_llama_server_image_tokens_text_variant_skipped(make_model, tmp_path):
    # The -text variant serves no vision: flags dropped without a warning.
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    m = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                "capabilities": ["image"],
                                "image_min_tokens": 1024})
    cmd, _ = LlamaServerBackend().build_cmd(m.view_for(False), 32768, 1,
                                            "q8_0", _tvars(),
                                            include_mmproj=False)
    assert "--image-min-tokens" not in cmd
    assert "--mmproj" not in cmd


def test_vllm_warns_on_reasoning_format(make_model, caplog):
    import logging
    m = make_model("v", hf_repo="org/model", **{"reasoning-format": "deepseek"})
    with caplog.at_level(logging.WARNING):
        VllmHostBackend().warn_unhandled(
            {k for k in SETTING_KEYS if k in m.frontmatter} - FRAMEWORK_CONSUMED - METADATA_ONLY
        )
    assert any("reasoning-format" in r.message for r in caplog.records)


def test_infer_backend_allowed_reorders_and_disables(make_model):
    from llama_packer.backends import infer_backend
    m = make_model("s", hf_repo="org/model")
    m.gguf_path = Path("/models/s.safetensors")
    avail = {"vllm_image": "img", "vllm_bin": "vllm"}
    # Preference reorder: host binary beats docker when listed first.
    assert infer_backend(m, avail, allowed=["vllm", "vllm-docker"]) == "vllm"
    # Disable: unlisted backends are not usable even with resources present.
    assert infer_backend(m, avail, allowed=["llama-server"]) is None
    # Unknown names in the list are ignored defensively.
    assert infer_backend(m, avail, allowed=["nope", "vllm"]) == "vllm"


def test_apply_overrides_pinned_backend_disabled(make_model):
    from llama_packer.scope import ScopeStack
    m = make_model("p", backend="vllm-docker")
    stack = ScopeStack(avail={"llama_bin": "/opt/llama-server",
                              "vllm_image": "img"},
                       allowed=["llama-server"])
    m.resolve_companions()
    stack.finalize(m)
    assert getattr(m, "_override_error", None) and "disabled" in m._override_error


def test_vllm_role_task_flags(make_model):
    host = VllmHostBackend()
    rerank = make_model("r", hf_repo="org/reranker", role="rerank")
    cmd, _ = host.build_cmd(rerank, 8192, 1, "q8_0", _tvars())
    assert "--task score" in cmd
    embed = make_model("e", hf_repo="org/embedder", role="embeddings")
    cmd, _ = host.build_cmd(embed, 8192, 1, "q8_0", _tvars())
    assert "--task embed" in cmd
    chat = make_model("c", hf_repo="org/chat")
    cmd, _ = host.build_cmd(chat, 8192, 1, "q8_0", _tvars())
    assert "--task" not in cmd


def test_vllm_docker_rerank_task_flag(make_model):
    m = make_model("dr", hf_repo="org/reranker", role="rerank")
    cmd, _ = VllmDockerBackend().build_cmd(m, 8192, 1, "q8_0", _tvars())
    assert "--task score" in cmd


def test_vllm_speculative_suppressed_off_chat(make_model):
    # mtp: true must not leak --speculative-config into a pooling model's cmd.
    r = make_model("mr", hf_repo="org/reranker", role="rerank", mtp=True)
    cmd, meta = VllmHostBackend().build_cmd(r, 8192, 1, "q8_0", _tvars())
    assert "--speculative-config" not in cmd
    assert meta["mtp_enabled"] is False
    # ...but chat models keep it.
    c = make_model("mc", hf_repo="org/chat", mtp=True)
    cmd, meta = VllmHostBackend().build_cmd(c, 8192, 1, "q8_0", _tvars())
    assert "--speculative-config" in cmd
    assert meta["mtp_enabled"] is True


def test_unsupported_reason_accepts_vllm_rerank(make_model):
    from llama_packer.backends import infer_backend
    m = make_model("r3", role="rerank")
    m.gguf_path = None
    m.frontmatter["hf_url"] = "https://huggingface.co/org/R3-rerank"
    assert infer_backend(m, {"vllm_image": "img"}) == "vllm-docker"


# ── vLLM recipe keys (C1-C5) ─────────────────────────────────────────────

def _safetensors_model(tmp_path, stem="st", **fm):
    """A vLLM Model whose local file is a real .safetensors under tmp_path."""
    from llama_packer.model import Model
    f = tmp_path / f"{stem}.safetensors"
    f.write_bytes(b"w")
    model = Model(tmp_path / f"{stem}.md", {"name": stem, **fm})
    model.gguf_path = f
    return model


def test_vllm_quantization_flag(make_model):
    m = make_model("v", hf_repo="org/model", vllm_quantization="modelopt_mixed")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--quantization modelopt_mixed" in cmd
    dcmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--quantization modelopt_mixed" in dcmd
    # Absent -> not emitted: vLLM auto-detects from the checkpoint config.
    m2 = make_model("v2", hf_repo="org/model")
    cmd2, _ = VllmHostBackend().build_cmd(m2, 65536, 1, "q8_0", _tvars())
    assert "--quantization" not in cmd2


def test_vllm_moe_backend_flag(make_model):
    m = make_model("v", hf_repo="org/moe", moe_backend="marlin")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--moe-backend marlin" in cmd
    m2 = make_model("v2", hf_repo="org/dense")
    cmd2, _ = VllmHostBackend().build_cmd(m2, 65536, 1, "q8_0", _tvars())
    assert "--moe-backend" not in cmd2


def test_vllm_mamba_flags(make_model, caplog):
    full = {
        "backend": "flashinfer",
        "ssm_cache_dtype": "float16",
        "stochastic_rounding": True,
        "philox_rounds": 5,
        "cache_mode": "align",
    }
    m = make_model("v", hf_repo="org/hybrid", mamba=full)
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    for flag in ("--mamba-backend flashinfer",
                 "--mamba-ssm-cache-dtype float16",
                 "--enable-mamba-cache-stochastic-rounding",
                 "--mamba-cache-philox-rounds 5",
                 "--mamba-cache-mode align"):
        assert flag in cmd, flag
    # Partial mapping -> only the set sub-keys emit.
    m2 = make_model("v2", hf_repo="org/hybrid2", mamba={"backend": "triton"})
    cmd2, _ = VllmHostBackend().build_cmd(m2, 65536, 1, "q8_0", _tvars())
    assert "--mamba-backend triton" in cmd2
    assert "--mamba-ssm-cache-dtype" not in cmd2
    # A bare boolean is malformed: warn + no flags.
    m3 = make_model("v3", hf_repo="org/hybrid3", mamba=True)
    with caplog.at_level(logging.WARNING):
        cmd3, _ = VllmHostBackend().build_cmd(m3, 65536, 1, "q8_0", _tvars())
    assert "--mamba" not in cmd3
    assert any("must be a mapping" in r.message for r in caplog.records)
    # Unknown sub-keys are warned, not silently dropped.
    m4 = make_model("v4", hf_repo="org/hybrid4",
                    mamba={"backend": "auto", "bakend": "oops"})
    with caplog.at_level(logging.WARNING):
        cmd4, _ = VllmHostBackend().build_cmd(m4, 65536, 1, "q8_0", _tvars())
    assert "--mamba-backend auto" in cmd4
    assert any("unknown mamba sub-key" in r.message for r in caplog.records)


def test_vllm_tool_call_parser(make_model):
    m = make_model("v", hf_repo="org/chat", tool_call_parser="qwen3_coder")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser qwen3_coder" in cmd
    # Non-chat roles get no parser flags (generation-only feature).
    e = make_model("e", hf_repo="org/embedder", role="embeddings",
                   tool_call_parser="qwen3_coder")
    ecmd, _ = VllmHostBackend().build_cmd(e, 8192, 1, "q8_0", _tvars())
    assert "--tool-call-parser" not in ecmd


def test_vllm_reasoning_parser(make_model):
    # Distinct from llama.cpp's --reasoning-format (which _strip_llama_only_flags
    # removes from vllm args): --reasoning-parser is a different string and stays.
    m = make_model("v", hf_repo="org/chat", reasoning_parser="nemotron_v3")
    cmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert "--reasoning-parser nemotron_v3" in cmd
    assert "--reasoning-format" not in cmd
    r = make_model("r", hf_repo="org/reranker", role="rerank",
                   reasoning_parser="nemotron_v3")
    rcmd, _ = VllmHostBackend().build_cmd(r, 8192, 1, "q8_0", _tvars())
    assert "--reasoning-parser" not in rcmd


# ── vLLM docker path mapping + mounts (C6) ───────────────────────────────

def _hf_snapshot(tmp_path, repo_dir="models--org--draft", snap="d1"):
    """Fake HF hub layout; returns (hf_root, snapshot_dir)."""
    hf = tmp_path / "hf"
    d = hf / "hub" / repo_dir / "snapshots" / snap
    d.mkdir(parents=True)
    (d / "draft.safetensors").write_bytes(b"x")
    return hf, d


def test_vllm_spec_draft_local_path_docker(make_model, tmp_path):
    hf, snap = _hf_snapshot(tmp_path)
    m = make_model("v", hf_repo="org/model",
                   speculative_config={"method": "draft_model",
                                       "model": str(snap),
                                       "num_speculative_tokens": 3})
    tv = {**_tvars(), "hf_cache": str(hf)}
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert f"-v {hf}:/root/.cache/huggingface" in cmd
    ref = "/root/.cache/huggingface/hub/models--org--draft/snapshots/d1"
    assert ref in cmd
    import json as _json
    # JSON is the trailing token of this cmd (no tool/reasoning parser set).
    spec = _json.loads(cmd.split("--speculative-config ", 1)[1].strip())
    assert spec.get("model") == ref
    # Host backend keeps the raw host path.
    hcmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert str(snap) in hcmd
    assert "/root/.cache/huggingface" not in hcmd


def test_vllm_spec_draft_models_dir_docker(make_model, tmp_path):
    (tmp_path / "draft.safetensors").write_bytes(b"x")
    m = make_model("v", hf_repo="org/model",
                   speculative_config={"method": "draft_model",
                                       "model": str(tmp_path / "draft.safetensors"),
                                       "num_speculative_tokens": 3})
    tv = {**_tvars(), "models_dirs": [str(tmp_path)]}
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    # Under a models_dir the ref rides the existing /models bind — no extra mount.
    assert '"model":"/models/draft.safetensors"' in cmd
    assert "-v" in cmd  # the models_dir bind itself


def test_vllm_spec_draft_elsewhere_docker(make_model, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "draft.safetensors").write_bytes(b"x")
    m = make_model("v", hf_repo="org/model",
                   speculative_config={"method": "draft_model",
                                       "model": str(other / "draft.safetensors")})
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert f"-v {other}:/ext0" in cmd
    assert '"model":"/ext0/draft.safetensors"' in cmd


def test_vllm_docker_model_ref_mapped(make_model, tmp_path):
    # Local safetensors refs are container-path rewritten (the old host-path leak).
    m = _safetensors_model(tmp_path)
    tv = {**_tvars(), "models_dirs": [str(tmp_path)]}
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert "--model /models/st.safetensors" in cmd
    # Under the HF cache root -> /root/.cache/huggingface/<rel>.
    hf, snap = _hf_snapshot(tmp_path)
    m2 = _safetensors_model(tmp_path, stem="st2")
    m2.gguf_path = snap / "draft.safetensors"
    tv2 = {**_tvars(), "hf_cache": str(hf)}
    cmd2, _ = VllmDockerBackend().build_cmd(m2, 65536, 1, "q8_0", tv2)
    assert "--model /root/.cache/huggingface/hub/models--org--draft/snapshots/d1/draft.safetensors" in cmd2
    # Host backend keeps the raw host path.
    hcmd, _ = VllmHostBackend().build_cmd(m, 65536, 1, "q8_0", _tvars())
    assert f"--model {tmp_path / 'st.safetensors'}" in hcmd


def test_vllm_docker_hf_mount_offline(make_model, caplog):
    tv = {**_tvars(), "hf_cache": "/nas/hf"}
    m = make_model("v", hf_repo="org/model")
    cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert "-v /nas/hf:/root/.cache/huggingface" in cmd
    assert "-e HF_HOME=/root/.cache/huggingface" in cmd
    assert "-e HF_HUB_OFFLINE=1" in cmd
    # Empty hf_cache + repo-id ref -> no hub mount (the offline env vars stay
    # — env is unconditional; the models_dir bind is unrelated); the pack-time
    # warning names the fix.  The hub mount's source is exactly the configured
    # hf_cache path, so its absence is the assertion.
    m2 = make_model("v2", hf_repo="org/model")
    with caplog.at_level(logging.WARNING):
        cmd2, _ = VllmDockerBackend().build_cmd(m2, 65536, 1, "q8_0", _tvars())
    assert "-v /nas/hf:" not in cmd2
    assert any("hf_cache is not configured" in r.message for r in caplog.records)


def test_vllm_docker_local_file_no_hf_warning(make_model, tmp_path, caplog):
    # A local-file model needs no hub; empty hf_cache must not warn.
    m = _safetensors_model(tmp_path)
    tv = {**_tvars(), "models_dirs": [str(tmp_path)]}
    with caplog.at_level(logging.WARNING):
        cmd, _ = VllmDockerBackend().build_cmd(m, 65536, 1, "q8_0", tv)
    assert not any("hf_cache is not configured" in r.message
                   for r in caplog.records)
    assert "--model /models/st.safetensors" in cmd


# ── whisper-server backend ────────────────────────────────────────────────

def test_whisper_server_registered():
    b = get_backend("whisper-server")
    assert b.formats == {".bin"}
    assert b.roles == {"s2t"}
    assert b.proxied is True


def test_whisper_server_requires_binary(make_model):
    b = get_backend("whisper-server")
    assert not b.is_available({})
    assert b.is_available({"whisper_bin": "/opt/whisper-server"})


def test_whisper_server_cmd(make_model):
    m = make_model("w", role="s2t")
    m.gguf_path = Path("/models/s2t/ggml-large-v3.bin")
    cmd, meta = get_backend("whisper-server").build_cmd(
        m, 32768, 1, "q8_0", {"whisper_bin": "/opt/whisper-server"})
    assert cmd.startswith("/opt/whisper-server --host 0.0.0.0 --port ${PORT}")
    assert "--model /models/s2t/ggml-large-v3.bin" in cmd
    assert "--parallel 1" in cmd
    assert meta == {}


def test_whisper_server_cli_args_pass_through(make_model):
    m = make_model("w", role="s2t", cli_args="--language en")
    m.gguf_path = Path("/models/s2t/ggml-base.bin")
    cmd, _ = get_backend("whisper-server").build_cmd(
        m, 32768, 1, "q8_0", {"whisper_bin": "whisper-server"})
    assert "--language en" in cmd


def test_infer_backend_whisper(make_model):
    from llama_packer.backends import infer_backend
    m = make_model("w", role="s2t")
    m.gguf_path = Path("/models/s2t/ggml-base.bin")
    # No whisper binary configured → no inference.
    assert infer_backend(m, {"llama_bin": "/opt/llama"}) is None
    # Configured → whisper-server wins for role s2t (.bin is not .gguf anyway).
    assert infer_backend(m, {"llama_bin": "/opt/llama",
                             "whisper_bin": "/opt/whisper-server"}) == "whisper-server"


def test_fixed_overhead_backends_include_whisper():
    from llama_packer.backends import FIXED_OVERHEAD_BACKENDS
    assert FIXED_OVERHEAD_BACKENDS == {"sd-server", "whisper-server", "kokoro-podman"}


# ── kokoro-podman backend ─────────────────────────────────────────────────

def test_kokoro_registered():
    b = get_backend("kokoro-podman")
    assert b.formats == {".onnx", "hf_repo"}
    assert b.roles == {"t2s"}
    assert b.proxied is True


def test_kokoro_requires_image():
    from llama_packer.backends.kokoro import KOKORO_DEFAULT_IMAGES
    b = get_backend("kokoro-podman")
    assert not b.is_available({})
    assert b.is_available({"kokoro_image": KOKORO_DEFAULT_IMAGES["nvidia"]})


def _kokoro_model(tmp_path, **fm):
    """hf_repo-only t2s Model (weights baked into the container image)."""
    from llama_packer.model import Model
    md = tmp_path / "k.md"
    md.write_text("---\nname: k\n---\n")
    fm.setdefault("role", "t2s")
    fm.setdefault("hf_repo", "hexgrad/Kokoro-82M")
    return Model(md, fm)


def test_kokoro_cmd_nvidia(tmp_path):
    m = _kokoro_model(tmp_path)
    assert m.gguf_path is None and m.hf_repo  # image-baked: no local file
    cmd, meta = get_backend("kokoro-podman").build_cmd(m, 0, 1, "q8_0", {
        "kokoro_image": "ghcr.io/remsky/kokoro-fastapi-gpu:latest",
        "kokoro_vendor": "nvidia",
        "kokoro_container_port": 8880,
    })
    assert cmd.startswith("podman run --init --rm --name ${MODEL_ID}")
    assert "-p ${PORT}:8880" in cmd
    assert "--device nvidia.com/gpu=all" in cmd
    assert cmd.endswith("ghcr.io/remsky/kokoro-fastapi-gpu:latest")
    assert meta == {}


def test_kokoro_cmd_amd_uses_native_rocm_devices(tmp_path):
    m = _kokoro_model(tmp_path)
    cmd, _ = get_backend("kokoro-podman").build_cmd(m, 0, 1, "q8_0", {
        "kokoro_image": "ghcr.io/remsky/kokoro-fastapi-rocm:latest",
        "kokoro_vendor": "amd",
    })
    assert "--device /dev/kfd --device /dev/dri" in cmd
    assert "--group-add video" in cmd and "--group-add render" in cmd
    assert "kokoro-fastapi-rocm:latest" in cmd


def test_kokoro_cmd_cpu_has_no_device_flags(tmp_path):
    m = _kokoro_model(tmp_path)
    cmd, _ = get_backend("kokoro-podman").build_cmd(m, 0, 1, "q8_0", {
        "kokoro_image": "ghcr.io/remsky/kokoro-fastapi-cpu:latest",
        "kokoro_vendor": "cpu",
    })
    assert "--device" not in cmd


def test_kokoro_podman_args_and_voices_override(tmp_path):
    m = _kokoro_model(tmp_path)
    cmd, _ = get_backend("kokoro-podman").build_cmd(m, 0, 1, "q8_0", {
        "kokoro_image": "img",
        "kokoro_vendor": "nvidia",
        "podman_args": "--device /dev/mygpu",
        "voices_dir": "/mnt/ai/models/t2s/voices",
    })
    # podman_args replaces the auto device flags entirely.
    assert "--device /dev/mygpu" in cmd
    assert "nvidia.com/gpu" not in cmd
    # Voices mount is read-write so combined voicepacks persist.
    assert "-v /mnt/ai/models/t2s/voices:/app/api/src/voices/v1_0" in cmd


def test_infer_backend_kokoro(tmp_path):
    from llama_packer.backends import infer_backend
    # Weights are baked into the image — hf_repo-only sidecars are typical.
    m = _kokoro_model(tmp_path)
    assert infer_backend(m, {"kokoro_image": "img"}) == "kokoro-podman"
