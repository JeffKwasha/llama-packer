# tests/test_writer.py
"""Command assembly: templates, cli_args, backend selection."""

from __future__ import annotations

from llama_packer.writer import _build_entry


def _entry(make_model, stem="m", **frontmatter):
    model = make_model(stem, **frontmatter)
    return _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768,
        ctx_size=32768,
    )


def test_cli_args_emitted_once(make_model):
    # Regression: cli_args used to be injected twice (template `extra` hole
    # plus the trailing frontmatter append).
    _, entry = _entry(make_model, cli_args="--seed 42")
    assert entry["cmd"].count("--seed 42") == 1


def test_chat_cmd_shape(make_model):
    _, entry = _entry(make_model)
    cmd = entry["cmd"]
    assert cmd.startswith("/opt/llama-server --port ${PORT} -m ")
    assert "--n-gpu-layers 999" in cmd
    assert "--cache-type-k q8_0 --cache-type-v q8_0" in cmd


def test_cpu_resident_uses_cpu_layers(make_model):
    _, entry = _entry(make_model, device="cpu")
    assert "--n-gpu-layers 0" in entry["cmd"]


def test_vllm_binary_cmd(make_model):
    _, entry = _entry(
        make_model, "v", template="vllm", hf_repo="org/model", context_length=65536,
    )
    cmd = entry["cmd"]
    assert cmd.startswith("vllm serve")
    assert "--model org/model" in cmd
    assert "--max-model-len 32768" in cmd
    assert "--gpu-memory-utilization" in cmd


def test_vllm_docker_cmd(make_model):
    _, entry = _entry(
        make_model, "d", template="vllm-docker", hf_repo="org/model",
    )
    cmd = entry["cmd"]
    assert cmd.startswith("docker run")
    assert "--model org/model" in cmd
    assert "--served-model-name ${MODEL_ID}" in cmd


def test_embeddings_role_flags(make_model):
    _, entry = _entry(make_model, "e", role="embeddings")
    assert "--embedding --embd-normalize 2" in entry["cmd"]


def test_rerank_role_flags(make_model):
    _, entry = _entry(make_model, "r", role="rerank")
    assert "--rerank --pooling rank" in entry["cmd"]
