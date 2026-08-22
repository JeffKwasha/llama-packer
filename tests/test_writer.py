# tests/test_writer.py
"""Command assembly: backends, cli_args, chat templates, role flags."""

from __future__ import annotations

from llama_packer.writer import _build_entry


def _entry(make_model, stem="m", backend="llama-server", **frontmatter):
    model = make_model(stem, backend=backend, **frontmatter)
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
        make_model, "v", backend="vllm", hf_repo="org/model", context_length=65536,
    )
    cmd = entry["cmd"]
    assert cmd.startswith("vllm serve")
    assert "--model org/model" in cmd
    assert "--max-model-len 32768" in cmd
    assert "--gpu-memory-utilization" in cmd


def test_vllm_docker_cmd(make_model):
    _, entry = _entry(
        make_model, "d", backend="vllm-docker", hf_repo="org/model",
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


def test_chat_template_emitted(make_model, tmp_path):
    tpl = tmp_path / "qwen.jinja"
    tpl.write_text("x")
    model = make_model("q", backend="llama-server")
    model._resolved_chat_template = tpl.resolve()
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768, ctx_size=32768,
    )
    assert "--jinja --chat-template-file" in entry["cmd"]
    assert "qwen" in entry["metadata"]["chat_template"]


def test_chat_template_kwargs_exposed(make_model, tmp_path):
    tpl = tmp_path / "qwen.jinja"
    tpl.write_text("x")
    model = make_model("q", backend="llama-server",
                       chat_template_kwargs={"enable_thinking": False})
    model._resolved_chat_template = tpl.resolve()
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768, ctx_size=32768,
    )
    assert entry["metadata"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_build_config_skips_unsupported_and_emits(make_model, monkeypatch):
    from llama_packer.writer import build_config

    llama = make_model("chat", backend="llama-server", context_length=8192,
                       base_model="llama3")
    # A vLLM backend with a local .gguf only is an unsupported format -> skipped.
    bad_vllm = make_model("bad", backend="vllm", context_length=8192)

    for m in (llama, bad_vllm):
        monkeypatch.setattr(m.vram, "calc_ctx", lambda *a, **k: 8192)

    profiles = {
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
    }
    config = build_config(
        [llama, bad_vllm], profiles,
        {"llama_bin": "/opt/llama-server", "models_dir": str(make_model("x").md_path.parent)},
        fit_bin="unused", vram_total=48 * 1024, spare=None, max_context=None,
    )
    ids = list(config["models"])
    assert "chat" in ids
    assert "bad" not in ids


def test_build_config_filters_before_vram_passes(make_model, monkeypatch):
    # The unsupported-model filter must run BEFORE the mmproj VRAM pre-pass:
    # rejected models must never reach calc_ctx.  The model gets an mmproj
    # companion so the pre-pass would otherwise consult it.
    from llama_packer.writer import build_config

    bad_vllm = make_model("bad", backend="vllm", mmproj="bad-mmproj.gguf")
    (bad_vllm.md_path.parent / "bad-mmproj.gguf").write_bytes(b"x")
    bad_vllm.mmproj = bad_vllm  # companion present -> pre-pass engages
    monkeypatch.setattr(
        bad_vllm.vram, "calc_ctx",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("filtered model reached calc_ctx")),
    )

    profiles = {
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
    }
    config = build_config(
        [bad_vllm], profiles,
        {"llama_bin": "/opt/llama-server"},
        fit_bin="unused", vram_total=48 * 1024,
    )
    assert config["models"] == {}

