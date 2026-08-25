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


def test_capabilities_context_is_max_trained_not_served(make_model):
    # capabilities.context must advertise the model's max trained context, not
    # the VRAM-served -c limit; the served limit stays in metadata.ctx_size.
    model = make_model("m", backend="llama-server")
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768, ctx_size=4096,
    )
    assert entry["capabilities"]["context"] == 32768
    assert entry["metadata"]["ctx_size"] == 4096
    assert "--ctx-size 4096" in entry["cmd"] or "-c 4096" in entry["cmd"]


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


def test_sidecar_cache_type_drives_cmd(make_model, monkeypatch):
    # A sidecar cache_type overrides the profile default for both the emitted
    # flags and the VRAM calc (which the grouped cache_type threads through).
    from llama_packer.writer import build_config

    m = make_model("chat", backend="llama-server", context_length=8192,
                   cache_type="f16")
    monkeypatch.setattr(m.vram, "calc_ctx", lambda *a, **k: 8192)
    profiles = {
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
    }
    config = build_config(
        [m], profiles, {"llama_bin": "/opt/llama-server"},
        fit_bin="unused", vram_total=48 * 1024,
    )
    cmd = config["models"]["chat"]["cmd"]
    assert "--cache-type-k f16 --cache-type-v f16" in cmd


def test_cache_type_and_parallel_not_in_metadata(make_model):
    _, entry = _entry(make_model, cache_type="f16", parallel=2)
    assert "cache_type" not in entry["metadata"]
    assert "parallel" not in entry["metadata"]


# ── Directional modalities ────────────────────────────────────────────────
# llama-swap derives badges from capabilities.in/out: vision = image INPUT,
# Image Gen = text→image OUT, Img→Img = image→image.  A VLM must therefore
# never advertise image on the output side.


def _entry_of(model):
    return _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768,
        ctx_size=32768,
    )[1]


def test_vision_is_input_only(make_model, tmp_path):
    # Regression: in/out used to share one modality list, so every VLM also
    # showed llama-swap's "Image Gen" and "Img→Img" badges.
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    model = make_model("v", mmproj="v-mmproj.gguf")
    caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["text", "image"]
    assert caps["out"] == ["text"]


def test_audio_capability_is_input_only(make_model):
    # audio → Transcription badge only; Speech requires an explicit output.
    caps = _entry_of(make_model("a", capabilities=["audio"]))["capabilities"]
    assert caps["in"] == ["text", "audio"]
    assert caps["out"] == ["text"]


def test_speech_capability_adds_audio_output(make_model):
    caps = _entry_of(make_model("s", capabilities=["speech"]))["capabilities"]
    assert caps["in"] == ["text"]
    assert caps["out"] == ["text", "audio"]


def test_dropped_mmproj_removes_image_input_not_output(make_model, tmp_path):
    (tmp_path / "t-mmproj.gguf").write_bytes(b"x")
    model = make_model("t", mmproj="t-mmproj.gguf")
    _, entry = _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=32768,
        ctx_size=32768,
        include_mmproj=False,
    )
    caps = entry["capabilities"]
    assert caps["in"] == ["text"]
    assert caps["out"] == ["text"]



# ── s2t (whisper-server) role ─────────────────────────────────────────────

def test_s2t_role_is_audio_in_text_out(make_model):
    model = make_model("w", role="s2t")
    caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["audio"]
    assert caps["out"] == ["text"]


def test_s2t_entry_gets_proxy_fields(make_model):
    # Proxied backends (whisper-server) must emit proxy + checkEndpoint "/"
    # — llama-swap's default /health never returns 200 for them.
    model = make_model("w", role="s2t", backend="whisper-server")
    entry = _entry_of(model)
    assert entry["proxy"] == "http://127.0.0.1:${PORT}"
    assert entry["checkEndpoint"] == "/"


def test_chat_models_still_have_no_proxy_fields(make_model):
    entry = _entry_of(make_model("m"))
    assert "proxy" not in entry
    assert "checkEndpoint" not in entry
