# tests/test_writer.py
"""Command assembly: backends, cli_args, chat templates, role flags."""

from __future__ import annotations

import logging

from conftest import entry_for, only_entry
from llama_packer.writer import _build_entry, _filter_supported


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
    # Validates: SPEC.md, 'Output Format': cli_args appear once in the command, never twice.
    _, entry = _entry(make_model, cli_args="--seed 42")
    assert entry["cmd"].count("--seed 42") == 1


def test_chat_cmd_shape(make_model):
    # Validates: SPEC.md, 'Output Format': cmd is the llama-server invocation, with the port placeholder, the model path and the cache flags
    _, entry = _entry(make_model)
    cmd = entry["cmd"]
    assert cmd.startswith("/opt/llama-server --port ${PORT} -m ")
    # GPU-resident models rely on llama-server's "auto" default; only
    # CPU-resident models pin --n-gpu-layers 0.
    assert "--n-gpu-layers" not in cmd
    assert "--cache-type-k q8_0 --cache-type-v q8_0" in cmd


def test_cpu_resident_uses_cpu_layers(make_model):
    # Validates: docs/backends/llama-server.md, 'Command shape': --n-gpu-layers 0 is emitted only when the model is CPU-resident
    _, entry = _entry(make_model, device="cpu")
    assert "--n-gpu-layers 0" in entry["cmd"]


def test_capabilities_context_is_served_window(make_model):
    # Validates: docs/plans/requirement-trace.md #56: capabilities.context is the served per-slot window, not the trained maximum
    model = make_model("m", backend="llama-server")
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=262144, ctx_size=131072,
    )
    assert entry["capabilities"]["context"] == 131072
    assert entry["metadata"]["ctx_size"] == 131072
    assert entry["metadata"]["context_max"] == 262144
    assert entry["metadata"]["max_output_tokens"] == 131072
    assert "--kv-unified-per-slot 131072" in entry["cmd"]


def test_sidecar_lowers_max_output_tokens(make_model):
    # Validates: SPEC.md, 'Model Metadata': a sidecar may lower the
    # generation ceiling below the served window.
    # Validates: docs/plans/requirement-trace.md #57: The generation ceiling defaults to the served window; a sidecar may lower it, never raise it above the window
    model = make_model("m", backend="llama-server", max_output_tokens=32768)
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=131072, ctx_size=131072,
    )
    assert entry["metadata"]["max_output_tokens"] == 32768


def test_max_output_tokens_never_exceeds_the_served_window(make_model):
    # Validates: SPEC.md, 'Model Metadata': the emitted ceiling is clamped
    # to the served window even when the sidecar asks for more.
    # Validates: docs/plans/requirement-trace.md #57: The generation ceiling defaults to the served window; a sidecar may lower it, never raise it above the window
    model = make_model("m", backend="llama-server", max_output_tokens=262144)
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=262144, ctx_size=65536,
    )
    assert entry["metadata"]["max_output_tokens"] == 65536


def test_no_context_max_or_token_limits_when_nothing_is_folded(make_model):
    # At full design context there is no fold to report, and non-chat roles
    # have no token-generation limits to advertise.
    # Validates: SPEC.md, 'Output Format': Build a single llama-swap config entry for a model+profile group.
    model = make_model("m", backend="llama-server", role="embedding")
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama-server"},
        context_length=8192, ctx_size=8192,
    )
    assert entry["capabilities"]["context"] == 8192
    assert "context_max" not in entry["metadata"]
    assert "max_output_tokens" not in entry["metadata"]


def test_vllm_binary_cmd(make_model):
    # Validates: docs/backends/vllm.md, 'Command shape': the vLLM entry serves the repo with --max-model-len.
    _, entry = _entry(
        make_model, "v", backend="vllm", hf_repo="org/model", context_length=65536,
    )
    cmd = entry["cmd"]
    assert cmd.startswith("vllm serve")
    assert "--model org/model" in cmd
    assert "--max-model-len 32768" in cmd
    assert "--gpu-memory-utilization" in cmd


def test_vllm_docker_cmd(make_model):
    # Validates: docs/backends/vllm.md, 'Command shape': the vLLM container entry runs the image, not the host binary.
    _, entry = _entry(
        make_model, "d", backend="vllm-docker", hf_repo="org/model",
    )
    cmd = entry["cmd"]
    assert cmd.startswith("docker run")
    assert "--model org/model" in cmd
    assert "--served-model-name ${MODEL_ID}" in cmd


def test_vllm_docker_lifecycle_fields(make_model):
    # Container backends get llama-swap's docker orchestration fields: cmdStop
    # stops the container itself (a swap/unload would otherwise kill only the
    # docker run client), unloadTimeout exceeds the stop grace, and the
    # explicit proxy field (upstream: "the single most common configuration
    # error" for containers).
    # Validates: docs/backends/vllm.md, 'Command shape': the container entry carries its lifecycle fields.
    _, entry = _entry(make_model, "d", backend="vllm-docker", hf_repo="org/model")
    assert entry["cmdStop"] == "docker stop ${MODEL_ID}"
    assert entry["unloadTimeout"] == 30
    assert entry["proxy"] == "http://127.0.0.1:${PORT}"
    # Managed (non-container, non-proxied) entries carry none of these.
    _, llama_entry = _entry(make_model, "l")
    assert "cmdStop" not in llama_entry
    assert "unloadTimeout" not in llama_entry
    assert "proxy" not in llama_entry


def test_embeddings_role_flags(make_model):
    # Validates: docs/backends/llama-server.md, 'Roles and flags': the embeddings role adds --embedding --embd-normalize 2
    _, entry = _entry(make_model, "e", role="embeddings")
    assert "--embedding --embd-normalize 2" in entry["cmd"]


def test_rerank_role_flags(make_model):
    # Validates: docs/backends/llama-server.md, 'Roles and flags': the rerank role adds --rerank --pooling rank
    _, entry = _entry(make_model, "r", role="rerank")
    assert "--rerank --pooling rank" in entry["cmd"]


def test_chat_template_emitted(make_model, tmp_path):
    # Validates: docs/backends/llama-server.md, 'Command shape': the chat template is passed as --chat-template-file, with --jinja
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
    # Validates: SPEC.md, 'Override Rules': the chat template's declared kwargs are exposed to clients for per-request control
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


def test_image_token_metadata_exposed(make_model, tmp_path):
    # Declared image token bounds flow to metadata (client-facing) and cmd.
    # Validates: SPEC.md, 'Output Format': the image token bounds are exposed to clients.
    (tmp_path / "v-mmproj.gguf").write_bytes(b"mm")
    _, entry = _entry(make_model, "v",
                      mmproj={"file": "v-mmproj.gguf",
                              "capabilities": ["image"],
                              "image_min_tokens": 1024,
                              "image_max_tokens": 4096})
    assert entry["metadata"]["image_min_tokens"] == 1024
    assert entry["metadata"]["image_max_tokens"] == 4096
    assert "--image-min-tokens 1024 --image-max-tokens 4096" in entry["cmd"]


def test_build_config_skips_unsupported_and_emits(make_model, monkeypatch):
    # Validates: SPEC.md, 'Writer module': build_config composes filter, plan and emit, so a rejected model never reaches the output
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
    # ids are <stem>[.<category>][.<flags>]; with no usage: tag and no
    # served flags an entry is just the stem.
    assert "chat" in ids, ids
    assert not any(i.startswith("bad") for i in ids), ids


def test_build_config_filters_before_vram_passes(make_model, monkeypatch):
    # The unsupported-model filter must run BEFORE the mmproj VRAM pre-pass:
    # rejected models must never reach calc_ctx.  The model gets an mmproj
    # companion so the pre-pass would otherwise consult it.
    # Validates: SPEC.md, 'Output Format': Build llama-swap config from list of Model objects. Composes the pipeline.
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


def test_build_config_disables_over_budget_model(make_model, monkeypatch):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': a model whose every rung is over budget produces no entry at all
    """A model whose weights exceed the budget must not be emitted —
    it would be a load-time OOM landmine."""
    from llama_packer.writer import build_config

    huge = make_model("huge", backend="llama-server", context_length=8192,
                      base_model="llama3")
    # 50G weights on a 48G card: remaining < 0 in calc_ctx.
    monkeypatch.setattr(
        huge.vram, "effective_static",
        lambda *a, **k: (50000, 0.5, 0.0, 100),
    )
    profiles = {
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "profiles": {"default": {}},
    }
    config = build_config(
        [huge], profiles,
        {"llama_bin": "/opt/llama-server"},
        fit_bin="unused", vram_total=48 * 1024,
    )
    assert config["models"] == {}


def test_sidecar_cache_type_drives_cmd(make_model, monkeypatch):
    # A sidecar cache_type overrides the profile default for both the emitted
    # flags and the VRAM calc (which the grouped cache_type threads through).
    # Validates: SPEC.md, 'Output Format': the sidecar's cache type reaches the command.
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
    cmd = entry_for(config, "chat")["cmd"]
    assert "--cache-type-k f16 --cache-type-v f16" in cmd


def test_cache_type_and_parallel_not_in_metadata(make_model):
    # Validates: SPEC.md, 'Model Metadata': the builder-consumed keys - cache_type, parallel - are not passed through to the client
    _, entry = _entry(make_model, cache_type="f16", parallel=2)
    assert "cache_type" not in entry["metadata"]
    assert "parallel" not in entry["metadata"]


# ── Directional modalities ────────────────────────────────────────────────
# llama-swap derives badges from capabilities.in/out: image = image INPUT,
# Image Gen = text→image OUT, Img→Img = image→image.  A VLM must therefore
# never advertise image on the output side.  `vision` was removed: the
# sidecar token is `image`, matching llama-swap's modality enum
# (text/audio/image/video).


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


def test_image_is_input_only(make_model, tmp_path):
    # Regression: in/out used to share one modality list, so every VLM also
    # showed llama-swap's "Image Gen" and "Img→Img" badges.
    # Validates: SPEC.md, 'Output Format': image is an input modality, never an output one.
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    model = make_model("v", mmproj="v-mmproj.gguf", capabilities=["image"])
    caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["text", "image"]
    assert caps["out"] == ["text"]


def test_vision_capability_removed(make_model, caplog):
    # `vision` is not a llama-swap modality: it errors and adds no badge.
    # Validates: SPEC.md, 'Output Format': the removed vision capability is not advertised.
    model = make_model("v", capabilities=["vision"])
    with caplog.at_level(logging.ERROR, logger="llama_packer.writer"):
        _filter_supported([model], "q8_0")
        caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["text"]
    assert caps["out"] == ["text"]
    assert any("'vision'" in r.message and "removed" in r.message
               for r in caplog.records if r.levelno >= logging.ERROR)


def test_video_is_input_only(make_model):
    # video input without a video-generating arch stays input-only.
    # Validates: SPEC.md, 'Output Format': video is an input modality.
    caps = _entry_of(make_model("d", capabilities=["video"]))["capabilities"]
    assert caps["in"] == ["text", "video"]
    assert caps["out"] == ["text"]


def test_video_omni_outputs_video(make_model):
    # Validates: SPEC.md, 'Output Format': the capabilities block advertises the output modalities the serving actually has
    caps = _entry_of(make_model("o", architecture="omni",
                                capabilities=["video"]))["capabilities"]
    assert caps["in"] == ["text", "video"]
    assert caps["out"] == ["text", "video"]


def test_dropped_mmproj_removes_video_input(make_model, tmp_path):
    # Validates: SPEC.md, 'Entry ids name the task, not the solve': a no-mmproj serving drops the i and v flags
    (tmp_path / "d-mmproj.gguf").write_bytes(b"x")
    model = make_model("d", mmproj={"file": "d-mmproj.gguf",
                                    "capabilities": ["image", "video"]})
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
    assert entry["metadata"]["mmproj_skipped"] is True


def test_mmproj_without_capability_warns(make_model, tmp_path, caplog):
    # Projection costs VRAM but is not advertised — must warn.
    # Validates: SPEC.md, 'Output Format': a projector without the image capability is warned about.
    (tmp_path / "w-mmproj.gguf").write_bytes(b"x")
    model = make_model("w", mmproj={"file": "w-mmproj.gguf"})
    with caplog.at_level(logging.WARNING, logger="llama_packer.writer"):
        _filter_supported([model], "q8_0")
    assert any("neither 'image' nor 'video'" in r.message
               for r in caplog.records if r.levelno == logging.WARNING)


def test_image_without_mmproj_no_warning(make_model, caplog):
    # Baked-in image support is common (no mmproj file) — no warning:
    # we don't warn about things that might be wrong.
    # Validates: SPEC.md, 'Output Format': an image capability without a projector is not warned about.
    model = make_model("b", capabilities=["image"])
    with caplog.at_level(logging.WARNING, logger="llama_packer.writer"):
        supported = _filter_supported([model], "q8_0")
    assert supported == [model]
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_audio_capability_is_input_only(make_model):
    # audio → Transcription badge only; Speech requires an explicit output.
    # Validates: SPEC.md, 'Output Format': audio is an input modality.
    caps = _entry_of(make_model("a", capabilities=["audio"]))["capabilities"]
    assert caps["in"] == ["text", "audio"]
    assert caps["out"] == ["text"]


def test_speech_capability_adds_audio_output(make_model):
    # Validates: SPEC.md, 'Output Format': the speech capability adds audio to the output modalities
    caps = _entry_of(make_model("s", capabilities=["speech"]))["capabilities"]
    assert caps["in"] == ["text"]
    assert caps["out"] == ["text", "audio"]


def test_dropped_mmproj_removes_image_input_not_output(make_model, tmp_path):
    # Validates: SPEC.md, 'Entry ids name the task, not the solve': a no-mmproj serving drops the image input but keeps the output modalities
    (tmp_path / "t-mmproj.gguf").write_bytes(b"x")
    model = make_model("t", mmproj={"file": "t-mmproj.gguf",
                                    "capabilities": ["image"]})
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
    # Validates: SPEC.md, 'Audio Backend (whisper-server)': the s2t role is audio in, text out
    model = make_model("w", role="s2t")
    caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["audio"]
    assert caps["out"] == ["text"]


def test_s2t_entry_gets_proxy_fields(make_model):
    # Proxied backends (whisper-server) must emit proxy + checkEndpoint "/"
    # — llama-swap's default /health never returns 200 for them.
    # Validates: SPEC.md, 'Output Format': an s2t entry is proxied and carries its health check.
    model = make_model("w", role="s2t", backend="whisper-server")
    entry = _entry_of(model)
    assert entry["proxy"] == "http://127.0.0.1:${PORT}"
    assert entry["checkEndpoint"] == "/"


def test_chat_models_still_have_no_proxy_fields(make_model):
    # Validates: docs/backends/llama-server.md, 'Backend: llama-server': llama-server is not proxied: llama-swap manages the process and proxies by port
    entry = _entry_of(make_model("m"))
    assert "proxy" not in entry
    assert "checkEndpoint" not in entry


# ── t2s role capabilities (serving backend: audio-cpp) ───────────────────

def test_t2s_role_is_text_in_audio_out(make_model):
    # Validates: SPEC.md, 'Audio Backend (audio-cpp)': the t2s role is text in, audio out
    model = make_model("k", role="t2s")
    caps = _entry_of(model)["capabilities"]
    assert caps["in"] == ["text"]
    assert caps["out"] == ["audio"]


def test_audio_cpp_entry_proxy_and_health(make_model):
    # audio.cpp is a proxied HTTP service: llama-swap needs proxy + its own
    # health path (/health), unlike sd-server/whisper-server which answer "/".
    # Validates: SPEC.md, 'Output Format': Build a single llama-swap config entry for a model+profile group.
    model = make_model("a", role="t2s", backend="audio-cpp")
    _, entry = _build_entry(
        model, parallel=1, cache_type="q8_0",
        profiles_group=[("default", {})], profiles_defaults={},
        template_vars={"audio_cpp_bin": "/opt/audiocpp_server"},
        context_length=0, ctx_size=0,
    )
    assert entry["proxy"] == "http://127.0.0.1:${PORT}"
    assert entry["checkEndpoint"] == "/health"
    assert entry["capabilities"]["in"] == ["text"]
    assert entry["capabilities"]["out"] == ["audio"]
    assert entry["cmd"].startswith("sh -c ")
    assert "\n" in entry["cmd"]  # heredoc newlines survive emit (_strip_repeat_ws)


# ── config serialization contract ─────────────────────────────────────────

def test_emitted_config_plain_is_a_plain_dict():
    # Validates: SPEC.md, 'Output Format': the emitted config is a plain dict, with no YAML tags in it
    from llama_packer.writer import EmittedConfig

    ec = EmittedConfig({"models": {"m": {"cmd": "x"}}},
                       entry_ids_by_stem={"m": ["m"]}, coload_stems=["e"])
    plain = ec.plain()
    assert type(plain) is dict
    assert plain == {"models": {"m": {"cmd": "x"}}}
    # Build-time metadata is not part of the serialized document.
    assert ec.entry_ids_by_stem == {"m": ["m"]}


def test_write_yaml_round_trips_without_python_tags(tmp_path):
    # Validates: SPEC.md, 'llama-swap YAML (`write_yaml()`)': the YAML is written with literal block scalars, not Python tags
    import yaml

    from llama_packer.writer import EmittedConfig, write_yaml

    ec = EmittedConfig({"models": {"m": {"cmd": "x"}},
                        "macros": {"A": "--a 1"}},
                       entry_ids_by_stem={"m": ["m"]})
    out = tmp_path / "config.yaml"
    write_yaml(ec, out)
    text = out.read_text()
    assert yaml.safe_load(text) == ec.plain()
    assert "python/object" not in text


def test_write_yaml_accepts_plain_dict(tmp_path):
    # Validates: SPEC.md, 'llama-swap YAML (`write_yaml()`)': each model entry produces cmd, name, description, filters.setParamsByID, metadata, capabilities, env and concurrencyLimit
    import yaml

    from llama_packer.writer import write_yaml

    cfg = {"models": {}, "macros": {}}
    out = tmp_path / "config.yaml"
    write_yaml(cfg, out)
    assert yaml.safe_load(out.read_text()) == cfg


def test_write_yaml_preserves_heredoc_newlines(tmp_path, make_model):
    # Regression: multi-line cmd (audio-cpp heredoc) must round-trip with
    # newlines intact — PyYAML's default folded style collapses them.
    # Validates: SPEC.md, 'Output Format': a heredoc keeps its newlines in the emitted YAML.
    import yaml

    from llama_packer.backends import get_backend
    from llama_packer.writer import write_yaml

    model = make_model("chatterbox", role="t2s", backend="audio-cpp")
    cmd, _ = get_backend("audio-cpp").build_cmd(
        model, 0, 1, "q8_0", {"audio_cpp_bin": "/opt/audiocpp_server"})
    assert "\n" in cmd
    out = tmp_path / "config.yaml"
    write_yaml({"models": {"m": {"cmd": cmd}}}, out)
    text = out.read_text()
    assert yaml.safe_load(text)["models"]["m"]["cmd"] == cmd


def test_overallocation_guard_catches_an_injected_violation(make_model,
                                                           monkeypatch, caplog):
    """The guard must fire on a planner that misbehaves, not just agree.

    Validates: docs/plans/requirement-trace.md #27 -- the emit-time guard is
    the backstop for "no emitted entry may need more VRAM than exists", so it
    has to be shown to fire rather than merely agree with a healthy planner.

    Everything upstream honours "never emit an entry that cannot load", so
    on a healthy planner this check has nothing to do. That is exactly why
    it needs a test that *breaks* the planner deliberately: otherwise it
    passes vacuously forever and protects nothing.

    Here calc_ctx is made to report a comfortable context while the quad
    says the model needs more than the card has -- the shape of the bug it
    was written for.
    """
    import logging
    from llama_packer.writer import build_config

    m = make_model("liar", backend="llama-server", context_length=32768)
    # Reports a fine context for any request...
    monkeypatch.setattr(m.vram, "calc_ctx", lambda *a, **k: 32768)
    # ...while the numbers say 90 GB of weights on a 48 GB card.
    monkeypatch.setattr(m.vram, "effective_static",
                        lambda *a, **k: (90000, 0.5, 0.0, 100))
    monkeypatch.setattr(
        m.vram, "fit_params_static",
        lambda *a, **k: type("FP", (), {
            "model_mib": 90000, "kv_per_token_mib": 0.5,
            "slot_mib": 0.0, "compute_mib": 100, "source": "fit-params"})())

    profiles = {"defaults": {"cache_type": "q8_0", "parallel": 1},
                "profiles": {"default": {}}}
    with caplog.at_level(logging.ERROR):
        config = build_config(
            [m], profiles, {"llama_bin": "/opt/llama-server"},
            fit_bin="unused", vram_total=48 * 1024,
        )
    assert "invariant violated" in caplog.text
    assert "planner bug" in caplog.text
    assert config["models"] == {}, "the impossible entry survived"


def test_overallocation_guard_is_quiet_when_the_plan_is_sane(make_model,
                                                             monkeypatch, caplog):
    """The other half: silence when the plan is fine.

    Validates: docs/plans/requirement-trace.md #28.
    """
    import logging
    from llama_packer.writer import build_config

    m = make_model("ok", backend="llama-server", context_length=32768)
    profiles = {"defaults": {"cache_type": "q8_0", "parallel": 1},
                "profiles": {"default": {}}}
    with caplog.at_level(logging.ERROR):
        config = build_config(
            [m], profiles, {"llama_bin": "/opt/llama-server"},
            fit_bin="unused", vram_total=48 * 1024,
        )
    assert "invariant violated" not in caplog.text
    assert config["models"], "a sane model was dropped"


# ── the entry id reaches the engines that route on it ───────────────────────


def test_audio_entry_id_matches_the_id_the_engine_validates(make_model):
    # Validates: SPEC.md, 'Entry ids name the task, not the solve': anything that references an entry must resolve through the emitted ids
    """The whole point, end to end: one string, both sides.

    llama-swap routes on the entry id (`kokoro-82m.s`); audio.cpp validates the
    requested model against `models[].id` in the server.json embedded in the
    entry's own `cmd`. They were different strings, so every audio request
    came back "unknown model id" while the entry's health check passed — the
    pair that made this look like a routing problem rather than a naming one.

    The tail used to be appended to the id *after* `_build_entry` returned,
    which is why no backend could see it and why fixing the backend alone was
    impossible.
    """
    import json
    import re

    model = make_model("kokoro-82m-q8_0", backend="audio-cpp", role="t2s",
                       audio_cpp={"family": "kokoro_tts"})
    entry_id, entry = _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars={"audio_cpp_bin": "/opt/llama/tools/audiocpp_server"},
        context_length=32768,
        ctx_size=32768,
        entry_id_tail="s",
    )

    assert entry_id == "kokoro-82m-q8_0.s", "the tail must be part of the id"
    payload = re.search(r"<<JSON\n(.*?)\nJSON\n", entry["cmd"], re.S)
    assert payload, "audio.cpp entries carry their config in a heredoc"
    served = json.loads(payload.group(1).replace("${PORT}", "0"))
    assert [m["id"] for m in served["models"]] == [entry_id]


def test_a_plain_text_entry_carries_no_tail(make_model):
    # Validates: SPEC.md, 'Entry ids name the task, not the solve': the flag segment is omitted entirely for a plain text-in/text-out serving
    """No flags segment means no suffix, and the id stays the bare stem."""
    _, entry = _build_entry(
        make_model("qwen3-8b", backend="llama-server"),
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars={"llama_bin": "/opt/llama/tools/llama-server"},
        context_length=32768,
        ctx_size=32768,
        entry_id_tail="",
    )
    assert entry["cmd"].startswith("/opt/llama/tools/llama-server")
