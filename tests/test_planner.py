# tests/test_planner.py
"""Planner / Profiles / emit_config seams: vision variants, grouping, clamps."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from llama_packer.profiles import Profiles
from llama_packer.writer import MatrixKnobs, Planner, emit_config, _solve_matrix_context

TVARS = {"llama_bin": "/opt/llama-server"}


@pytest.fixture
def profiles():
    return Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                     "profiles": {"default": {}, "big": {"temperature": 0.9}}})


def _scripted_ctx(model, by_mmproj, quad=(100, 0.01, 10, 100)):
    """Replace calc_ctx with a fake keyed on include_mmproj.

    Companion-on variants carry their own VRAM budget (bound to the serving
    view), so the fake is installed on the base budget and every view.
    The affine quad is faked the same way: the planner reports per-variant
    memory tags from effective_static, which must never reach a subprocess
    in hermetic tests.
    """
    def fake(vram_total_mb, *, server_bin=None, parallel=1, spare_mb=0,
             include_mmproj=True, baseline_mb=0, cache_type="q8_0",
             design_ctx=None, **kw):
        return by_mmproj[bool(include_mmproj)]
    model.vram.calc_ctx = fake
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.calc_ctx = fake

    def fake_static(fit_bin, cache_type="q8_0", design_ctx=None,
                    include_mmproj=True, llama_args="", allow_cpu=False):
        return quad
    model.vram.effective_static = fake_static
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.effective_static = fake_static


def _vision_model(tmp_path, make_model, name):
    """Model with an mmproj companion; file must exist before construction."""
    (tmp_path / f"{name}-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    return make_model(name, mmproj={"file": f"{name}-mmproj.gguf",
                                    "capabilities": ["image"]})


def test_profiles_spare_precedence():
    p = Profiles({"defaults": {"spare": "2G"}, "profiles": {}})
    assert p.global_spare_mb(vram_total=48000) == 2048
    assert p.global_spare_mb("4G", vram_total=48000) == 2048  # defaults win
    q = Profiles({"profiles": {}})
    assert q.global_spare_mb("512m", vram_total=48000) == 512
    assert q.global_spare_mb() == 0
    assert p.spare_mb(preferred="1G", cli_override="4G", vram_total=48000) == 1024


def test_profiles_groups_fallback_when_nothing_matches(profiles, make_model):
    m = make_model("m", allow_profiles="nomatch")
    groups = profiles.groups_for(m, vram_total=48000)
    # Single defaults-derived group; parallel_for(1)/cache_type_for(default)
    (key, group), = groups.items()
    assert key[:2] == (1, "q8_0")
    assert group == [("default", {"cache_type": "q8_0", "parallel": 1})]


def test_profiles_groups_by_cache_type(make_model):
    m = make_model("m")
    p = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                  "profiles": {"a": {"cache_type": "f16"},
                               "b": {"cache_type": "q8_0"}}})
    groups = p.groups_for(m, vram_total=48000)
    assert len(groups) == 2
    names = [n for g in groups.values() for n, _ in g]
    assert sorted(names) == ["a", "b"]


def test_planner_vision_dropped_and_variant_planned(make_model, profiles, tmp_path):
    # Design ctx 256k so only the budget clamps: vision misses min-context,
    # text-only reaches it → mmproj is dropped and vision kept as a variant.
    (tmp_path / "vis-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("vis", mmproj={"file": "vis-mmproj.gguf",
                                  "capabilities": ["image"]},
                   context_length=262144)
    _scripted_ctx(m, {True: 65536, False: 200000})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["vis"]
    assert len(variants) == 1
    v = variants[0]
    assert v.include_mmproj is False
    assert v.ctx_size == 200000
    assert v.vision_ctx == 65536

    config = emit_config([m], {"vis": variants}, profiles, TVARS)
    # Auto-dropped main entry is renamed <id>-text (bare id always = vision).
    assert set(config["models"]) == {"vis-text", "vis-vision-65k"}
    main = config["models"]["vis-text"]
    vision = config["models"]["vis-vision-65k"]
    assert "--mmproj" not in main["cmd"]
    assert main["name"].endswith("[text]")
    assert main["metadata"]["mmproj_skipped"] is True
    assert main["capabilities"]["in"] == ["text"]
    assert "--mmproj" in vision["cmd"]
    assert vision["name"].endswith("[vision 65k]")


def test_plan_unestimable_flag_on_companion_view(make_model, profiles, tmp_path):
    """view_for() rebuilds VramBudget per view: a measurement failure
    flagged on the companion-on view's budget must still surface as
    estimate_error metadata — the re-check reads the solving budget."""
    m = _vision_model(tmp_path, make_model, "ue")
    _scripted_ctx(m, {True: 4096, False: 4096})
    on = m.view_for(True)
    on.vram.unestimated_reason = "VRAM measurement failed"
    variants = Planner([m], profiles, fit_bin="unused",
                       vram_total=48 * 1024).plan()["ue"]
    assert variants
    assert all(v.estimate_error == "VRAM measurement failed"
               for v in variants)


def test_profiles_rag_parallel_pinned_to_one(make_model, profiles, caplog):
    """Embed/rerank serve single-slot: a declared parallel is ignored with
    a once-per-model note — resident parallelism must never shrink the
    main chat context it serves."""
    import logging

    emb = make_model("emb", role="embeddings", parallel=8)
    (key, _), = profiles.groups_for(emb, vram_total=48000).items()
    assert key[0] == 1
    assert any("single-slot" in r.message for r in caplog.records)
    # chat keeps its declared parallel
    chat = make_model("chat", parallel=8)
    (ckey, _), = profiles.groups_for(chat, vram_total=48000).items()
    assert ckey[0] == 8


def test_planner_below_min_keeps_vision_no_warning(make_model, profiles, tmp_path,
                                                   caplog):
    # Small design ctx: below min-context with AND without vision — dropping
    # buys nothing, so vision is kept and only an info line is logged.
    m = _vision_model(tmp_path, make_model, "tiny")
    _scripted_ctx(m, {True: 32768, False: 32768})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["tiny"]
    assert len(variants) == 2  # bare (vision) + on-demand -text, per group
    assert variants[0].include_mmproj is True   # main entry keeps vision
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_planner_vision_kept_adds_text_variant(make_model, profiles, tmp_path):
    # Big design context so the design-clamp doesn't mask which budget was
    # used; text ctx < vision ctx proves the variant was budgeted WITHOUT the
    # mmproj (include_mmproj=False path).
    (tmp_path / "vis-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    m = make_model("vis", mmproj={"file": "vis-mmproj.gguf",
                                  "capabilities": ["image"]},
                   context_length=262144)
    _scripted_ctx(m, {True: 131072, False: 100000})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["vis"]
    assert len(variants) == 2
    v = variants[0]
    assert v.include_mmproj is True
    assert v.ctx_size == 131072
    assert v.vision_ctx is None
    tv = variants[1]
    assert tv.include_mmproj is False
    assert tv.vision_ctx is None
    assert tv.ctx_size == 100000

    config = emit_config([m], {"vis": variants}, profiles, TVARS)
    assert set(config["models"]) == {"vis", "vis-text"}
    assert "--mmproj" in config["models"]["vis"]["cmd"]
    text = config["models"]["vis-text"]
    assert "--mmproj" not in text["cmd"]
    assert text["name"].endswith("[text]")
    assert text["metadata"]["mmproj_skipped"] is True
    assert text["metadata"]["ctx_size"] == 100000
    assert text["capabilities"]["in"] == ["text"]


def test_bounded_ctx_clamps_design_then_cli(make_model, profiles):
    m = make_model("m")  # sidecar context_length 32768
    m.vram.calc_ctx = lambda *a, **k: 999999
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      max_context=16384, min_context=0)
    ctx = planner._bounded_ctx(m, parallel=1, cache_type="q8_0", spare_mb=0,
                               include_mmproj=True, context_length=m.design_context)
    assert ctx == 16384
    planner.max_context = None
    ctx = planner._bounded_ctx(m, parallel=1, cache_type="q8_0", spare_mb=0,
                               include_mmproj=True, context_length=m.design_context)
    assert ctx == 32768


FP_PARAMS = (1000.0, 0.5, 0.0, 100.0)


def test_solve_matrix_excludes_roles_cpu_and_threads_drop_stems(
        make_model, profiles, monkeypatch):
    from llama_packer import writer

    chat_a = make_model("chat_a", role="chat")
    chat_b = make_model("chat_b", role="chat")
    embed = make_model("e", role="embeddings")
    cpu_chat = make_model("cpu", device="cpu")

    seen = {}

    def fake_effective_static(fit_bin, cache_type="q8_0",
                              design_ctx=None, include_mmproj=True, **kw):
        seen[include_mmproj] = seen.get(include_mmproj, 0) + 1
        return FP_PARAMS

    def fake_fit_params_static(fit_bin, cache_type="q8_0", **kw):
        return SimpleNamespace(model_mib=FP_PARAMS[0],
                               kv_per_token_mib=FP_PARAMS[1],
                               slot_mib=FP_PARAMS[2],
                               compute_mib=FP_PARAMS[3], source="fit-params")

    for m in (chat_a, chat_b, embed):
        m.vram.effective_static = fake_effective_static
        m.vram.fit_params_static = fake_fit_params_static

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 24576

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    result = _solve_matrix_context(
        [chat_a, chat_b, embed, cpu_chat], embed, embed,
        fit_bin="unused", vram_total=48 * 1024, spare="1G",
        profiles=profiles, baseline_mb=0, drop_stems={"chat_b"},
    )
    assert result is not None
    assert result.chat_ctx == 24576
    assert result.coloads == ()
    # Only the two real chat models enter the shared budget;
    # chat_b's mmproj is omitted via drop_stems.
    assert seen == {True: 1, False: 1}
    assert len(captured["chat_models"]) == 2
    assert captured["embed_params"] == FP_PARAMS


# ── matrix knobs ──────────────────────────────────────────────────────────


def test_matrix_knobs_defaults_and_overrides():
    k = MatrixKnobs.from_cfg(None)
    assert (k.min_chat_ctx, k.tools_min_ctx, k.coload_min_ctx,
            k.embed_context, k.rerank_context,
            k.ctx_gain_min, k.estimate_headroom) == (
        65536, 131072, 20480, 4096, 4096, 4096, 1.25)
    k = MatrixKnobs.from_cfg({"min_chat_ctx": 32000, "estimate_headroom": 1.5})
    assert k.min_chat_ctx == 32000
    assert k.estimate_headroom == 1.5


def test_matrix_knobs_rag_caps():
    k = MatrixKnobs.from_cfg({"embed_context": 8192, "rerank_context": 16384})
    assert (k.embed_context, k.rerank_context) == (8192, 16384)
    assert k.rag_cap("embeddings") == 8192
    assert k.rag_cap("rerank") == 16384
    assert k.rag_cap("chat") == k.coload_min_ctx


def test_matrix_knobs_vllm_auto_parallel():
    k = MatrixKnobs.from_cfg({})
    assert k.auto_parallel_max == 8
    assert k.vllm_auto_parallel_max == 16
    k = MatrixKnobs.from_cfg({"vllm_auto_parallel_max": 32})
    assert k.vllm_auto_parallel_max == 32
    k = MatrixKnobs.from_cfg({"vllm_auto_parallel_max": 0})
    assert k.vllm_auto_parallel_max == 16


def test_plan_vllm_uncapped_parallel_zero(make_model, profiles):
    # sidecar parallel 0 on a vLLM model = uncapped: ctx solves single-seq
    # (vLLM validates one max-length seq at startup), and the variant emits
    # parallel 0 so --max-num-seqs is omitted.
    m = make_model("uv", backend="vllm", parallel=0)
    _fake_vram(m, (1000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024)
    variants = planner.plan()["uv"]
    assert variants[0].parallel == 0
    assert variants[0].ctx_size == 32768


def test_plan_llama_parallel_zero_warns_and_defaults(make_model, profiles,
                                                     caplog):
    # parallel 0 is a vLLM-only convention; on llama.cpp it's invalid and
    # falls back to the fleet default with a warning.
    m = make_model("m0", parallel=0)
    _fake_vram(m, (1000, 0.5, 100))
    planner = Planner([m], profiles, fit_bin="unused", vram_total=64 * 1024)
    variants = planner.plan()["m0"]
    assert variants[0].parallel == 1
    assert any("parallel 0 is invalid" in r.message for r in caplog.records)


def test_auto_parallel_vllm_reaches_higher_cap(make_model, profiles):
    # Same budget, same quads: the vLLM backend may use more slots than the
    # llama.cpp cap — continuous batching makes a slot cost KV only.
    # Pin-free profiles (no parallel in defaults) so auto-parallel runs;
    # an explicit sidecar min_context (cascade row 1) sets the floor so the
    # value function can actually reach both caps.
    no_pin = Profiles({"defaults": {"cache_type": "q8_0"},
                       "profiles": {"default": {}}})
    vllm_m = make_model("av", backend="vllm", min_context=4096)
    del vllm_m.frontmatter["context_length"]  # no pin: cap = design
    _fake_vram(vllm_m, (1000, 0.5, 100))
    planner = Planner([vllm_m], no_pin, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    assert planner.plan()["av"][0].parallel == 16

    llama_m = make_model("al", min_context=4096)
    del llama_m.frontmatter["context_length"]
    _fake_vram(llama_m, (1000, 0.5, 100))
    planner = Planner([llama_m], no_pin, fit_bin="unused",
                      vram_total=48 * 1024, min_context=4096)
    assert planner.plan()["al"][0].parallel == 8


def test_matrix_knobs_invalid_values_warn_and_default(caplog):
    with caplog.at_level(logging.WARNING):
        k = MatrixKnobs.from_cfg({"min_chat_ctx": -1, "coload_min_ctx": "big",
                                  "embed_context": 0, "rerank_context": "4k",
                                  "estimate_headroom": 0.5})
    assert k.min_chat_ctx == 65536
    assert k.coload_min_ctx == 20480
    assert (k.embed_context, k.rerank_context) == (4096, 4096)
    assert k.estimate_headroom == 1.25
    assert "min_chat_ctx" in caplog.text
    assert "embed_context" in caplog.text
    assert "rerank_context" in caplog.text
    assert "estimate_headroom" in caplog.text


# ── opportunistic co-load pass ────────────────────────────────────────────


def _fake_vram(m, params):
    mib, kv_factor, compute = params
    m.vram.effective_static = lambda *a, **k: (mib, kv_factor, 0.0, compute)
    m.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=mib, kv_per_token_mib=kv_factor, slot_mib=0.0,
        compute_mib=compute, source="fit-params")


def test_solve_matrix_includes_smallest_coload_skips_big(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", backend="whisper-server", vram_mb=640,
               context_length=8192)
    img = make("img", role="image", backend="sd-server", vram_mb=40000,
               context_length=8192)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    # Pinned vram_mb: authoritative fixed overhead (no headroom).
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None
    img.vram.effective_static = lambda *a, **k: (40000, 0.0, 0.0, 0)
    img.vram.fit_params_static = lambda *a, **k: None

    result = _solve_matrix_context(
        [chat, embed, rerank, s2t, img], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192))
    assert result is not None
    # available = 49152-2048 = 47104; emb+rnk capped to 4096 = 2*1009 = 2018
    # baseline chat budget = 45086-8500 = 36586 -> capped 32768
    assert result.chat_ctx == 32768
    # s2t (640 MB) fits; image (40000 MB) leaves a negative chat budget.
    assert result.coloads == (("s2t", 640),)


def test_solve_matrix_reads_block_tokens_from_on_view(
        tmp_path, make_model, profiles, monkeypatch):
    # image_max_tokens living only in the mmproj block must still reserve
    # the image floor in the matrix solve (read via the companion-on view).
    from llama_packer import writer
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x" * 1024)
    m = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                "capabilities": ["image"],
                                "image_max_tokens": 9000})
    assert m.image_max_tokens is None  # base frontmatter: no floor visible
    e = make_model("e", role="embeddings", context_length=8192)
    r = make_model("r", role="rerank", context_length=8192)
    # The matrix solve reads VRAM through the companion-on view (its own
    # budget object), so the fake goes there; the floor under test comes
    # from the view's merged frontmatter, not from VRAM.
    _fake_vram(m.view_for(True), (8000, 0.4, 500))
    _fake_vram(e, (500, 0.1, 100))
    _fake_vram(r, (500, 0.1, 100))
    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 8192

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)
    result = _solve_matrix_context(
        [m, e, r], e, r, fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192))
    assert result is not None
    floors = {mod.stem: floor for mod, _, _, _, _, _, floor
              in captured["chat_models"]}
    assert floors == {"v": 9000}


def test_solve_matrix_floor_blocks_all_coloads(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", vram_mb=640, context_length=8192)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None

    from llama_packer.writer import MatrixKnobs
    result = _solve_matrix_context(
        [chat, embed, rerank, s2t], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=24576))
    assert result is not None
    # chat_budget = 47104-2018-8500 = 36586 -> factor 1.0 -> round 32768,
    # capped 32768 >= floor -> s2t fits.
    assert result.chat_ctx == 32768
    assert result.coloads == (("s2t", 640),)


def test_solve_matrix_tools_floor_blocks_coload(profiles):
    chat = make("chat", role="chat", capabilities=["tools"])
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    img = make("img", role="image", vram_mb=20000, context_length=8192)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    img.vram.effective_static = lambda *a, **k: (20000, 0.0, 0.0, 0)
    img.vram.fit_params_static = lambda *a, **k: None

    from llama_packer.writer import MatrixKnobs
    # floor: chat solves to 32768 >= tools_min 20000 -> floor 20000.
    # image (20000 MB) would drop chat to ~12288 < 20000 -> skipped.
    result = _solve_matrix_context(
        [chat, embed, rerank, img], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(tools_min_ctx=20000))
    assert result is not None
    assert result.chat_ctx == 32768
    assert result.coloads == ()


def test_solve_matrix_squeeze_adopted(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=32768)
    rerank = make("r", role="rerank", context_length=32768)
    _fake_vram(chat, (4000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.5, 100))
    # High RAG caps so the baseline really serves 32768 and the squeeze
    # (down to coload_min_ctx) has something to yield.
    result = _solve_matrix_context(
        [chat, embed, rerank], embed, rerank,
        fit_bin="unused", vram_total=64 * 1024, spare=None,
        profiles=profiles,
        knobs=MatrixKnobs(embed_context=32768, rerank_context=32768))
    assert result is not None
    # baseline: rnk+emb at 32768 eat 33968; chat -> 22528. squeezed to 20480:
    # chat -> 32768. gain 10240 >= 4096 -> adopted.
    assert result.squeeze is True
    assert (result.embed_ctx, result.rerank_ctx) == (20480, 20480)
    assert result.chat_ctx == 32768


def test_coload_on_cpu_costs_zero(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", device="cpu", context_length=8192)
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    result = _solve_matrix_context(
        [chat, embed, rerank, s2t], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192))
    assert result is not None
    assert result.coloads == (("s2t", 0),)


def make(stem, **fm):
    """Minimal Model-like stub for the solve tests (no filesystem)."""
    from llama_packer.model import Model
    from pathlib import Path
    import tempfile
    d = tempfile.mkdtemp()
    (Path(d) / f"{stem}.gguf").write_bytes(b"x")
    m = Model(Path(d) / f"{stem}.md", {"name": stem, "context_length": 32768,
                                       **fm})
    return m


# ── plan(): demotion, squeeze clamp, coload flags ─────────────────────────


def test_plan_threads_matrix_result_into_variants(profiles, monkeypatch):
    chat = make("chat", role="chat", capabilities=["tools"])
    embed = make("e", role="embeddings", context_length=32768)
    rerank = make("r", role="rerank", context_length=32768)
    s2t = make("s2t", role="s2t", vram_mb=640, context_length=8192)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.5, 100))
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None

    # calc_ctx: echo the design_ctx the planner proposes, else the model's own.
    def fake_calc_ctx(*a, **kw):
        return kw.get("design_ctx") or 32768

    for m in (chat, embed, rerank, s2t):
        m.vram.calc_ctx = fake_calc_ctx

    planner = Planner([chat, embed, rerank, s2t], profiles, fit_bin="unused",
                      vram_total=64 * 1024,
                      matrix_cfg={"min_chat_ctx": 8192,
                                  "embed_context": 32768,
                                  "rerank_context": 32768},
                      embed_model=embed, rerank_model=rerank)
    plan = planner.plan()

    assert planner.matrix_result is not None
    # squeeze adopted (chat 22528 -> 32768), RAG entries served at 20480.
    assert planner.matrix_result.squeeze
    assert plan["e"][0].ctx_size == 20480
    assert plan["r"][0].ctx_size == 20480
    # chat solved below tools_min_ctx (131072) -> demoted.
    assert plan["chat"][0].tools_demoted is True
    # included co-load flagged.
    assert [v.coload for v in plan["s2t"]] == [True]

    config = emit_config([chat, embed, rerank, s2t], plan, profiles, TVARS)
    chat_entry = config["models"]["chat"]
    assert chat_entry["capabilities"]["tools"] is False
    assert chat_entry["metadata"]["tools_demoted"] is True
    assert config["models"]["s2t"]["metadata"]["ctx_size"] == 8192
    assert config.coload_stems == ["s2t"]


def test_plan_no_matrix_undemoted(profiles):
    chat = make("chat", role="chat", capabilities=["tools"])
    chat.vram.calc_ctx = lambda *a, **k: 32768
    planner = Planner([chat], profiles, fit_bin="unused", vram_total=64 * 1024)
    plan = planner.plan()
    assert plan["chat"][0].tools_demoted is False
    config = emit_config([chat], plan, profiles, TVARS)
    assert config["models"]["chat"]["capabilities"]["tools"] is True
    assert "tools_demoted" not in config["models"]["chat"]["metadata"]


def test_coload_estimate_gets_headroom(profiles):
    # No pin, no fit-params: the file-size + buffer estimate is padded by
    # estimate_headroom (1.25x default) so a bad guess errs toward reserving.
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", backend="whisper-server",
               context_length=8192)
    s2t.gguf_path.write_bytes(b"x" * (1024 * 1024))  # 1 MB file
    _fake_vram(chat, (8000, 0.4, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    result = _solve_matrix_context(
        [chat, embed, rerank, s2t], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192))
    assert result is not None
    # overhead = (1 + 100) * 1.25 = 126
    assert result.coloads == (("s2t", 126),)


# ── unestimable models: conservative bound, never a crash ─────────────────


def test_plan_unestimable_chat_floor_and_metadata(make_model, profiles):
    from llama_packer.consts import ESTIMATE_ERROR_REASON

    # No sidecar context_length: the floor cascade (no pin, no tools) is
    # half the design context.
    m = make("ue")
    del m.frontmatter["context_length"]
    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["ue"]
    assert len(variants) == 1
    v = variants[0]
    assert v.ctx_size == 16384
    assert v.estimate_error == ESTIMATE_ERROR_REASON
    config = emit_config([m], {"ue": variants}, profiles, TVARS)
    meta = config["models"]["ue"]["metadata"]
    assert meta["estimated"] is False
    assert meta["estimate_error"] == ESTIMATE_ERROR_REASON
    assert meta["ctx_size"] == 16384


def test_matrix_unestimable_chat_rides_free(profiles, monkeypatch):
    from llama_packer import writer

    good = make("good", role="chat")
    bad = make("bad", role="chat")
    embed = make("e", role="embeddings")
    for m in (good, embed):
        _fake_vram(m, (1000, 0.5, 100))
    bad.vram.effective_static = lambda *a, **k: None
    bad.vram.fit_params_static = lambda *a, **k: None

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 8192

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    planner = Planner([good, bad, embed], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=embed)
    plan = planner.plan()
    # The unestimable chat model stays in the shared solve but rides free:
    # zero quad — no expansion of the set's measured allocation.
    assert len(captured["chat_models"]) == 2
    assert captured["chat_models"][1][1:5] == (0, 0.0, 0.0, 0)
    assert planner.synthetic_quads["bad"] == (0, 0.0, 0.0, 0)
    assert bad.vram.unestimated_reason is not None
    # It serves at the shared solved context and is flagged.
    v = plan["bad"][0]
    assert v.ctx_size == 8192
    assert v.estimate_error is not None
    # The measurable chat model plans normally at the same shared ctx.
    assert plan["good"][0].ctx_size == 8192
    assert plan["good"][0].estimate_error is None


def test_matrix_unestimable_embed_clamped_and_flagged(profiles, monkeypatch):
    from llama_packer import writer

    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=32768)
    rerank = make("r", role="rerank", context_length=32768)
    _fake_vram(chat, (8000, 0.5, 500))
    for m in (embed, rerank):
        m.vram.effective_static = lambda *a, **k: None
        m.vram.fit_params_static = lambda *a, **k: None

    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return 12000

    monkeypatch.setattr(writer, "solve_matrix_ctx", fake_solver)

    planner = Planner([chat, embed, rerank], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"sets": {}},
                      embed_model=embed, rerank_model=rerank)
    plan = planner.plan()
    # Riders are charged nothing extra (zero quads) and serve at the
    # configured RAG cap (4096), not their design context.
    assert captured["embed_params"] == (0, 0.0, 0.0, 0)
    assert captured["rerank_params"] == (0, 0.0, 0.0, 0)
    assert captured["embed_ctx"] == 4096
    assert captured["rerank_ctx"] == 4096
    assert "e" in planner.synthetic_quads and "r" in planner.synthetic_quads
    for stem in ("e", "r"):
        v = plan[stem][0]
        assert v.ctx_size == 4096
        assert v.estimate_error is not None


def test_solve_matrix_aborts_when_nothing_measurable(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings")
    chat.vram.effective_static = lambda *a, **k: None
    embed.vram.fit_params_static = lambda *a, **k: None
    result = _solve_matrix_context(
        [chat], embed, embed,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles)
    assert result is None
