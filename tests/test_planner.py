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


def _scripted_ctx(model, by_mmproj):
    """Replace calc_ctx with a fake keyed on include_mmproj."""
    def fake(vram_total_mb, *, fit_bin=None, parallel=1, spare_mb=0,
             include_mmproj=True, baseline_mb=0, cache_type="q8_0",
             design_ctx=None, **kw):
        return by_mmproj[bool(include_mmproj)]
    model.vram.calc_ctx = fake


def _vision_model(tmp_path, make_model, name):
    """Model with an mmproj companion; file must exist before construction."""
    (tmp_path / f"{name}-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    return make_model(name, mmproj=f"{name}-mmproj.gguf")


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
    m = make_model("vis", mmproj="vis-mmproj.gguf", context_length=262144)
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
    m = make_model("vis", mmproj="vis-mmproj.gguf", context_length=262144)
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


FP_TRIPLE = (1000.0, 0.5, 100.0)


def test_solve_matrix_excludes_roles_cpu_and_threads_drop_stems(
        make_model, profiles, monkeypatch):
    from llama_packer import writer

    chat_a = make_model("chat_a", role="chat")
    chat_b = make_model("chat_b", role="chat")
    embed = make_model("e", role="embeddings")
    cpu_chat = make_model("cpu", device="cpu")

    seen = {}

    def fake_effective_static(fit_bin, cache_type="q8_0", parallel=1,
                              design_ctx=None, include_mmproj=True, **kw):
        seen[include_mmproj] = seen.get(include_mmproj, 0) + 1
        return FP_TRIPLE

    def fake_fit_params_static(fit_bin, cache_type="q8_0", parallel=1, **kw):
        return SimpleNamespace(model_mib=FP_TRIPLE[0], ctx_factor=FP_TRIPLE[1],
                               compute_mib=FP_TRIPLE[2])

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
    assert captured["embed_params"] == FP_TRIPLE


# ── matrix knobs ──────────────────────────────────────────────────────────


def test_matrix_knobs_defaults_and_overrides():
    k = MatrixKnobs.from_cfg(None)
    assert (k.min_chat_ctx, k.tools_min_ctx, k.coload_min_ctx,
            k.ctx_gain_min, k.estimate_headroom) == (
        65536, 131072, 20480, 4096, 1.25)
    k = MatrixKnobs.from_cfg({"min_chat_ctx": 32000, "estimate_headroom": 1.5})
    assert k.min_chat_ctx == 32000
    assert k.estimate_headroom == 1.5


def test_matrix_knobs_invalid_values_warn_and_default(caplog):
    with caplog.at_level(logging.WARNING):
        k = MatrixKnobs.from_cfg({"min_chat_ctx": -1, "coload_min_ctx": "big",
                                  "estimate_headroom": 0.5})
    assert k.min_chat_ctx == 65536
    assert k.coload_min_ctx == 20480
    assert k.estimate_headroom == 1.25
    assert "min_chat_ctx" in caplog.text
    assert "estimate_headroom" in caplog.text


# ── opportunistic co-load pass ────────────────────────────────────────────


def _fake_vram(m, triple):
    m.vram.effective_static = lambda *a, **k: triple
    m.vram.fit_params_static = lambda *a, **k: SimpleNamespace(
        model_mib=triple[0], ctx_factor=triple[1], compute_mib=triple[2],
        source="fit-params")


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
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None
    img.vram.effective_static = lambda *a, **k: (40000, 0.0, 0)
    img.vram.fit_params_static = lambda *a, **k: None

    result = _solve_matrix_context(
        [chat, embed, rerank, s2t, img], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=8192))
    assert result is not None
    # available = 49152-2048 = 47104; emb+rnk at 8192 = 2*(600+819) = 2838
    # baseline chat budget = 44266-8500 = 35766 -> capped 32768
    assert result.chat_ctx == 32768
    # s2t (640 MB) fits; image (40000 MB) leaves a negative chat budget.
    assert result.coloads == (("s2t", 640),)


def test_solve_matrix_floor_blocks_all_coloads(profiles):
    chat = make("chat", role="chat")
    embed = make("e", role="embeddings", context_length=8192)
    rerank = make("r", role="rerank", context_length=8192)
    s2t = make("s2t", role="s2t", vram_mb=640, context_length=8192)
    _fake_vram(chat, (8000, 1.0, 500))
    for m in (embed, rerank):
        _fake_vram(m, (500, 0.1, 100))
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None

    from llama_packer.writer import MatrixKnobs
    result = _solve_matrix_context(
        [chat, embed, rerank, s2t], embed, rerank,
        fit_bin="unused", vram_total=48 * 1024, spare=None,
        profiles=profiles, knobs=MatrixKnobs(min_chat_ctx=24576))
    assert result is not None
    # chat_budget = 47104-2838-8500 = 35766 -> ctx 35766 -> capped 32768? No:
    # factor 1.0 -> 35766 -> round 35840 -> capped 32768 >= floor -> s2t fits.
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
    img.vram.effective_static = lambda *a, **k: (20000, 0.0, 0)
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
    result = _solve_matrix_context(
        [chat, embed, rerank], embed, rerank,
        fit_bin="unused", vram_total=64 * 1024, spare=None,
        profiles=profiles)
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
    s2t.vram.effective_static = lambda *a, **k: (640, 0.0, 0)
    s2t.vram.fit_params_static = lambda *a, **k: None

    # calc_ctx: echo the design_ctx the planner proposes, else the model's own.
    def fake_calc_ctx(*a, **kw):
        return kw.get("design_ctx") or 32768

    for m in (chat, embed, rerank, s2t):
        m.vram.calc_ctx = fake_calc_ctx

    planner = Planner([chat, embed, rerank, s2t], profiles, fit_bin="unused",
                      vram_total=64 * 1024, matrix_cfg={"min_chat_ctx": 8192},
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
