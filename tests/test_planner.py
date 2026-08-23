# tests/test_planner.py
"""Planner / Profiles / emit_config seams: vision variants, grouping, clamps."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llama_packer.profiles import Profiles
from llama_packer.writer import Planner, emit_config, _solve_matrix_context

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
    m = _vision_model(tmp_path, make_model, "vis")
    _scripted_ctx(m, {True: 4096, False: 65536})

    planner = Planner([m], profiles, fit_bin="unused", vram_total=48 * 1024,
                      min_context=131072)
    variants = planner.plan()["vis"]
    assert len(variants) == 1
    v = variants[0]
    assert v.include_mmproj is False
    # text ctx clamped to the sidecar's max trained context (32768)
    assert v.ctx_size == 32768
    assert v.vision_ctx == 4096

    config = emit_config([m], {"vis": variants}, profiles, TVARS)
    # Auto-dropped main entry is renamed <id>-text (bare id always = vision).
    assert set(config["models"]) == {"vis-text", "vis-vision-4k"}
    main = config["models"]["vis-text"]
    vision = config["models"]["vis-vision-4k"]
    assert "--mmproj" not in main["cmd"]
    assert main["name"].endswith("[text]")
    assert main["metadata"]["mmproj_skipped"] is True
    assert main["capabilities"]["in"] == ["text"]
    assert "--mmproj" in vision["cmd"]
    assert vision["name"].endswith("[vision 4k]")


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

    chat_ctx = _solve_matrix_context(
        [chat_a, chat_b, embed, cpu_chat], embed, embed,
        fit_bin="unused", vram_total=48 * 1024, spare="1G",
        profiles=profiles, baseline_mb=0, drop_stems={"chat_b"},
    )
    assert chat_ctx == 24576
    # Only the two real chat models enter the shared budget;
    # chat_b's mmproj is omitted via drop_stems.
    assert seen == {True: 1, False: 1}
    assert len(captured["chat_models"]) == 2
    assert captured["embed_params"] == FP_TRIPLE
