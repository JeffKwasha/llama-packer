# tests/test_mem_tag.py
"""Memory-allocation description tags: formatting, entry emission, planning."""

from __future__ import annotations

from conftest import entry_for, only_entry
from llama_packer.profiles import Profiles
from llama_packer.writer import (
    _build_entry,
    _fmt_mib,
    _with_mem_tag,
    emit_config,
    format_mem_tag,
    Planner,
)

TVARS = {"llama_bin": "/opt/llama-server"}


def _entry(make_model, stem="m", mem_vram=None, mem_ram=None, **frontmatter):
    model = make_model(stem, **frontmatter)
    return _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars=dict(TVARS),
        context_length=32768,
        ctx_size=8192,
        mem_vram_mib=mem_vram,
        mem_ram_mib=mem_ram,
    )


# ── units ──


def test_fmt_mib_gb_and_mb():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': a size is printed in GB or MB, and a small but nonzero value stays visible.
    assert _fmt_mib(9318.0) == "9.1GB"
    assert _fmt_mib(1024.0) == "1.0GB"
    assert _fmt_mib(64.4) == "64MB"
    assert _fmt_mib(0.0) == "0MB"
    # Small but nonzero stays visible (never "0.0GB").
    assert _fmt_mib(0.4) == "0MB"
    assert _fmt_mib(1.2 * 1024) == "1.2GB"


# ── tag grammar ──


def test_format_gpu_fit():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': a GPU entry is tagged with its VRAM and its RAM.
    assert format_mem_tag(9318.0, 1228.0) == "[VRAM 9.1GB RAM 1.2GB]"


def test_format_gpu_small_ram_uses_mb():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': a small RAM figure is printed in MB.
    assert format_mem_tag(9318.0, 64.0) == "[VRAM 9.1GB RAM 64MB]"


def test_format_cpu_only_is_ram_only():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': a CPU-resident entry is tagged with RAM only.
    assert format_mem_tag(None, 2355.0) == "[RAM 2.3GB]"


def test_format_verbatim_no_capping():
    # The decided number is reported even when physically impossible —
    # the tag never re-solves or caps.
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': the tag reports the decided numbers; it never re-solves or caps them.
    assert format_mem_tag(73.0 * 1024, 512.0) == "[VRAM 73.0GB RAM 512MB]"


def test_format_reserved_spill_ssd_grammar():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': a spilling entry names the VRAM it keeps, the RAM it spills to and the SSD.
    tag = format_mem_tag(22.0 * 1024, 512.0, spill_mib=1.2 * 1024, ssd_mib=50.0 * 1024)
    assert tag == "[VRAM 22.0GB + 1.2GB RAM 512MB SSD 50.0GB]"


# ── description composition ──


def test_with_mem_tag_appends_and_idempotent():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': the tag is appended, and re-tagging replaces it instead of doubling it.
    first = _with_mem_tag("Abliterated Gemma 3 12B.", "[VRAM 9.1GB RAM 1.2GB]")
    assert first == "Abliterated Gemma 3 12B. [VRAM 9.1GB RAM 1.2GB]"
    # Re-tagging (e.g. description copied back into a sidecar) replaces.
    second = _with_mem_tag(first, "[VRAM 9.2GB RAM 1.2GB]")
    assert second == "Abliterated Gemma 3 12B. [VRAM 9.2GB RAM 1.2GB]"
    assert second.count("[VRAM") == 1


def test_with_mem_tag_no_description():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': with no description the tag is the whole description.
    assert _with_mem_tag(None, "[RAM 2.3GB]") == "[RAM 2.3GB]"


def test_with_mem_tag_none_passthrough():
    # Validates: docs/plans/memory-allocation-tags.md, 'v1 contract (shipped)': no tag leaves the description alone.
    assert _with_mem_tag("text", None) == "text"
    assert _with_mem_tag(None, None) is None


# ── entry emission ──


def test_entry_description_gets_tag(make_model):
    # Validates: SPEC.md, 'Output Format': the entry description carries the memory tag.
    _, entry = _entry(
        make_model, description="Some model.", mem_vram=9318.0, mem_ram=64.0
    )
    assert entry["description"] == "Some model. [VRAM 9.1GB RAM 64MB]"


def test_entry_cpu_description_ram_only(make_model):
    # Validates: SPEC.md, 'Output Format': a CPU-resident entry is tagged with RAM only.
    _, entry = _entry(
        make_model,
        device="cpu",
        description="CPU model.",
        mem_vram=None,
        mem_ram=2355.0,
    )
    assert entry["description"] == "CPU model. [RAM 2.3GB]"


def test_entry_no_tag_without_mem(make_model):
    # Existing behavior preserved: no numbers → description untouched.
    # Validates: SPEC.md, 'Output Format': without measured memory the description is untouched.
    _, entry = _entry(make_model, description="Some model.")
    assert entry["description"] == "Some model."


def test_entry_no_tag_when_unestimated(make_model):
    # Validates: SPEC.md, 'Output Format': an entry whose VRAM could not be measured is not tagged with numbers.
    model = make_model("m", description="Some model.")
    _, entry = _build_entry(
        model,
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars=dict(TVARS),
        context_length=32768,
        ctx_size=4096,
        estimate_error="no VRAM estimate available",
        mem_vram_mib=9318.0,
        mem_ram_mib=64.0,
    )
    assert entry["description"] == "Some model."


def test_entry_tag_without_sidecar_description(make_model):
    # Validates: SPEC.md, 'Output Format': without a sidecar description the tag is the whole description.
    _, entry = _entry(make_model, mem_vram=9318.0, mem_ram=64.0)
    # make_model sets no description → the tag is the whole description.
    assert entry["description"] == "[VRAM 9.1GB RAM 64MB]"


def test_entry_vram_note_appended_after_mem_tag(make_model):
    # Validates: SPEC.md, 'VRAM gating and reservation notes': the
    # over-budget note follows the memory tag.
    _, entry = _build_entry(
        make_model("mn", description="Some model."),
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars=dict(TVARS),
        context_length=32768,
        ctx_size=8192,
        mem_vram_mib=9318.0,
        mem_ram_mib=64.0,
        vram_note="(over configured VRAM limits — serving at 8,192 of "
                  "32,768 design tokens)",
    )
    assert entry["description"] == (
        "Some model. [VRAM 9.1GB RAM 64MB] (over configured VRAM limits "
        "— serving at 8,192 of 32,768 design tokens)"
    )


def test_entry_vram_note_without_mem_tag_or_description(make_model):
    # The note alone still yields a description (entry is visibly flagged).
    # Validates: SPEC.md, 'VRAM gating and reservation notes': the note
    # alone still yields a description, so the entry is visibly flagged.
    _, entry = _build_entry(
        make_model("bare"),
        parallel=1,
        cache_type="q8_0",
        profiles_group=[("default", {})],
        profiles_defaults={},
        template_vars=dict(TVARS),
        context_length=32768,
        ctx_size=8192,
        vram_note="(over configured VRAM limits)",
    )
    assert entry["description"] == "(over configured VRAM limits)"


# ── planner → emit end to end ──


def _scripted(model, ctx, quad_by_mmproj, ctx_by_mmproj=None):
    """Script both solves: ctx via calc_ctx, quads via effective_static.

    *ctx_by_mmproj* lets a test give the text-only rung a different (larger,
    as shedding a projection frees VRAM) context than the vision one.
    """

    def fake_ctx(vram_total_mb, include_mmproj=True, **kw):
        if ctx_by_mmproj is not None:
            return ctx_by_mmproj[bool(include_mmproj)]
        return ctx

    def fake_static(
        fit_bin,
        cache_type="q8_0",
        design_ctx=None,
        include_mmproj=True,
        llama_args="",
        allow_cpu=False,
    ):
        return quad_by_mmproj[bool(include_mmproj)]

    model.vram.calc_ctx = fake_ctx
    model.vram.effective_static = fake_static
    for flag in (True, False):
        view = model.view_for(flag)
        if view is not model:
            view.vram.calc_ctx = fake_ctx
            view.vram.effective_static = fake_static


def _profiles():
    return Profiles(
        {"defaults": {"cache_type": "q8_0", "parallel": 1}, "profiles": {"default": {}}}
    )


def test_planner_mem_numbers_and_tag(make_model, tmp_path):
    # Validates: SPEC.md, 'Output Format': the emitted VRAM number is weights + compute + context KV + slots.
    m = make_model("m", parallel=1, context_length=32768)
    (tmp_path / "m.gguf").write_bytes(b"x" * 2 * 1024 * 1024)
    quad = (1, 0.001, 10, 20)  # model_mib=1: 2MB file leaves 1MB residue
    _scripted(m, 8192, {True: quad, False: quad})

    planner = Planner([m], _profiles(), fit_bin="unused", vram_total=48 * 1024)
    variants = planner.plan()["m"]
    assert len(variants) == 1
    v = variants[0]
    # TOTAL attributable: weights + compute + ctx KV + slots.
    assert v.mem_vram_mib == (1 + 20 + 0.001 * 8192 + 10)
    assert v.mem_ram_mib == 1.0

    config = emit_config([m], {"m": variants}, _profiles(), TVARS)
    desc = entry_for(config, "m")["description"]
    assert desc.endswith("[VRAM 39MB RAM 1MB]")


def test_planner_cpu_model_ram_only_tag(make_model):
    # Validates: SPEC.md, 'Output Format': a CPU-resident model is sized in RAM and carries no VRAM number.
    m = make_model("c", device="cpu", parallel=1, context_length=32768)
    quad = (5000, 0.5, 100, 500)
    _scripted(m, 32768, {True: quad, False: quad})

    planner = Planner([m], _profiles(), fit_bin="unused", vram_total=48 * 1024)
    variants = planner.plan()["c"]
    v = variants[0]
    assert v.mem_vram_mib is None
    assert v.mem_ram_mib == (5000 + 500 + 0.5 * 32768 + 100)

    config = emit_config([m], {"c": variants}, _profiles(), TVARS)
    desc = entry_for(config, "c")["description"]
    assert desc.endswith("[RAM 21.5GB]")
