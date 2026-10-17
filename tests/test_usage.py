# tests/test_usage.py
"""Demand-driven planning: the blessed ``usage:`` catalogue and the
capability/context ladder."""

from __future__ import annotations

import pytest

import llama_packer.writer as writer
from llama_packer.writer import (
    FALLBACK,
    KNOWN_CAPABILITIES,
    UsageCatalogue,
    UsageDemand,
    _parse_ctx_size,
    plan_ladder,
    resolve_usage,
)


# ── context size parsing ───────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    (131072, 131072),
    ("128k", 131072),
    ("128K", 131072),
    ("1M", 1048576),
    ("1m", 1048576),
    (" 64k ", 65536),
    (65536.0, 65536),
])
def test_parse_ctx_size_accepts_ints_and_suffixes(raw, expected):
    assert _parse_ctx_size(raw) == expected


@pytest.mark.parametrize("raw", [0, -1, "abc", "", None, True, [], {}])
def test_parse_ctx_size_rejects_junk(raw):
    assert _parse_ctx_size(raw) is None


# ── the catalogue ──────────────────────────────────────────────────────────


def test_catalogue_parses_full_spec():
    cat = UsageCatalogue.from_cfg({
        "coding": {"min_context": "128k", "min_parallel": 1,
                   "sacrifice": ["vision", "draft"]},
        "chat": {"min_context": "64k", "min_parallel": 2,
                 "sacrifice": ["draft", "vision"]},
        "qa": {"min_context": "32k", "min_parallel": 2, "require": ["vision"]},
    })
    assert cat.order == ("coding", "chat", "qa")
    coding = _d(cat, "coding")
    assert (coding.min_context, coding.min_parallel) == (131072, 1)
    # sacrifice keeps the declared order -- it is a preference order
    assert coding.sacrifice == ("vision", "draft")
    assert _d(cat, "chat").sacrifice == ("draft", "vision")
    assert cat.get("qa").require == frozenset({"vision"})


def test_catalogue_int_shorthand_is_min_context():
    cat = UsageCatalogue.from_cfg({"chat": 65536})
    assert cat.get("chat").min_context == 65536
    assert cat.get("chat").min_parallel == 1


def test_catalogue_warns_and_drops_malformed_categories(caplog):
    with caplog.at_level("WARNING"):
        cat = UsageCatalogue.from_cfg({
            "good": {"min_context": "32k"},
            "not-a-mapping": "garbage",
            "bad-ctx": {"min_context": "abc"},
            "bad-par": {"min_parallel": 0},
        })
    assert cat.order == ("good", "bad-ctx", "bad-par")
    assert "not a mapping" in caplog.text
    # an unusable minimum is dropped, the category survives with its default
    assert cat.get("bad-ctx").min_context == 0
    assert cat.get("bad-par").min_parallel == 1


def test_catalogue_warns_on_unknown_capability(caplog):
    with caplog.at_level("WARNING"):
        cat = UsageCatalogue.from_cfg(
            {"qa": {"require": ["vision", "telepathy"]}})
    assert "unknown capabilities" in caplog.text
    assert "telepathy" in caplog.text
    assert cat.get("qa").require == frozenset({"vision"})


def test_catalogue_empty_config_is_falsy():
    assert not UsageCatalogue.from_cfg(None)
    assert not UsageCatalogue.from_cfg({})


def test_known_capabilities_are_the_ladder_axes():
    assert KNOWN_CAPABILITIES == frozenset({"vision", "draft"})


# ── tag resolution: only blessed tags count ────────────────────────────────


def test_resolve_usage_accepts_blessed_tags():
    cat = UsageCatalogue.from_cfg({"coding": {"min_context": "128k"},
                                   "chat": {"min_context": "64k"}})
    got = resolve_usage(["coding", "chat"], cat)
    assert [d.name for d in got] == ["coding", "chat"]


def test_resolve_usage_dedupes_repeats():
    cat = UsageCatalogue.from_cfg({"chat": {"min_context": "64k"}})
    assert [d.name for d in resolve_usage(["chat", "chat"], cat)] == ["chat"]


def test_resolve_usage_ignores_unblessed_tag_with_warning(caplog):
    cat = UsageCatalogue.from_cfg({"chat": {"min_context": "64k"}})
    with caplog.at_level("WARNING"):
        got = resolve_usage(["chat", "code"], cat)
    assert [d.name for d in got] == ["chat"]
    assert "not declared" in caplog.text
    assert "code" in caplog.text
    # the warning names what *is* blessed, so the operator can see the fix
    assert "chat" in caplog.text


def test_resolve_usage_accepts_bare_string():
    cat = UsageCatalogue.from_cfg({"chat": {"min_context": "64k"}})
    assert [d.name for d in resolve_usage("chat", cat)] == ["chat"]


def test_resolve_usage_no_tags_means_no_demands(caplog):
    cat = UsageCatalogue.from_cfg({"chat": {"min_context": "64k"}})
    assert resolve_usage(None, cat) == ()
    with caplog.at_level("WARNING"):
        assert resolve_usage(42, cat) == ()


# ── the ladder ─────────────────────────────────────────────────────────────


def _cat(**specs) -> UsageCatalogue:
    return UsageCatalogue.from_cfg(specs)


def _d(cat: UsageCatalogue, name: str) -> UsageDemand:
    """Narrowing accessor: a declared category always exists."""
    demand = cat.get(name)
    assert demand is not None, f"category {name!r} not declared"
    return demand


def _demands(cat: UsageCatalogue, *names: str) -> tuple[UsageDemand, ...]:
    return tuple(_d(cat, n) for n in names)


def test_ladder_keeps_full_rung_when_shedding_buys_nothing():
    # A model already at its design cap reclaims no context by shedding, so
    # it must not get a redundant text-only twin.
    cat = _cat(coding={"min_context": "128k", "min_parallel": 1})
    rungs = plan_ladder(lambda shed: (262144, 1), (_d(cat, "coding"),),
                        frozenset({"vision", "draft"}))
    assert len(rungs) == 1
    assert rungs[0].shed == frozenset()
    assert rungs[0].ctx == 262144


def test_ladder_sheds_the_capability_that_actually_buys_context():
    cat = _cat(coding={"min_context": "128k", "min_parallel": 1})
    # full is short of the demand; dropping vision clears it; the draft is
    # then worthless because context is already maxed.
    def afford(shed):
        if "vision" in shed:
            return (262144, 1)
        if "draft" in shed:
            return (131072, 1)
        return (98304, 1)

    rungs = plan_ladder(afford, (_d(cat, "coding"),),
                        frozenset({"vision", "draft"}))
    assert [r.shed for r in rungs] == [frozenset({"vision"})]
    assert rungs[0].ctx == 262144


def test_ladder_never_sheds_a_required_capability():
    cat = _cat(qa={"min_context": "32k", "require": ["vision"]})
    def afford(shed):
        return (131072, 1) if shed else (98304, 1)

    rungs = plan_ladder(afford, (_d(cat, "qa"),),
                        frozenset({"vision", "draft"}))
    # the full rung and the draft-shed rung survive; nothing sheds vision
    assert [r.shed for r in rungs] == [frozenset(), frozenset({"draft"})]
    assert all("vision" not in r.shed for r in rungs)


def test_ladder_drops_rung_satisfying_nothing():
    cat = _cat(coding={"min_context": "128k", "min_parallel": 1})
    rungs = plan_ladder(lambda shed: (16384, 1), (_d(cat, "coding"),),
                        frozenset({"vision", "draft"}))
    assert rungs == ()


def test_ladder_drops_rung_gaining_nothing_over_a_more_capable_one():
    # Every rung lands on 256k, so only the most capable one is worth an
    # entry: the rest are strictly worse duplicates (same context, fewer
    # capabilities).
    cat = _cat(coding={"min_context": "128k", "min_parallel": 1})
    rungs = plan_ladder(lambda shed: (262144, 1), (_d(cat, "coding"),),
                        frozenset({"vision", "draft"}))
    assert [r.shed for r in rungs] == [frozenset()]


def test_ladder_min_parallel_is_a_floor_too():
    cat = _cat(chat={"min_context": "32k", "min_parallel": 4})
    # ctx is ample but only 2 slots: the demand is unmet, so no rung stands
    rungs = plan_ladder(lambda shed: (262144, 2), (_d(cat, "chat"),),
                        frozenset({"vision"}))
    assert rungs == ()


def test_ladder_satisfies_several_categories_with_one_entry():
    cat = _cat(coding={"min_context": "128k", "min_parallel": 1},
               chat={"min_context": "64k", "min_parallel": 2},
               qa={"min_context": "32k", "min_parallel": 2})
    demands = tuple(cat.get(n) for n in cat.order)
    rungs = plan_ladder(lambda shed: (262144, 2), demands,
                        frozenset({"vision", "draft"}))
    assert len(rungs) == 1
    # one entry at 256k/2 covers all three
    assert {d.name for d in rungs[0].demands} == {"coding", "chat", "qa"}


def test_ladder_without_demands_keeps_every_distinct_rung():
    # Back-compat path: a model with no `usage:` tag is governed by the
    # global min_context floor instead, so no demand filtering applies and
    # every distinct rung is kept.  Shedding vision frees VRAM, so it must
    # raise the affordable context.
    def afford(shed):
        return (196608, 1) if "vision" not in shed else (262144, 1)

    rungs = plan_ladder(afford, (), frozenset({"vision"}))
    assert [r.shed for r in rungs] == [frozenset(), frozenset({"vision"})]


def test_ladder_respects_the_most_protective_demand():
    # One category wants the draft kept, another wants vision kept; the
    # merged order sheds neither until both are gone.
    cat = _cat(a={"sacrifice": ["vision", "draft"]},
               b={"sacrifice": ["draft", "vision"]})
    from llama_packer.writer import _shed_order
    order = _shed_order((_d(cat, "a"), _d(cat, "b")), frozenset({"vision", "draft"}))
    # 'vision' sorts first because category `a` prefers to keep the draft
    assert order == ("vision", "draft")


def test_rung_label_is_readable():
    cat = _cat(coding={"min_context": "32k"})
    def afford(shed):
        if "vision" in shed and "draft" in shed:
            return (131072, 1)
        if "vision" in shed:
            return (98304, 1)
        return (65536, 1)

    labels = {r.label for r in plan_ladder(afford, (_d(cat, "coding"),),
                                           frozenset({"vision", "draft"}))}
    assert "full" in labels
    assert "vision" in labels
    assert "draft+vision" in labels


# ── per-set co-resident overhead ────────────────────────────────────────────

from llama_packer.writer import (  # noqa: E402
    MatrixKnobs,
    Planner,
    set_coresident_vars,
    set_overhead_mb,
    _split_top,
)

_MB = {"emb": 749, "rnk": 1002, "rnk2": 5766, "stt": 1898, "tts": 1204,
       "s2t": 2814, "s2t2": 1914}
_CH = frozenset(f"c{i}" for i in range(1, 39))
_CO = frozenset({"s2t", "s2t2"})


def test_split_top_respects_parentheses():
    assert _split_top("a & b & c", "&") == ["a", "b", "c"]
    assert _split_top("a | b & c", "|") == ["a", "b & c"]
    # a term fully wrapped in parens yields the bare term
    assert _split_top("(a | b) & c", "&") == ["a | b", "c"]
    # outer parens are dropped so consumers see var names
    assert _split_top("((a))", "&") == ["a"]
    # a `|` nested inside parens is not a top-level alternative, and the
    # outer parens are not stripped because they belong to sub-terms
    assert _split_top("(a | b) & c", "|") == ["(a | b) & c"]


@pytest.mark.parametrize("expr,expected", [
    ("__CHAT_VARS__ & emb & rnk", 749 + 1002),
    ("__CHAT_VARS__ & emb & rnk2", 749 + 5766),
    ("__CHAT_VARS__ & stt", 1898),
    ("__CHAT_VARS__ & tts", 1204),
    ("__CHAT_VARS__", 0),
])
def test_set_overhead_sums_coresidents(expr, expected):
    assert set_overhead_mb(expr, _MB, _CH, _CO) == expected


def test_set_overhead_charges_priciest_or_branch_only():
    # emb and rnk2 are alternatives -- only one is ever resident, and the
    # entry must survive the heavier one.
    assert set_overhead_mb("__CHAT_VARS__ & (emb | rnk2)", _MB, _CH, _CO) == 5766
    assert set_overhead_mb("((c1 | c2)) & emb & (rnk | rnk2)",
                           _MB, _CH, _CO) == 749 + 5766


def test_set_overhead_sums_coloads():
    # co-loads genuinely stack with each other, unlike OR branches
    assert set_overhead_mb("__CHAT_VARS__ & __COLOAD_VARS__",
                           _MB, _CH, _CO) == 2814 + 1914


def test_set_overhead_ignores_explicit_chat_vars():
    assert set_overhead_mb("(c1 | c2) & emb", _MB, _CH, _CO) == 749
    assert set_overhead_mb("c1 & emb & stt", _MB, _CH, _CO) == 749 + 1898


def test_set_coresident_vars_names_the_residents():
    assert set_coresident_vars("__CHAT_VARS__ & emb & rnk", _CH, _CO) == \
        frozenset({"emb", "rnk"})
    assert set_coresident_vars("__CHAT_VARS__ & __COLOAD_VARS__", _CH, _CO) == _CO
    assert set_coresident_vars("__CHAT_VARS__", _CH, _CO) == frozenset()


def test_set_overhead_ignores_unknown_vars():
    # an unresolvable var charges nothing rather than exploding the solve
    assert set_overhead_mb("__CHAT_VARS__ & mystery", _MB, _CH, _CO) == 0


# ── the id label ────────────────────────────────────────────────────────────

from llama_packer.writer import ctx_label  # noqa: E402


@pytest.mark.parametrize("ctx,expected", [
    (262144, "256k"),        # the user's worked example
    (139264, "136k"),
    (32768, "32k"),
    (4096, "4k"),
    (999424, "976k"),
    (1048762, "1M"),         # and the other worked example
    (1048576, "1M"),
    (2097152, "2M"),
])
def test_ctx_label(ctx, expected):
    assert ctx_label(ctx) == expected


def test_ctx_label_never_exceeds_four_characters():
    # A client scans the id list; an id that wanders past four characters
    # stops being comparable at a glance.
    for ctx in range(0, 4 * 1024 * 1024, 4096):
        assert len(ctx_label(ctx)) <= 4, ctx


def test_ctx_labels_below_the_mega_threshold_are_unique():
    # Exact contexts must not collide, or two entries would share an id.
    seen = {}
    for ctx in range(0, 1000 * 1024, 1024):
        lbl = ctx_label(ctx)
        assert lbl not in seen, (lbl, seen.get(lbl), ctx)
        seen[lbl] = ctx


# ── auto-parallel respects demand floors ────────────────────────────────────


def _auto_par(table, **kw):
    """Drive _auto_parallel against a scripted (parallel -> ctx) table."""
    from llama_packer.writer import MatrixKnobs as _MK
    pl = Planner.__new__(Planner)
    pl.knobs = _MK(auto_parallel_max=8, parallel_power=0.75)
    pl.vram_total = 48 * 1024
    pl.fit_bin = "unused"
    pl.memory_margin = 0.04
    pl.baseline_mb = 0
    pl.chat_ctx = 262144
    pl.profiles = _profiles_stub()
    pl.max_context = None
    pl._bounded_ctx = lambda *a, **k: table.get(k.get("parallel", 1), 32768)

    class _V:
        stem = "m"
        backend = "llama-server"

    return pl._auto_parallel(
        _V(), cache_type="q8_0", spare_mb=0, include_mmproj=False,
        cap_ctx=262144, pin_ctx=None, group_parallel=1, **kw)


def test_min_parallel_is_honoured_when_a_candidate_qualifies():
    # Every slot count clears the 64k chat floor, so a demand for 4 slots
    # must not be quietly served with fewer.
    table = {p: 262144 for p in range(1, 9)}
    p, ctx = _auto_par(table, floor=65536, min_parallel=4)
    assert p >= 4, (p, ctx)
    assert ctx >= 65536


def test_context_floor_outranks_min_parallel():
    # Two slots would drop under the demand's 64k floor, so the single wide
    # window is served instead. Concurrency is a preference; the context is
    # what the entry id advertises, and a client that cannot fit its prompt
    # gains nothing from a spare slot.
    table = {1: 262144, 2: 40000}
    p, ctx = _auto_par(table, floor=65536, min_parallel=2)
    assert (p, ctx) == (1, 262144), (p, ctx)


def test_demand_never_trades_below_its_context_floor():
    # The budget only reaches ~96k however few slots are used, so a 128k
    # coding demand is unreachable: no candidate qualifies and the solver
    # falls back rather than inventing a context. Dropping the model from
    # the sets it cannot serve is the admission pass's job, not this one's.
    table = {p: 98304 for p in range(1, 9)}
    p, ctx = _auto_par(table, floor=131072, min_parallel=1)
    assert (p, ctx) == (1, 98304), (p, ctx)


def _profiles_stub(**kw):
    from types import SimpleNamespace
    return SimpleNamespace(default_cache_type=kw.get("default_cache_type", "q8_0"),
                           default_parallel=1)


# ── id grammar: a stable task name + served capability flags ────────────────


from llama_packer.writer import (  # noqa: E402
    Variant,
    category_label,
    emit_config,
    flags_label,
    served_flags,
)
from llama_packer.profiles import Profiles  # noqa: E402

TVARS = {"llama_bin": "/opt/llama-server"}


def _id_cat() -> UsageCatalogue:
    return UsageCatalogue.from_cfg({
        "coding": {"min_context": "128k", "min_parallel": 1,
                   "sacrifice": ["vision"]},
        "chat": {"min_context": "64k", "min_parallel": 2},
        "qa": {"min_context": "32k", "min_parallel": 2, "require": ["vision"]},
    })


@pytest.mark.parametrize("caps,expected", [
    (["tools", "image", "video", "audio", "speech"], "tivas"),
    (["video", "tools"], "tv"),          # emission order, not input order
    (["image"], "i"),
    ([], ""),
    (None, ""),
])
def test_flags_label_subset_and_order(caps, expected):
    assert flags_label(caps) == expected


def test_category_label_picks_heaviest_satisfied():
    cat = _id_cat()
    demands = _demands(cat, "qa", "chat", "coding")
    assert category_label(demands, cat) == "coding"


def test_category_label_tie_breaks_by_declaration_order():
    cat = UsageCatalogue.from_cfg({"a": {"min_context": "64k"},
                                   "b": {"min_context": "64k"}})
    # b listed first but a is declared first — declaration order wins.
    assert category_label(_demands(cat, "b", "a"), cat) == "a"


def test_category_label_none_without_demands():
    assert category_label((), _id_cat()) is None


def test_served_flags_drops_image_and_demoted_tools(make_model, tmp_path):
    (tmp_path / "vis-mmproj.gguf").write_bytes(b"x")
    m = make_model("vis", capabilities=["tools", "image", "video"],
                   mmproj={"file": "vis-mmproj.gguf", "capabilities": ["image"]})
    assert served_flags(m, True, False) == "tiv"
    # no-mmproj serving sheds the image/video inputs the companion enabled
    assert served_flags(m, False, False) == "t"
    # tools demoted (served below tools_min_ctx) cannot be claimed
    assert served_flags(m, True, True) == "iv"


def _variant(ctx, usage):
    return Variant(parallel=1, cache_type="q8_0", spare_mb=0,
                   profiles_group=[("default", {})], ctx_size=ctx,
                   include_mmproj=True, usage=usage)


def _profiles_with_usage() -> Profiles:
    return Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "usage": {"coding": {"min_context": "128k", "min_parallel": 1},
                  "chat": {"min_context": "64k", "min_parallel": 2},
                  "qa": {"min_context": "32k", "min_parallel": 2}},
    })


def test_emit_id_names_heaviest_category_with_flags(make_model):
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    cat = UsageCatalogue.from_cfg(_profiles_with_usage().usage_cfg)
    usage = _demands(cat, "coding", "chat", "qa")
    config = emit_config([m], {"qwen38": [_variant(160000, usage)]},
                         _profiles_with_usage(), TVARS)
    assert list(config["models"]) == ["qwen38.coding.ti"]
    entry = config["models"]["qwen38.coding.ti"]
    assert entry["metadata"]["usage"] == ["coding", "chat", "qa"]
    assert entry["metadata"]["ctx_size"] == 160000
    assert "coding" in entry["name"] and "156k" in entry["name"]


def test_ids_stable_when_context_changes(make_model):
    # The whole point: a repack that moves the solve must not rename the id,
    # because opencode/hermes cache model ids at startup.
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    cat = UsageCatalogue.from_cfg(profiles.usage_cfg)
    usage = _demands(cat, "coding", "chat")
    a = emit_config([m], {"qwen38": [_variant(140000, usage)]}, profiles, TVARS)
    b = emit_config([m], {"qwen38": [_variant(155000, usage)]}, profiles, TVARS)
    assert list(a["models"]) == list(b["models"]) == ["qwen38.coding.ti"]


def test_untagged_gets_a_bare_id(make_model):
    """No category, no context segment -- just the model.

    Every model emits an id: an untagged one is served on the bare minimums
    and still reaches clients. The solved context deliberately stays out of
    the id; a repack that moved 182k to 180k must not rename the model out
    from under a client that cached it.
    """
    m = make_model("plain", context_length=262144)
    config = emit_config([m], {"plain": [_variant(131072, ())]},
                         _profiles_with_usage(), TVARS)
    assert list(config["models"]) == ["plain"]
    assert config["models"]["plain"]["metadata"]["ctx_size"] == 131072


def test_same_model_two_groups_disambiguated(make_model):
    # Two profile groups can land on the same category+flags; the first keeps
    # the stable name, the duplicate appends a discriminator rather than
    # colliding into a fatal error.
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    cat = UsageCatalogue.from_cfg(profiles.usage_cfg)
    usage = (cat.get("coding"),)
    variants = [_variant(150000, usage), _variant(140000, usage)]
    config = emit_config([m], {"qwen38": variants}, profiles, TVARS)
    assert list(config["models"]) == ["qwen38.coding.ti",
                                      "qwen38.coding.ti-q8_0-vis"]


def test_duplicate_ids_do_not_move_when_context_moves(make_model):
    """The suffix must survive a repack that only changes VRAM.

    It used to be the context label, which emit_config's own docstring rules
    out: opencode/hermes cache ids at startup, so a context-derived suffix
    repoints a cached id at a different serving on every repack.
    """
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    usage = (UsageCatalogue.from_cfg(profiles.usage_cfg).get("coding"),)
    first = emit_config([m], {"qwen38": [_variant(150000, usage),
                                         _variant(140000, usage)]},
                        profiles, TVARS)
    # Same two servings, both re-solved against a different budget.
    second = emit_config([m], {"qwen38": [_variant(300000, usage),
                                          _variant(280000, usage)]},
                         profiles, TVARS)
    assert list(first["models"]) == list(second["models"])


def test_category_slugged_for_the_id(make_model):
    # Operator category names must not leak invalid characters into the id
    # (opencode excludes '#', llama-swap needs routeable/YAML-safe keys).
    m = make_model("m", context_length=262144, capabilities=["tools"])
    profiles = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                         "usage": {"My Code!": {"min_context": "64k"}}})
    cat = UsageCatalogue.from_cfg(profiles.usage_cfg)
    v = _variant(80000, (cat.get("My Code!"),))
    config = emit_config([m], {"m": [v]}, profiles, TVARS)
    assert list(config["models"]) == ["m.my-code.t"]


def test_unusable_category_name_falls_back_to_a_bare_id(make_model):
    """A name that cannot survive slugging drops the segment, not the entry."""
    m = make_model("m", context_length=262144)
    profiles = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1},
                         "usage": {"***": {"min_context": "64k"}}})
    cat = UsageCatalogue.from_cfg(profiles.usage_cfg)
    v = _variant(80000, (cat.get("***"),))
    config = emit_config([m], {"m": [v]}, profiles, TVARS)
    assert list(config["models"]) == ["m"]


def test_dump_yaml_annotates_set_memory():
    # The config documents its own memory split: a comment above each set.
    # Only direct children of `sets:` are annotated, so a var that happens to
    # share a set's name is left alone.
    from llama_packer.writer import dump_yaml
    text = dump_yaml(
        {"routing": {"router": {"settings": {"matrix": {
            "vars": {"rag": "m", "c1": "m2"},
            "sets": {"rag": "c1 & emb", "voice": "c1 & stt"},
        }}}}},
        {"rag": 1751, "voice": 1898})
    assert "# rag: 1751 MiB allocated to its co-resident models (1.71 GiB)" in text
    assert "# voice: 1898 MiB allocated to its co-resident models (1.85 GiB)" in text
    assert text.count("# rag:") == 1  # the `vars: rag` key is not annotated


def test_met_demands_enforces_require(make_model, tmp_path):
    # A serving must never be named for a category whose `require` it fails:
    # a text-only serving cannot claim a `require: [vision]` category.
    from llama_packer.writer import Planner
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x")
    profiles = Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "usage": {"qa": {"min_context": "32k", "require": ["vision"]}},
    })
    with_vis = make_model("v", context_length=65536,
                          capabilities=["image"],
                          usage=["qa"],
                          mmproj={"file": "v-mmproj.gguf",
                                  "capabilities": ["image"]})
    planner = Planner([with_vis], profiles, fit_bin="unused",
                      vram_total=48 * 1024)
    assert [d.name for d in planner.met_demands(
        with_vis, 65536, 1, include_mmproj=True)] == ["qa"]
    assert planner.met_demands(
        with_vis, 65536, 1, include_mmproj=False) == ()
    # No mmproj at all (the `…-vision` misnomer case): text only, no claim.
    plain = make_model("p", context_length=65536, usage=["qa"])
    planner = Planner([plain], profiles, fit_bin="unused", vram_total=48 * 1024)
    assert planner.met_demands(plain, 65536, 1, include_mmproj=True) == ()


# ── degradation levels: a category's own order, not a global threshold ────


def test_shed_levels_are_prefixes_of_the_declared_order():
    """`sacrifice: [draft]` must try draft first, not vision.

    The old `_mmproj_drop_pass` dropped vision from a global context
    threshold before any category was read, so a category that would rather
    keep vision and give up its draft had no say in it.
    """
    # Cumulative along the declared order, then the default tail: a category
    # that named one thing still concedes the other once that is gone.
    assert writer.shed_levels(("vision",)) == (
        frozenset(), frozenset({"vision"}),
        frozenset({"vision", "draft"}))
    assert writer.shed_levels(("draft",)) == (
        frozenset(), frozenset({"draft"}),
        frozenset({"draft", "vision"}))
    assert writer.shed_levels(("vision", "draft")) == (
        frozenset(), frozenset({"vision"}), frozenset({"vision", "draft"}))


def test_required_capabilities_are_never_shed():
    order = writer.shed_levels(("vision",), required=frozenset({"vision"}))
    assert frozenset({"vision"}) not in order, order


def test_unnamed_capabilities_come_last_not_first():
    """A category that names nothing still degrades, in the default order.

    Empty tuple is *no stated preference*, not *refuse to degrade* — the
    distinction matters, and conflating them was the bug.
    """
    assert writer.shed_levels(()) == (
        frozenset(), frozenset({"vision"}), frozenset({"vision", "draft"}))


def test_none_sacrifice_means_never_degrade():
    assert writer.shed_levels(None) == (frozenset(),)


def test_fallback_is_a_bare_minimum_category():
    """Every model emits this; it has no name and never degrades."""
    assert FALLBACK.name == ""
    assert FALLBACK.min_context == 4096
    assert FALLBACK.min_parallel == 1
    assert FALLBACK.sacrifice is None   # never degrades
