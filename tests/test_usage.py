# tests/test_usage.py
"""Demand-driven planning: the blessed ``usage:`` catalogue and the
capability/context ladder."""

from __future__ import annotations

import pytest

from llama_packer.writer import (
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
