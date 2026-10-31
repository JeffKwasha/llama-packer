# tests/test_usage.py
"""Demand-driven planning: the blessed ``tasks:`` catalogue and the
capability/context ladder."""

from __future__ import annotations

import pytest

import llama_packer.writer as writer
from llama_packer.endpoints import CHAT
from llama_packer import tasks
from conftest import TASK_ENDPOINT, catalogue, task_cfg, tasks_section
from llama_packer.tasks import Task, TaskCatalogue, resolve_tasks
from llama_packer.utils import parse_ctx_size as _parse_ctx_size
from llama_packer.writer import FALLBACK, KNOWN_CAPABILITIES


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
    # Validates: SPEC.md, 'Context Tiers': min_context states a floor, as a positive int or a 128k/1M string
    assert _parse_ctx_size(raw) == expected


@pytest.mark.parametrize("raw", [0, -1, "abc", "", None, True, [], {}])
def test_parse_ctx_size_rejects_junk(raw):
    # Validates: SPEC.md, 'Context Tiers': a context size is a positive integer or a 128k / 1M string; anything else is rejected
    assert _parse_ctx_size(raw) is None


# ── the catalogue ──────────────────────────────────────────────────────────


def test_catalogue_parses_full_spec():
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': the tasks: catalogue is parsed from profiles.yaml.
    cat = catalogue({
        "coding": {"min_context": "128k", "min_parallel": 1,
                   "sacrifice": ["vision", "draft"]},
        "chat": {"min_context": "64k", "min_parallel": 2,
                 "sacrifice": ["draft", "vision"]},
        "qa": {"min_context": "32k", "min_parallel": 2, "require": ["vision"]},
    })
    assert set(cat.order) == set(TASK_ENDPOINT)
    coding = _d(cat, "coding")
    assert (coding.min_context, coding.min_parallel) == (131072, 1)
    # sacrifice keeps the declared order -- it is a preference order
    assert coding.sacrifice == ("vision", "draft")
    assert _d(cat, "chat").sacrifice == ("draft", "vision")
    assert cat.get("qa").require == frozenset({"vision"})


def test_catalogue_int_shorthand_is_min_context():
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': a bare integer is the category's min_context.
    cat = catalogue({"coding": 65536})
    assert cat.get("coding").min_context == 65536
    assert cat.get("coding").min_parallel == 1


def test_catalogue_warns_and_drops_malformed_categories(caplog):
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': a malformed category is warned about and dropped.
    with caplog.at_level("WARNING"):
        cat = catalogue({
            "chat": {"min_context": "32k"},
            "coding": "garbage",
            "qa": {"min_context": "abc"},
            "aux": {"min_parallel": 0},
        })
    assert set(cat.order) == set(TASK_ENDPOINT) - {"coding"}
    assert "not a mapping" in caplog.text
    # an unusable minimum is dropped, the task survives with its default
    assert cat.get("qa").min_context == 0
    assert cat.get("aux").min_parallel == 1


def test_catalogue_warns_on_unknown_capability(caplog):
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': an unknown capability is warned about and dropped.
    with caplog.at_level("WARNING"):
        cat = catalogue(
            {"qa": {"require": ["vision", "telepathy"]}})
    assert "unknown capabilities" in caplog.text
    assert "telepathy" in caplog.text
    assert cat.get("qa").require == frozenset({"vision"})


def test_catalogue_empty_config_keeps_the_defaults():
    # Validates: SPEC.md, 'Tasks': an empty tasks: section keeps the catalog defaults.
    assert not TaskCatalogue.from_cfg(None)


def test_known_capabilities_are_the_ladder_axes():
    # Validates: docs/plans/category-semantics.md, 'Degradation': the ladder's axes are vision and draft
    assert KNOWN_CAPABILITIES == frozenset({"vision", "draft"})


# ── tag resolution: only blessed tags count ────────────────────────────────


def test_resolve_tasks_accepts_blessed_tags():
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': only the tags the catalogue declares become demands.
    cat = catalogue({"coding": {"min_context": "128k"},
                                   "chat": {"min_context": "64k"}})
    got = resolve_tasks(["coding", "chat"], cat)
    assert [d.id for d in got] == ["coding", "chat"]


def test_resolve_tasks_dedupes_repeats():
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': a repeated tag is one demand.
    cat = catalogue({"chat": {"min_context": "64k"}})
    assert [d.id for d in resolve_tasks(["chat", "chat"], cat)] == ["chat"]


def test_resolve_tasks_ignores_unblessed_tag_with_warning(caplog):
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': an undeclared tag is warned about and ignored.
    cat = catalogue({"chat": {"min_context": "64k"}})
    with caplog.at_level("WARNING"):
        got = resolve_tasks(["chat", "code"], cat)
    assert [d.id for d in got] == ["chat"]
    assert "is not a declared task" in caplog.text
    assert "code" in caplog.text
    # the warning names what *is* blessed, so the operator can see the fix
    assert "chat" in caplog.text


def test_resolve_tasks_accepts_bare_string():
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': a bare string is one tag.
    cat = catalogue({"chat": {"min_context": "64k"}})
    assert [d.id for d in resolve_tasks("chat", cat)] == ["chat"]


def test_resolve_tasks_no_tags_means_no_demands(caplog):
    # Validates: SPEC.md, 'Usage categories (`tasks:`)': no tags means no demands.
    cat = catalogue({"chat": {"min_context": "64k"}})
    assert resolve_tasks(None, cat) == ()
    with caplog.at_level("WARNING"):
        assert resolve_tasks(42, cat) == ()


# ── the ladder ─────────────────────────────────────────────────────────────


def _cat(**specs) -> TaskCatalogue:
    return catalogue(specs)


def _d(cat: TaskCatalogue, name: str) -> Task:
    """Narrowing accessor: a declared category always exists."""
    demand = cat.get(name)
    assert demand is not None, f"category {name!r} not declared"
    return demand


def _demands(cat: TaskCatalogue, *names: str) -> tuple[Task, ...]:
    return tuple(_d(cat, n) for n in names)


# ── per-set co-resident overhead ────────────────────────────────────────────

from llama_packer.writer import (  # noqa: E402
    MatrixKnobs,
    Planner,
    _expr_leaves,
    set_overhead_mb,
    _split_top,
)

_MB = {"emb": 749, "rnk": 1002, "rnk2": 5766, "stt": 1898, "tts": 1204,
       "s2t": 2814, "s2t2": 1914}
_CH = frozenset(f"c{i}" for i in range(1, 39))
_CO = frozenset({"s2t", "s2t2"})


def test_split_top_respects_parentheses():
    # Validates: docs/plans/opportunistic-coload.md, 'Design sketch': the split is at parenthesis depth zero.
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
    # Validates: SPEC.md, 'VRAM Allocation': a co-resident set is charged the sum of its members
    assert set_overhead_mb(expr, _MB, _CH, _CO) == expected


def test_set_overhead_charges_priciest_or_branch_only():
    # emb and rnk2 are alternatives -- only one is ever resident, and the
    # entry must survive the heavier one.
    # Validates: docs/plans/opportunistic-coload.md, 'Design': an OR branch charges only the pricier side.
    assert set_overhead_mb("__CHAT_VARS__ & (emb | rnk2)", _MB, _CH, _CO) == 5766
    assert set_overhead_mb("((c1 | c2)) & emb & (rnk | rnk2)",
                           _MB, _CH, _CO) == 749 + 5766


def test_set_overhead_sums_coloads():
    # co-loads genuinely stack with each other, unlike OR branches
    # Validates: docs/plans/opportunistic-coload.md, 'Design': co-loads stack with each other.
    assert set_overhead_mb("__CHAT_VARS__ & __COLOAD_VARS__",
                           _MB, _CH, _CO) == 2814 + 1914


def test_set_overhead_ignores_explicit_chat_vars():
    # Validates: SPEC.md, 'VRAM Allocation': an explicit chat var is not charged as a co-resident
    assert set_overhead_mb("(c1 | c2) & emb", _MB, _CH, _CO) == 749
    assert set_overhead_mb("c1 & emb & stt", _MB, _CH, _CO) == 749 + 1898


def test_set_overhead_ignores_unknown_vars():
    # an unresolvable var charges nothing rather than exploding the solve
    # Validates: docs/plans/opportunistic-coload.md, 'Design': an unknown var is charged nothing.
    assert set_overhead_mb("__CHAT_VARS__ & mystery", _MB, _CH, _CO) == 0


def test_set_overhead_prices_nested_branches():
    # Validates: docs/plans/cpu-residency-budget.md #22 -- a set expression is
    # a boolean tree: `&` sums, `|` maxes, and a nested group is never
    # mistaken for an unknown var worth 0.
    buckets = {"one": 1000, "two": 2000, "three": 4000}
    ch = frozenset({"c1"})
    # (two & three) is a branch, so this is max(1000, 6000) -- the old
    # depth-0 split saw an unknown var and returned 1000.
    assert set_overhead_mb("(c1) & (one | (two & three))", buckets,
                           ch, frozenset()) == 6000
    assert set_overhead_mb("(c1) & (three | (two & one))", buckets,
                           ch, frozenset()) == 4000


def test_set_overhead_expands_named_sets():
    # Validates: docs/plans/cpu-residency-budget.md #22 -- `+set` expands the
    # referenced expression, and a cycle terminates instead of recursing.
    buckets = {"one": 1000, "two": 2000, "three": 4000}
    cfgs = {"combo": "three | (two & one)"}
    assert set_overhead_mb("(c1) & +combo", buckets, frozenset({"c1"}),
                           frozenset(), sets_cfg=cfgs) == 4000
    cyc = {"a": "+b", "b": "+a"}
    assert set_overhead_mb("(c1) & +a", buckets, frozenset({"c1"}),
                           frozenset(), sets_cfg=cyc) == 0


def test_expr_leaves_walks_nested_branches():
    # Validates: docs/plans/cpu-residency-budget.md #22 -- the walker names the
    # real leaves of a nested branch, not a variable called "two & three", and
    # expands `+set` references.
    assert _expr_leaves("(c1) & (three | (two & one))") == \
        {"c1", "one", "two", "three"}
    assert _expr_leaves("(c1) & +combo",
                        {"combo": "three | (two & one)"}) == \
        {"c1", "one", "two", "three"}


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
    # Validates: docs/plans/category-semantics.md, 'Ids carry no context': the served context is labelled compactly.
    assert ctx_label(ctx) == expected


def test_ctx_label_never_exceeds_four_characters():
    # A client scans the id list; an id that wanders past four characters
    # stops being comparable at a glance.
    # Validates: docs/plans/category-semantics.md, 'Ids carry no context': the label is compact enough for an id.
    for ctx in range(0, 4 * 1024 * 1024, 4096):
        assert len(ctx_label(ctx)) <= 4, ctx


def test_ctx_labels_below_the_mega_threshold_are_unique():
    # Exact contexts must not collide, or two entries would share an id.
    # Validates: docs/plans/category-semantics.md, 'Ids carry no context': the labels are unique below the mega threshold.
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
    # Validates: docs/plans/auto-parallel.md, 'Design': a candidate that meets min_parallel is taken.
    table = {p: 262144 for p in range(1, 9)}
    p, ctx = _auto_par(table, floor=65536, min_parallel=4)
    assert p >= 4, (p, ctx)
    assert ctx >= 65536


def test_context_floor_outranks_min_parallel():
    # Two slots would drop under the demand's 64k floor, so the single wide
    # window is served instead. Concurrency is a preference; the context is
    # what the entry id advertises, and a client that cannot fit its prompt
    # gains nothing from a spare slot.
    # Validates: docs/plans/auto-parallel.md, 'Design': the context floor outranks min_parallel.
    table = {1: 262144, 2: 40000}
    p, ctx = _auto_par(table, floor=65536, min_parallel=2)
    assert (p, ctx) == (1, 262144), (p, ctx)


def test_demand_never_trades_below_its_context_floor():
    # The budget only reaches ~96k however few slots are used, so a 128k
    # coding demand is unreachable: no candidate qualifies and the solver
    # falls back rather than inventing a context. Dropping the model from
    # the sets it cannot serve is the admission pass's job, not this one's.
    # Validates: docs/plans/auto-parallel.md, 'Design': a demand is never served below its own context floor.
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
    emit_config,
    flags_label,
    served_flags,
)
from llama_packer.profiles import Profiles  # noqa: E402

TVARS = {"llama_bin": "/opt/llama-server"}


def _id_cat() -> TaskCatalogue:
    return catalogue({
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
    # Validates: SPEC.md, 'Entry ids name the task, not the solve': the flag segment is the served capability set as first letters, in the fixed order t, i, v, a, s
    assert flags_label(caps) == expected



def test_served_flags_drops_image_and_demoted_tools(make_model, tmp_path):
    # Validates: SPEC.md, 'VRAM Allocation': the flag segment is what is served, not what was declared.
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
                   include_mmproj=True, task_demands=usage)


def _profiles_with_usage() -> Profiles:
    return Profiles({
        "defaults": {"cache_type": "q8_0", "parallel": 1},
        "tasks": {"coding": {"min_context": "128k", "min_parallel": 1},
                  "chat": {"min_context": "64k", "min_parallel": 2},
                  "qa": {"min_context": "32k", "min_parallel": 2}},
    })


def test_emit_id_names_heaviest_category_with_flags(make_model):
    # Validates: SPEC.md, 'Output Format': the id names the heaviest satisfied category and the served flags.
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    cat = catalogue(_profiles_with_usage().tasks_cfg)
    usage = _demands(cat, "coding", "chat", "qa")
    config = emit_config([m], {"qwen38": [_variant(160000, usage)]},
                         _profiles_with_usage(), TVARS)
    assert list(config["models"]) == ["qwen38.coding.ti"]
    entry = config["models"]["qwen38.coding.ti"]
    assert entry["metadata"]["tasks"] == ["coding", "chat", "qa"]
    assert entry["metadata"]["ctx_size"] == 160000
    assert "coding" in entry["name"] and "156k" in entry["name"]


def test_ids_stable_when_context_changes(make_model):
    # The whole point: a repack that moves the solve must not rename the id,
    # because opencode/hermes cache model ids at startup.
    # Validates: SPEC.md, 'Output Format': the id does not move when the served context changes.
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    cat = catalogue(profiles.tasks_cfg)
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

    Validates: SPEC.md, 'Ids carry no context': an untagged entry is the bare stem.
    """
    m = make_model("plain", context_length=262144)
    config = emit_config([m], {"plain": [_variant(131072, ())]},
                         _profiles_with_usage(), TVARS)
    assert list(config["models"]) == ["plain"]
    assert config["models"]["plain"]["metadata"]["ctx_size"] == 131072


def test_same_model_two_groups_disambiguated(make_model):
    """
    Validates: SPEC.md, 'Ids carry no context': ids stay stable across repacks, so the
    disambiguator describes the serving and never the solve.
    """
    # Two profile groups can land on the same category+flags; the first keeps
    # the stable name, the duplicate appends a discriminator rather than
    # colliding into a fatal error.
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    cat = catalogue(profiles.tasks_cfg)
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

    Validates: SPEC.md, 'Ids carry no context': 182k -> 180k must not rename a model.
    """
    m = make_model("qwen38", context_length=262144,
                   capabilities=["tools", "image"])
    profiles = _profiles_with_usage()
    usage = (catalogue(profiles.tasks_cfg).get("coding"),)
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
    # Validates: SPEC.md, 'Output Format': the category is slugged for the id.
    m = make_model("m", context_length=262144, capabilities=["tools"])
    profiles = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1}})
    v = _variant(80000, (Task("My Code!", CHAT, "chat"),))
    config = emit_config([m], {"m": [v]}, profiles, TVARS)
    assert list(config["models"]) == ["m.my-code.t"]


def test_unusable_category_name_falls_back_to_a_bare_id(make_model):
    """A name that cannot survive slugging drops the segment, not the entry.

    Validates: SPEC.md, 'Ids carry no context': a name that cannot slug drops the
    segment, not the entry.
    """
    m = make_model("m", context_length=262144)
    profiles = Profiles({"defaults": {"cache_type": "q8_0", "parallel": 1}})
    v = _variant(80000, (Task("***", CHAT, "chat"),))
    config = emit_config([m], {"m": [v]}, profiles, TVARS)
    assert list(config["models"]) == ["m"]


def test_dump_yaml_annotates_set_memory():
    # The config documents its own memory split: a comment above each set.
    # Only direct children of `sets:` are annotated, so a var that happens to
    # share a set's name is left alone.
    # Validates: SPEC.md, 'Output Format': the emitted YAML carries the memory set it was built with.
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


def test_require_gates_existence(make_model, tmp_path):
    """A model lacking a required capability does not satisfy the category.

    This is `require` and not `keep`: `keep` only forbids shedding, while a
    category that `require`s vision simply does not apply to a text-only
    model, so no entry may be named for it. The gate lived in
    Planner.met_demands and went when that did, which briefly let a
    text-only model claim `require: [vision]`.

    Validates: SPEC.md: 'require and keep are different and both are
    honoured'.
    """
    from llama_packer.writer import Planner
    (tmp_path / "v-mmproj.gguf").write_bytes(b"x" * 3 * 1024 * 1024)
    text_only = make_model("t")
    with_vision = make_model("v", mmproj={"file": "v-mmproj.gguf",
                                          "capabilities": ["image"]})
    planner = Planner([text_only, with_vision], _profiles_with_usage(),
                      fit_bin="unused", vram_total=48 * 1024)
    need_vision = Task("qa", CHAT, "chat", min_context=32768,
                              require=frozenset({"vision"}))
    assert planner.can_meet(with_vision, need_vision)
    assert not planner.can_meet(text_only, need_vision)

    # And the draft case: no companion, no category.
    need_draft = Task("c", CHAT, "chat", min_context=32768,
                             require=frozenset({"draft"}))
    assert not planner.can_meet(text_only, need_draft)
    # keep, by contrast, does not gate.
    keeps = Task("c2", CHAT, "chat", min_context=32768,
                        keep=frozenset({"draft"}))
    assert planner.can_meet(text_only, keeps)


def test_shed_levels_are_prefixes_of_the_declared_order():
    """`sacrifice: [draft]` must try draft first, not vision.

    The old `_mmproj_drop_pass` dropped vision from a global context
    threshold before any category was read, so a category that would rather
    keep vision and give up its draft had no say in it.

    Validates: SPEC.md, 'Shedding is described per category': the ladder walks the
    category's own sacrifice order, not a global one.
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
    """
    Validates: SPEC.md, 'Shedding is described per category': require gates existence.
    """
    order = writer.shed_levels(("vision",), required=frozenset({"vision"}))
    assert frozenset({"vision"}) not in order, order


def test_unnamed_capabilities_come_last_not_first():
    """A category that names nothing still degrades, in the default order.

    Empty tuple is *no stated preference*, not *refuse to degrade* — the
    distinction matters, and conflating them was the bug.

    Validates: SPEC.md, 'Shedding is described per category': what a category omits is
    shed last, in the order from `shed:`.
    """
    assert writer.shed_levels(()) == (
        frozenset(), frozenset({"vision"}), frozenset({"vision", "draft"}))


def test_none_sacrifice_means_never_degrade():
    """
    Validates: docs/plans/category-semantics.md, 'The null category (fallback)': the
    fallback keeps mmproj and MTP even at the cost of all its context.
    """
    assert writer.shed_levels(None) == (frozenset(),)


def test_fallback_is_a_bare_minimum_category():
    """Every model emits this; it has no name and never degrades.

    Validates: docs/plans/category-semantics.md, 'The null category (fallback)'.
    """
    assert FALLBACK.id == ""
    assert FALLBACK.min_context == 4096
    assert FALLBACK.min_parallel == 1
    assert FALLBACK.sacrifice == ()      # default order, floor-gated


# ── keep vs require ───────────────────────────────────────────────────────


def test_keep_protects_from_shedding_without_gating_existence():
    """`coding` refuses to drop MTP, but a model without MTP is still coding.

    Making this `require` would silently drop every coding model that has no
    draft companion -- a large share of the fleet -- which is why it needed
    a word of its own rather than a wider meaning for `require`.

    Validates: SPEC.md: 'require and keep are different and both are honoured' -- keep
    forbids shedding; only require gates existence.
    """
    d = Task("coding", CHAT, "chat", min_context=133120, keep=frozenset({"draft"}))
    assert d.require == frozenset()
    # MTP is protected from the ladder...
    levels = writer.shed_levels(d.sacrifice, required=d.require, kept=d.keep)
    assert all("draft" not in lvl for lvl in levels), levels
    # ...but a model without it still satisfies the category. The existence
    # gate lives in met_demands, which drops a demand whose required
    # capability is not in the served flags -- satisfied_by itself only ever
    # looked at context and slots.
    assert _served_without_draft(d), "keep must NOT gate existence"
    strict = Task("coding", CHAT, "chat", min_context=133120,
                         require=frozenset({"draft"}))
    assert not _served_without_draft(strict), "require must gate existence"
    # satisfied_by itself only ever looked at context and slots; the
    # capability gate is met_demands' job, so it cannot tell them apart.
    assert d.satisfied_by(133120, 1) == strict.satisfied_by(133120, 1)


def _served_without_draft(demand):
    """Whether *demand* counts as met by a text-only, no-draft serving.

    Mirrors Planner.met_demands' capability gate: a demand is kept only if
    every capability it requires is in the served flags. Text-only serving
    serves neither vision nor draft, so both keys reduce to the same question.
    """
    served = {"t"}
    for cap in demand.require:
        if cap == "vision" and not ({"i", "v"} & served):
            return False
        if cap == "draft":
            return False
    return demand.satisfied_by(133120, 1)


def test_keep_is_parsed_from_profiles():
    """
    Validates: SPEC.md, `tasks:` block: `keep:` parses like `require:`.
    """
    profiles = Profiles({"defaults": {}, "tasks": {
        "coding": {"min_context": "128k", "keep": ["draft"],
                   "sacrifice": ["vision"]}}})
    cat = catalogue(profiles.tasks_cfg)
    d = cat.get("coding")
    assert d.keep == frozenset({"draft"})
    assert d.sacrifice == ("vision",)
    assert d.require == frozenset()


# ── the category is a declaration, and these prove each rule ──────────────


def test_task_ids_resolve_to_declared_demands():
    """A sidecar's `tasks:` resolves to exactly what profiles.yaml declared.

    Validates: docs/plans/requirement-trace.md #1 -- the category comes from
    the sidecar tag; nothing infers or overrides it.
    """
    profiles = Profiles({"defaults": {}, "tasks": {
        "coding": {"min_context": "130k", "min_parallel": 1},
        "aux": {"min_context": "80k", "min_parallel": 2}}})
    cat = catalogue(profiles.tasks_cfg)
    got = resolve_tasks(["coding", "aux"], cat)
    assert [d.id for d in got] == ["coding", "aux"]
    assert [d.min_context for d in got] == [133120, 81920]
    # Declaration order is preserved, not sorted by weight.
    assert [d.id for d in resolve_tasks(["aux", "coding"], cat)] == \
        ["aux", "coding"]


def test_usage_tag_not_declared_warns_and_is_ignored(caplog):
    """Only declared categories exist; an undeclared one cannot be invented.

    Validates: docs/plans/requirement-trace.md #2 -- an undeclared tag
    warns and is ignored, so the categories in force are exactly the ones
    the operator wrote.
    """
    import logging
    profiles = Profiles({"defaults": {}, "tasks": {
        "coding": {"min_context": "130k"}}})
    cat = catalogue(profiles.tasks_cfg)
    with caplog.at_level(logging.WARNING):
        got = resolve_tasks(["coding", "invented"], cat)
    assert [d.id for d in got] == ["coding"]
    assert any("invented" in r.message for r in caplog.records), [
        r.message for r in caplog.records]


def test_usage_warnings_lead_with_the_sidecar_name(caplog):
    """Every `tasks:` warning is about one sidecar, so it names it first.

    Validates: SPEC.md, 'Usage categories (`tasks:`)': a diagnostic leads
    with the sidecar's name, so an operator can triage it by name.
    """
    import logging
    cat = catalogue({"chat": {"min_context": "64k"}})
    with caplog.at_level(logging.WARNING):
        # all three failure shapes: not a list, not a category, not declared
        assert resolve_tasks(42, cat, stem="s1") == ()
        assert [d.id for d in resolve_tasks(["chat", 7], cat, stem="s1")] \
            == ["chat"]
        assert [d.id for d in resolve_tasks(["chat", "zork"], cat, stem="s1")] \
            == ["chat"]
    warned = [r.message for r in caplog.records]
    assert warned, "expected the three failures to warn"
    assert all(m.startswith("s1:") for m in warned), warned


def test_audio_cpp_task_must_match_the_model_task(make_model, caplog):
    # Validates: docs/plans/requirement-trace.md #98 -- an engine-level task cannot contradict the model's task.
    from llama_packer.discover import _warn_unsupported_values

    m = make_model("tts", tasks=["s2t"], audio_cpp={"family": "qwen3_tts",
                                                    "task": "tts"})
    with caplog.at_level("WARNING"):
        _warn_unsupported_values(m, "tts.md")
    assert "audio_cpp.task 'tts' is a t2s task" in caplog.text


# ── the catalogue is data, not a hardcoded list ────────────────────────────


def test_a_declared_task_id_is_arbitrary(make_model, tmp_path):
    """A task id that appears nowhere in the code still serves.

    Validates: docs/plans/requirement-trace.md #99 -- task ids are declared
    in profiles; only the endpoint and scheduling role are checked.
    """
    arbitrary = "zq7-odd-job"
    cfg = {arbitrary: {"endpoint": CHAT, "role": "chat",
                       "min_context": "8k"}}
    cat = TaskCatalogue.from_cfg(cfg)
    assert cat.get(arbitrary) is not None

    m = make_model("qwen3-8b", tasks=[arbitrary])
    from llama_packer.scope import ScopeStack
    ScopeStack(avail={"llama_bin": "/opt/llama-server"},
               tasks=cat).finalize(m)
    assert m.role == "chat"
    assert getattr(m, "_override_error", None) is None

    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "profiles": {"default": {}},
                         "tasks": cfg})
    config = emit_config([m], {"qwen3-8b": [_variant(16000, (cat.get(arbitrary),))]},
                         profiles, TVARS)
    assert list(config["models"]) == [f"qwen3-8b.{arbitrary}"]


def test_an_undeclared_task_id_fails(make_model, caplog):
    """The same arbitrary id, undeclared, is not a task.

    Validates: docs/plans/requirement-trace.md #99 -- an id outside the
    declared catalogue warns and produces no task entry.
    """
    declared = "chat"
    cat = TaskCatalogue.from_cfg({declared: task_cfg(declared)})
    m = make_model("qwen3-8b", tasks=["zq7-odd-job"])
    with caplog.at_level("WARNING"):
        got = resolve_tasks(m.task_ids, cat, stem="qwen3-8b.md")
    assert got == ()
    assert "is not a declared task" in caplog.text
    assert "zq7-odd-job" in caplog.text

    profiles = Profiles({"defaults": {"cache_type": "q8_0"},
                         "profiles": {"default": {}},
                         "tasks": {declared: task_cfg(declared)}})
    config = emit_config([m], {"qwen3-8b": [_variant(16000, ())]},
                         profiles, TVARS)
    assert list(config["models"]) == ["qwen3-8b"]
