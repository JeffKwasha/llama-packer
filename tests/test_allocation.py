"""The allocation algebra: set floors and the weighted auction."""

from __future__ import annotations

import logging

import pytest

from llama_packer.allocation import (PoolSchema, auction, floor_mb,
                                     floor_of, parse, role_floors)
from llama_packer.roles import Share

FILL = Share("fill")
ZERO = Share("zero")
MIN = Share("min")


def w(value):
    return Share("weight", float(value))


def pct(value):
    return Share("percent", float(value))


def test_the_pool_tuple_starts_with_cpu():
    # Validates: docs/allocation.md, 'Pools': the cost tuple is built from
    # the host, cpu first, and names every field.
    with pytest.raises(ValueError, match="starts with cpu"):
        PoolSchema(["gpu0", "cpu"])
    schema = PoolSchema(["cpu", "gpu0"])
    assert schema(0, 3).gpu0 == 3


def test_set_floor_is_sum_and_max():
    # Validates: docs/allocation.md, 'Floor': & sums, | takes the maximum.
    floors = {"a": 4, "b": 6, "c": 5}
    assert floor_of(parse("a & b"), floors) == 10
    assert floor_of(parse("a | b"), floors) == 6
    assert floor_of(parse("a & (b | c)"), floors) == 10
    assert floor_of(parse("(a | b) & c"), floors) == 11


def test_auction_splits_by_weight():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    # their sum, so 15 becomes 6 and 9.
    alloc, left = auction(15, {"s": parse("a & b")}, {"a": 0, "b": 0},
                          {"a": w(60), "b": w(90)})
    assert alloc == {"a": 6, "b": 9}
    assert left == {"s": 0}


def test_each_set_keeps_its_own_offer_and_the_role_takes_the_minimum():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    # its own; a higher offer in one set changes nothing in another.
    floors = {"a": 0, "b": 0, "c": 10}
    sets = {"one": parse("a & b"), "two": parse("a & c")}
    alloc, left = auction(15, sets, floors,
                          {"a": w(60), "b": FILL, "c": MIN})
    assert alloc["a"] == 5
    assert alloc["b"] == 6
    assert alloc["c"] == 10
    assert left["one"] == 0
    assert left["two"] == 0


def test_min_and_zero():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    # zero accepts what the requesters leave.
    floors = {"a": 0, "b": 0}
    alloc, left = auction(10, {"s": parse("a & b")}, floors,
                          {"a": w(100), "b": ZERO})
    assert alloc == {"a": 10, "b": 0}
    assert left == {"s": 0}

    alloc, _left = auction(10, {"s": parse("a & b")}, floors,
                           {"a": MIN, "b": ZERO})
    assert alloc == {"a": 0, "b": 10}


def test_fill_takes_the_remainder():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    # the fill takes 4.
    alloc, left = auction(10, {"s": parse("a & b")}, {"a": 0, "b": 0},
                          {"a": w(60), "b": FILL})
    assert alloc == {"a": 6, "b": 4}
    assert left == {"s": 0}


def test_two_fills_split_the_remainder():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    alloc, left = auction(10, {"s": parse("a & b & c")},
                          {"a": 0, "b": 0, "c": 0},
                          {"a": w(60), "b": FILL, "c": FILL})
    assert alloc == {"a": 6, "b": 2, "c": 2}
    assert left == {"s": 0}


def test_overclaimed_fill_weighs_zero():
    # Validates: docs/allocation.md, 'The auction': a fill gets nothing when
    # the weighted roles already claim the whole auction.
    alloc, left = auction(10, {"s": parse("a & b")}, {"a": 0, "b": 0},
                          {"a": w(100), "b": FILL})
    assert alloc == {"a": 10, "b": 0}
    assert left == {"s": 0}


def test_percentage_claims_are_weights_in_the_same_space():
    # Validates: docs/allocation.md, 'The auction': N% claims N weight points;
    # a raw weight and a claim of the same number contribute equally.
    floors = {"a": 0, "b": 0}
    claim, left = auction(100, {"s": parse("a & b")}, floors,
                          {"a": pct(25), "b": w(75)})
    assert claim == {"a": 25, "b": 75}
    assert left == {"s": 0}
    raw, _ = auction(100, {"s": parse("a & b")}, floors,
                     {"a": w(25), "b": w(75)})
    assert raw == claim


def test_fill_weight_is_the_residual_after_percentage_claims():
    # Validates: docs/allocation.md, 'The auction': the fill's weight is the
    # unclaimed part of the 100-point space, whatever the other kind is.
    alloc, left = auction(10, {"s": parse("a & b")}, {"a": 0, "b": 0},
                          {"a": pct(60), "b": FILL})
    assert alloc == {"a": 6, "b": 4}
    assert left == {"s": 0}


def test_fill_mixed_with_a_raw_weight_warns(caplog):
    # Validates: docs/allocation.md, 'The auction': a fill's residual depends
    # Validates: FLOW_FILL_CLAIMS_RESIDUAL — docs/allocation.md §The auction
    # on a raw weight's arbitrary magnitude, so the mix is warned about.
    with caplog.at_level(logging.WARNING, logger="llama_packer.allocation"):
        alloc, _ = auction(1000, {"big": parse("A & B")},
                           {"A": 0, "B": 0}, {"A": w(123), "B": FILL})
    assert alloc == {"A": 1000, "B": 0}
    warns = [r.message for r in caplog.records if "fill" in r.message]
    assert len(warns) == 1
    assert warns[0] == (
        "set 'big': fill roles [B] mixed with raw weights [A]! "
        "use '%' shares with fill")


def test_percent_and_fill_split_the_whole_space(caplog):
    # Validates: docs/allocation.md, 'The auction': a percentage claim leaves
    # a real residual for fill, with no mixed-weight warning.
    shares = {"A": pct(66), "B": FILL}
    with caplog.at_level(logging.WARNING, logger="llama_packer.allocation"):
        alloc, left = auction(1000, {"big": parse("A & B")},
                              {"A": 0, "B": 0}, shares)
    assert alloc == {"A": 660, "B": 340}
    assert left == {"big": 0}
    assert not [r for r in caplog.records if "fill" in r.message]

    # The fill also gets what a claimant leaves behind: with a pool that does
    # not divide evenly, A's rounded-down share is not lost.
    alloc, left = auction(1001, {"big": parse("A & B")},
                          {"A": 0, "B": 0}, shares)
    assert alloc == {"A": 660, "B": 341}
    assert left == {"big": 0}


def test_fill_is_the_last_acceptor_of_unclaimed_memory():
    # Validates: docs/allocation.md, 'The auction': when a fill has no weight
    # Validates: FLOW_FILL_CLAIMS_RESIDUAL — tests/test_allocation.py
    # left, it gets the surplus that other accepting roles cannot use.
    floors = {r: 0 for r in "abcdef"}
    alloc, left = auction(10, {"s": parse("a & b & c & d & e & f")}, floors,
                          {"a": w(60), "b": w(60), "c": w(60), "d": w(60),
                           "e": ZERO, "f": FILL})
    assert alloc == {"a": 2, "b": 2, "c": 2, "d": 2, "e": 0, "f": 2}
    assert left == {"s": 0}


def test_infeasible_floor_is_an_error():
    with pytest.raises(ValueError, match="exceeds available"):
        auction(8, {"s": parse("a & b")}, {"a": 5, "b": 5},
                {"a": w(50), "b": w(50)})


def test_floor_mb_from_the_quad():
    # Validates: docs/allocation.md, 'Floor': weights + compute + KV per
    # token per slot + slot overhead.
    quad = (1000, 0.1, 50, 100)
    assert floor_mb(quad, 1000, 1) == 1250
    assert floor_mb(quad, 1000, 2) == 1400
    assert floor_mb(quad, 1000, 2, 0.04) == 1456


def test_role_floor_is_the_maximum_model_floor():
    # Validates: FLOW_ROLE_FLOOR_AND_WEIGHTED_AUCTION — SPEC.md §Host pool
    quad = (1000, 0.1, 50, 100)
    entries = {"a": [(quad, 1000, 1), (quad, 1000, 2)],
               "b": [(quad, 500, 1)]}
    assert role_floors(entries) == {"a": 1400, "b": 1200}
