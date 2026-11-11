"""The allocation algebra: set floors and the weighted auction."""

from __future__ import annotations

import pytest

from llama_packer.allocation import PoolSchema, auction, floor_of, parse


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
    # Validates: docs/allocation.md, 'The auction': 60 and 90 are scaled by
    # their sum, so 15 becomes 6 and 9.
    floors = {"a": 0, "b": 0}
    alloc, left = auction(15, {"s": parse("a & b")}, floors,
                          {"a": 60, "b": 90})
    assert alloc == {"a": 6, "b": 9}
    assert left == {"s": 0}


def test_capped_role_renormalises_its_set():
    # Validates: docs/allocation.md, 'The auction': the fill takes the
    # remainder when another set caps the weighted role.
    floors = {"a": 0, "b": 0, "c": 10}
    sets = {"one": parse("a & b"), "two": parse("a & c")}
    alloc, _left = auction(15, sets, floors, {"a": 60, "b": 0, "c": 0},
                           fills=frozenset({"b"}))
    assert alloc["a"] == 5
    assert alloc["b"] == 10
    assert alloc["c"] == 10


def test_min_and_zero():
    # Validates: docs/allocation.md, 'The auction': min accepts nothing,
    # zero accepts what the requesters leave.
    floors = {"a": 0, "b": 0}
    alloc, left = auction(10, {"s": parse("a & b")}, floors,
                          {"a": 100, "b": 0}, accepts=frozenset({"b"}))
    assert alloc == {"a": 10, "b": 0}
    assert left == {"s": 0}

    alloc, _left = auction(10, {"s": parse("a & b")}, floors,
                           {"a": 0, "b": 0}, accepts=frozenset({"b"}))
    assert alloc == {"a": 0, "b": 10}


def test_fill_takes_the_remainder():
    # Validates: docs/allocation.md, 'The auction': A takes 60% of 10 and
    # the fill takes 4.
    floors = {"a": 0, "b": 0}
    alloc, left = auction(10, {"s": parse("a & b")}, floors,
                          {"a": 60, "b": 0}, fills=frozenset({"b"}))
    assert alloc == {"a": 6, "b": 4}
    assert left == {"s": 0}


def test_two_fills_split_the_remainder():
    floors = {"a": 0, "b": 0, "c": 0}
    alloc, left = auction(10, {"s": parse("a & b & c")}, floors,
                          {"a": 60, "b": 0, "c": 0},
                          fills=frozenset({"b", "c"}))
    assert alloc == {"a": 6, "b": 2, "c": 2}
    assert left == {"s": 0}


def test_overclaimed_fill_weighs_zero():
    # Validates: docs/allocation.md, 'The auction': a fill gets nothing when
    # the weighted roles already claim the whole auction.
    floors = {"a": 0, "b": 0}
    alloc, left = auction(10, {"s": parse("a & b")}, floors,
                          {"a": 100, "b": 0}, fills=frozenset({"b"}))
    assert alloc == {"a": 10, "b": 0}
    assert left == {"s": 0}


def test_infeasible_floor_is_an_error():
    with pytest.raises(ValueError, match="exceeds available"):
        auction(8, {"s": parse("a & b")}, {"a": 5, "b": 5},
                {"a": 50, "b": 50})
