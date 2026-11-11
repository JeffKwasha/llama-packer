"""The allocation algebra: set floors and the weighted auction."""

from __future__ import annotations

import pytest

from llama_packer.allocation import (PoolSchema, auction, floor_mb,
                                     floor_of, parse, role_floors)


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


def test_a_capped_role_leaves_memory_unallocated():
    # Validates: docs/allocation.md, 'The auction': each set is allocated
    # on its own; a role's final size is the minimum across sets. Memory a
    # capped role leaves is reported, not handed to another role.
    floors = {"a": 0, "b": 0, "c": 10}
    sets = {"one": parse("a & b"), "two": parse("a & c")}
    alloc, left = auction(15, sets, floors, {"a": 60, "b": 0, "c": 0},
                          fills=frozenset({"b"}))
    assert alloc["a"] == 5
    assert alloc["b"] == 6
    assert alloc["c"] == 10
    assert left["one"] == 4


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


def test_floor_mb_from_the_quad():
    # Validates: docs/allocation.md, 'Floor': weights + compute + KV per
    # token per slot + slot overhead.
    quad = (1000, 0.1, 50, 100)
    assert floor_mb(quad, 1000, 1) == 1250
    assert floor_mb(quad, 1000, 2) == 1400
    assert floor_mb(quad, 1000, 2, 0.04) == 1456


def test_role_floor_is_the_maximum_model_floor():
    quad = (1000, 0.1, 50, 100)
    entries = {"a": [(quad, 1000, 1), (quad, 1000, 2)],
               "b": [(quad, 500, 1)]}
    assert role_floors(entries) == {"a": 1400, "b": 1200}
