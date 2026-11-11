"""The allocation algebra: set floors and the weighted auction."""

from __future__ import annotations

import pytest

from llama_packer.allocation import (PoolSchema, auction, floor_of, parse)

SCHEMA = PoolSchema(["cpu", "gpu0", "gpu1"])
ZERO = SCHEMA.zero()


def m(cpu=0, gpu0=0, gpu1=0):
    return SCHEMA(cpu, gpu0, gpu1)


def test_the_pool_tuple_starts_with_cpu():
    # Validates: docs/allocation.md, 'Pools': the tuple is built from the
    # host, cpu first, and names every field.
    with pytest.raises(ValueError, match="starts with cpu"):
        PoolSchema(["gpu0", "cpu"])
    assert m(gpu0=3).cpu == 0
    assert m(gpu0=3).gpu0 == 3


def test_set_floor_is_sum_and_max():
    # Validates: docs/allocation.md, 'Floor': & sums, | takes the maximum.
    floors = {"a": m(gpu0=4), "b": m(gpu0=6), "c": m(gpu0=5)}
    assert floor_of(parse("a & b"), floors) == m(gpu0=10)
    assert floor_of(parse("a | b"), floors) == m(gpu0=6)
    assert floor_of(parse("a & (b | c)"), floors) == m(gpu0=10)
    assert floor_of(parse("(a | b) & c"), floors) == m(gpu0=11)


def test_auction_splits_by_weight():
    # Validates: docs/allocation.md, 'The auction': requests are normalised
    # inside the set. 60 and 90 of 15 GiB are 6 and 9.
    floors = {"a": ZERO, "b": ZERO}
    alloc, left = auction(m(gpu0=15), {"s": parse("a & b")}, floors,
                          {"a": 60, "b": 90})
    assert alloc == {"a": m(gpu0=6), "b": m(gpu0=9)}
    assert left == {"s": ZERO}


def test_capped_role_renormalises_its_set():
    # Validates: docs/allocation.md, 'The auction': A's other set offers 5,
    # so B takes the remaining 10.
    floors = {"a": ZERO, "b": ZERO, "c": m(gpu0=10)}
    sets = {"one": parse("a & b"), "two": parse("a & c")}
    alloc, _left = auction(m(gpu0=15), sets, floors,
                           {"a": 60, "b": 0, "c": 0},
                           fills=frozenset({"b"}))
    assert alloc["a"] == m(gpu0=3)
    assert alloc["b"] == m(gpu0=12)
    assert alloc["c"] == m(gpu0=10)


def test_min_and_zero():
    # Validates: docs/allocation.md, 'The auction': min accepts nothing,
    # zero accepts what the requesters leave.
    floors = {"a": ZERO, "b": ZERO}
    alloc, left = auction(m(gpu0=10), {"s": parse("a & b")}, floors,
                          {"a": 100, "b": 0}, accepts=frozenset({"b"}))
    assert alloc["a"] == m(gpu0=10)
    assert alloc["b"] == ZERO
    assert left["s"] == ZERO

    alloc, left = auction(m(gpu0=10), {"s": parse("a & b")}, floors,
                          {"a": 0, "b": 0}, accepts=frozenset({"b"}))
    assert alloc["a"] == ZERO
    assert alloc["b"] == m(gpu0=10)


def test_fill_takes_the_remainder():
    # Validates: docs/allocation.md, 'The auction': fill is not a weight;
    # it claims after the weighted roles.
    floors = {"a": ZERO, "b": ZERO}
    alloc, left = auction(m(gpu0=10), {"s": parse("a & b")}, floors,
                          {"a": 60, "b": 0}, fills=frozenset({"b"}))
    assert alloc["a"] == m(gpu0=6)
    assert alloc["b"] == m(gpu0=4)
    assert left["s"] == ZERO


def test_two_fills_split_the_remainder():
    floors = {"a": ZERO, "b": ZERO, "c": ZERO}
    alloc, left = auction(m(gpu0=10), {"s": parse("a & b & c")}, floors,
                          {"a": 60, "b": 0, "c": 0},
                          fills=frozenset({"b", "c"}))
    assert alloc["a"] == m(gpu0=6)
    assert alloc["b"] == m(gpu0=2)
    assert alloc["c"] == m(gpu0=2)
    assert left["s"] == ZERO


def test_two_zero_roles_split_the_leftover():
    floors = {"a": ZERO, "b": ZERO, "c": ZERO, "d": m(gpu0=10)}
    sets = {"one": parse("a & b & c"), "two": parse("a & d")}
    alloc, left = auction(m(gpu0=15), sets, floors,
                          {"a": 100, "b": 0, "c": 0, "d": 0},
                          accepts=frozenset({"b", "c"}))
    assert alloc["a"] == m(gpu0=5)
    assert alloc["b"] == m(gpu0=5)
    assert alloc["c"] == m(gpu0=5)
    assert left["one"] == ZERO


def test_fields_are_independent():
    floors = {"a": ZERO, "b": ZERO}
    alloc, _left = auction(m(cpu=8, gpu0=16), {"s": parse("a & b")}, floors,
                           {"a": 50, "b": 50})
    assert alloc["a"] == m(cpu=4, gpu0=8)
    assert alloc["b"] == m(cpu=4, gpu0=8)


def test_infeasible_floor_is_an_error():
    with pytest.raises(ValueError, match="exceeds available"):
        auction(m(gpu0=8), {"s": parse("a & b")},
                {"a": m(gpu0=5), "b": m(gpu0=5)}, {"a": 50, "b": 50})
