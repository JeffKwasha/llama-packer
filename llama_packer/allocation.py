"""The allocation algebra: floors, set expressions, and the auction.

The authority is ``docs/allocation.md``. This module is pure arithmetic over
``Memory`` tuples; it does not read profiles, models, or llama-swap.
"""

from __future__ import annotations

import re
from collections import namedtuple
from dataclasses import dataclass

Memory = namedtuple("Memory", ["cpu", "gpu0", "gpu1"])
ZERO = Memory(0, 0, 0)
MAX_ROUNDS = 32

_TOKEN = re.compile(r"[A-Za-z0-9._:-]+|[&|()]")


def add(a: Memory, b: Memory) -> Memory:
    return Memory(*(x + y for x, y in zip(a, b)))


def maximum(a: Memory, b: Memory) -> Memory:
    return Memory(*(max(x, y) for x, y in zip(a, b)))


def subtract(a: Memory, b: Memory) -> Memory:
    return Memory(*(x - y for x, y in zip(a, b)))


def add_scaled(a: Memory, factor: float) -> Memory:
    return Memory(*(int(x * factor) for x in a))


@dataclass(frozen=True)
class Leaf:
    name: str


@dataclass(frozen=True)
class And:
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Or:
    left: "Expr"
    right: "Expr"


Expr = Leaf | And | Or


class _Parser:
    def __init__(self, text: str):
        self.tokens = _TOKEN.findall(text)
        self.pos = 0

    def parse(self) -> Expr:
        expr = self._or()
        if self.pos != len(self.tokens):
            raise ValueError(f"unexpected token {self.tokens[self.pos]!r}")
        return expr

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of set expression")
        self.pos += 1
        return token

    def _or(self) -> Expr:
        left = self._and()
        while self._peek() == "|":
            self._take()
            left = Or(left, self._and())
        return left

    def _and(self) -> Expr:
        left = self._atom()
        while self._peek() == "&":
            self._take()
            left = And(left, self._atom())
        return left

    def _atom(self) -> Expr:
        token = self._take()
        if token == "(":
            expr = self._or()
            if self._take() != ")":
                raise ValueError("missing ')' in set expression")
            return expr
        if token in {"&", "|", ")"}:
            raise ValueError(f"unexpected {token!r}")
        return Leaf(token)


def parse(text: str) -> Expr:
    return _Parser(text).parse()


def leaves(expr: Expr) -> tuple[str, ...]:
    if isinstance(expr, Leaf):
        return (expr.name,)
    return leaves(expr.left) + leaves(expr.right)


def floor_of(expr: Expr, floors: dict[str, Memory]) -> Memory:
    if isinstance(expr, Leaf):
        try:
            return floors[expr.name]
        except KeyError:
            raise KeyError(f"set leaf {expr.name!r} has no floor") from None
    if isinstance(expr, And):
        return add(floor_of(expr.left, floors),
                   floor_of(expr.right, floors))
    return maximum(floor_of(expr.left, floors), floor_of(expr.right, floors))


def _offer(auction: Memory, members: list[str], weights: dict[str, float],
           floors: dict[str, Memory], fixed: dict[str, Memory]) -> dict[str, Memory]:
    free = [r for r in members if r not in fixed and weights.get(r, 0.0) > 0]
    total = sum(weights.get(r, 0.0) for r in free)
    fixed_used = ZERO
    for role, amount in fixed.items():
        fixed_used = add(fixed_used, subtract(amount, floors[role]))
    shared = Memory(*(max(0, x) for x in subtract(auction, fixed_used)))
    out = {r: floors[r] for r in members}
    for role in members:
        if role in fixed:
            out[role] = fixed[role]
        elif total > 0 and role in free:
            part = add_scaled(shared, weights[role] / total)
            out[role] = add(floors[role], part)
    return out


def auction(available: Memory, sets: dict[str, Expr],
            floors: dict[str, Memory], weights: dict[str, float],
            accepts: frozenset[str] = frozenset()
            ) -> tuple[dict[str, Memory], dict[str, Memory]]:
    """Allocate ``available`` across roles by the docs/allocation.md auction.

    Each set is one budget. Floors come out first, the rest is split by
    ``weights``. A role accepts the minimum offer across its sets. When one
    set offers less than another requested, the offering set fixes that role
    and renormalises the remainder. Ties resolve by ascending floor, then
    name, so the fixed point is deterministic.
    """
    members = {name: leaves(expr) for name, expr in sets.items()}
    ceilings = {name: subtract(available, floor_of(expr, floors))
                for name, expr in sets.items()}
    for name, ceiling in ceilings.items():
        if any(x < 0 for x in ceiling):
            raise ValueError(f"set {name!r} floor exceeds available")

    def order(roles):
        return sorted(roles, key=lambda r: (floors[r], r))

    offers = {name: _offer(ceilings[name], order(members[name]), weights,
                           floors, {}) for name in sets}
    alloc: dict[str, Memory] = {}
    for role in {r for rs in members.values() for r in rs}:
        values = [offers[name][role] for name in sets if role in members[name]]
        alloc[role] = values[0] if len(values) == 1 else Memory(
            *(min(v[i] for v in values) for i in range(3)))

    for _ in range(MAX_ROUNDS):
        free = {
            name: _offer(ceilings[name], order(members[name]), weights,
                         floors, {})
            for name in sets
        }
        fixed = {
            name: {r: alloc[r] for r in members[name]
                   if alloc[r] != free[name][r]}
            for name in sets
        }
        offers = {
            name: _offer(ceilings[name], order(members[name]), weights,
                         floors, fixed[name])
            for name in sets
        }
        nxt = {}
        for role in alloc:
            values = [offers[name][role] for name in sets
                      if role in members[name]]
            nxt[role] = values[0] if len(values) == 1 else Memory(
                *(min(v[i] for v in values) for i in range(3)))
        if nxt == alloc:
            break
        alloc = nxt
    else:
        raise ValueError("auction did not converge")

    unallocated = {}
    for name in sets:
        used = ZERO
        for role in members[name]:
            used = add(used, subtract(alloc[role], floors[role]))
        unallocated[name] = subtract(ceilings[name], used)

    for name in sets:
        idle = [r for r in members[name]
                if r in accepts and weights.get(r, 0.0) == 0.0]
        left = unallocated[name]
        if not idle or all(x <= 0 for x in left):
            continue
        share = Memory(*(x // len(idle) for x in left))
        for role in order(idle):
            alloc[role] = add(alloc[role], share)
        unallocated[name] = subtract(left, add_scaled(share, len(idle)))

    return alloc, unallocated
