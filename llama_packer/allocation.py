"""The allocation algebra: floors, set expressions, and the auction.

The authority is ``docs/allocation.md``. The auction runs on one device pool
at a time: a role vies for the pool its model runs on (``gpu0``, ``cpu``,
...). The other fields of the cost tuple are static charges and capacities,
not memory to be shared out.
"""

from __future__ import annotations

import re
from collections import namedtuple
from dataclasses import dataclass

MAX_ROUNDS = 32

_TOKEN = re.compile(r"[A-Za-z0-9._:-]+|[&|()]")


class PoolSchema:
    """The host's pool ids, in order, ``cpu`` first.

    ``memory`` is a namedtuple class, so every cost names its fields.
    """

    def __init__(self, names):
        self.names = tuple(names)
        if not self.names or self.names[0] != "cpu":
            raise ValueError("the pool tuple starts with cpu")
        self.memory = namedtuple("Memory", self.names)

    def __call__(self, *values):
        return self.memory(*values)

    def zero(self):
        return self.memory(*([0] * len(self.names)))


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


def floor_of(expr: Expr, floors: dict[str, int]) -> int:
    if isinstance(expr, Leaf):
        try:
            return floors[expr.name]
        except KeyError:
            raise KeyError(f"set leaf {expr.name!r} has no floor") from None
    if isinstance(expr, And):
        return floor_of(expr.left, floors) + floor_of(expr.right, floors)
    return max(floor_of(expr.left, floors), floor_of(expr.right, floors))


def effective_weights(members, weights, fills):
    """The weight each claimant uses in the normalised split.

    A fill weighs what the weighted roles leave below 100, split equally
    among fills. If the weighted roles already claim 100 or more, the fill
    weighs 0 and becomes an acceptor of any remainder a capped role frees.
    """
    weighted_sum = sum(weights.get(r, 0.0) for r in members
                       if r not in fills)
    out = {r: weights.get(r, 0.0) for r in members}
    free_fill = [r for r in members if r in fills]
    if free_fill:
        each = (100.0 - weighted_sum) / len(free_fill) \
            if weighted_sum < 100.0 else 0.0
        for role in free_fill:
            out[role] = each
    return out


def _offer(budget: int, members, weights, floors, fills, fixed):
    free = [r for r in members if r not in fixed]
    effective = effective_weights(free, weights, fills)
    total = sum(effective.values())

    fixed_used = sum(amount - floors[role]
                     for role, amount in fixed.items())
    shared = max(0, budget - fixed_used)
    out = {r: floors[r] for r in members}
    for role, amount in fixed.items():
        out[role] = amount
    if total > 0:
        for role in free:
            if effective[role] > 0:
                out[role] = floors[role] + int(
                    shared * effective[role] / total)
    return out


def auction(budget: int, sets: dict[str, Expr], floors: dict[str, int],
            weights: dict[str, float], fills: frozenset[str] = frozenset(),
            accepts: frozenset[str] = frozenset()
            ) -> tuple[dict[str, int], dict[str, int]]:
    """Allocate one pool by the docs/allocation.md auction.

    A set is one budget. Floors come out first. Weighted roles divide the
    rest, then `fill` roles take the remainder, then `0` roles take what is
    left. A role accepts the minimum offer across its sets; a set that offers
    less than another requested fixes that role and renormalises. Ties
    resolve by ascending floor, then name, so the fixed point is
    deterministic.
    """
    members = {name: leaves(expr) for name, expr in sets.items()}
    ceilings = {name: budget - floor_of(expr, floors)
                for name, expr in sets.items()}
    for name, ceiling in ceilings.items():
        if ceiling < 0:
            raise ValueError(f"set {name!r} floor exceeds available")

    def order(roles):
        return sorted(roles, key=lambda r: (floors[r], r))

    offers = {name: _offer(ceilings[name], order(members[name]), weights,
                           floors, fills, {}) for name in sets}
    alloc = {}
    for role in {r for rs in members.values() for r in rs}:
        values = [offers[name][role] for name in sets if role in members[name]]
        alloc[role] = min(values)

    for _ in range(MAX_ROUNDS):
        free = {name: _offer(ceilings[name], order(members[name]), weights,
                             floors, fills, {}) for name in sets}
        fixed = {
            name: {r: alloc[r] for r in members[name]
                   if alloc[r] < free[name][r]}
            for name in sets
        }
        offers = {
            name: _offer(ceilings[name], order(members[name]), weights,
                         floors, fills, fixed[name])
            for name in sets
        }
        nxt = {}
        for role in alloc:
            values = [offers[name][role] for name in sets
                      if role in members[name]]
            nxt[role] = min(values)
        if nxt == alloc:
            break
        alloc = nxt
    else:
        raise ValueError("auction did not converge")

    unallocated = {}
    for name in sets:
        used = sum(alloc[role] - floors[role] for role in members[name])
        unallocated[name] = ceilings[name] - used

    for name in sets:
        idle = [r for r in members[name]
                if r in accepts and (r in fills
                                     or weights.get(r, 0.0) == 0.0)]
        left = unallocated[name]
        if not idle or left <= 0:
            continue
        share = left // len(idle)
        for role in order(idle):
            alloc[role] += share
        unallocated[name] = left - share * len(idle)

    return alloc, unallocated
