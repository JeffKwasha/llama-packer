"""Floors, set expressions and the auction. See docs/allocation.md."""

from __future__ import annotations

import logging
import math
import re
from collections import namedtuple
from dataclasses import dataclass

log = logging.getLogger(__name__)


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


#: The weight-space whole; docs/allocation.md §The auction.
WEIGHT_WHOLE = 100.0


def resolve_shares(members, shares, set_name=""):
    """Claims -> weights, fills, accepts for one set; docs/allocation.md."""
    weights, fills, accepts = {}, set(), set()
    for name in members:
        share = shares.get(name)
        kind = share.kind if share is not None else ""
        weights[name] = share.weight if share is not None and share.claims else 0.0
        if kind == "fill":
            fills.add(name)
        if share is not None and share.accepts:
            accepts.add(name)
    if fills:
        raw = sorted(n for n in members
                     if (s := shares.get(n)) is not None
                     and s.kind == "weight" and weights[n] > 0)
        if raw:
            log.warning(
                "set %r: fill roles [%s] mixed with raw weights [%s]! "
                "use '%%' shares with fill",
                set_name, " ".join(sorted(fills)), " ".join(raw))
        each = max(0.0, WEIGHT_WHOLE - sum(
            weights[n] for n in members if n not in fills)) / len(fills)
        for name in fills:
            weights[name] = each
    return weights, frozenset(fills), frozenset(accepts)


def auction(budget: int, sets: dict[str, Expr], floors: dict[str, int],
            shares: dict) -> tuple[dict[str, int], dict[str, int]]:
    """Allocate each set, take the minimum per role; docs/allocation.md."""
    members = {name: leaves(expr) for name, expr in sets.items()}
    ceilings = {name: budget - floor_of(expr, floors)
                for name, expr in sets.items()}
    for name, ceiling in ceilings.items():
        if ceiling < 0:
            raise ValueError(f"set {name!r} floor exceeds available")

    def order(roles):
        return sorted(roles, key=lambda r: (floors[r], r))

    offers = {}
    for name, roles in members.items():
        roles = order(roles)
        available = ceilings[name]
        weights, fills, accepts = resolve_shares(roles, shares, name)
        total = sum(weights.values())
        alloc = {role: floors[role] for role in roles}
        if total > 0:
            for role in roles:
                if weights[role] > 0:
                    alloc[role] = floors[role] + int(
                        available * weights[role] / total)
        left = available - sum(alloc[role] - floors[role] for role in roles)
        if left > 0:
            fill_accept = [r for r in roles if r in fills and r in accepts]
            idle = [r for r in roles if r in accepts and weights[r] == 0.0]
            recipients = fill_accept or idle
            if recipients:
                part = left // len(recipients)
                for role in order(recipients):
                    alloc[role] += part
        offers[name] = alloc

    alloc = {}
    for role in {r for roles in members.values() for r in roles}:
        alloc[role] = min(offers[name][role] for name in members
                          if role in members[name])

    unallocated = {}
    for name, roles in members.items():
        used = sum(offers[name][role] - floors[role] for role in roles)
        unallocated[name] = ceilings[name] - used
    return alloc, unallocated


def floor_mb(quad: tuple[int, float, float, int], min_context: int,
             min_parallel: int, margin: float = 0.0) -> int:
    """One serving's floor in MiB from its affine quad.

    ``quad`` is ``(model_mib, kv_per_token, slot_mib, compute_mib)``.
    """
    model_mib, kv, slot, compute = quad
    raw = (model_mib + compute + kv * min_context * min_parallel
           + slot * min_parallel)
    return math.ceil(raw * (1.0 + margin))


def role_floors(entries: dict[str, list[tuple]], margin: float = 0.0
                ) -> dict[str, int]:
    """Per-role floor: the maximum floor of the role's model/task pairs.

    ``entries`` maps a role to ``(quad, min_context, min_parallel)`` triples.
    """
    return {
        role: max(floor_mb(quad, ctx, parallel, margin)
                  for quad, ctx, parallel in demands)
        for role, demands in entries.items()
    }
