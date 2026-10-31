"""Roles: the scheduling table for the kinds of work a model can do.

A role is a kind of work with exactly one name.  What an engine serves is
declared as *endpoints* (``endpoints.py``): a task maps to the one endpoint
it is served at, and compatibility is that intersection.  Nothing here names
an engine, an endpoint or an operation.

Everything derived from a role — the served set, the chat-pass set, the
``device_roles`` set, the matrix context knob — comes from this one table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    """One kind of work a model can do."""

    name: str
    #: Participates in the chat passes (mmproj keep/drop, shared solve, …).
    chat: bool
    #: ``pinned`` (always resident), ``optional`` (resident when declared),
    #: ``demand`` (never pinned).
    residency: str
    #: ``MatrixKnobs`` attribute holding this role's served context, if any.
    ctx_knob: str | None
    #: May be named in ``profiles.yaml device_roles:``.
    device_role: bool


ROLES: tuple[Role, ...] = (
    Role("chat", True, "demand", None, False),
    Role("embeddings", False, "pinned", "embed_context", True),
    Role("rerank", False, "pinned", "rerank_context", True),
    Role("image", False, "demand", None, False),
    Role("s2t", False, "optional", None, True),
    Role("t2s", False, "optional", None, True),
)

BY_NAME: dict[str, Role] = {r.name: r for r in ROLES}

SERVED_ROLES: tuple[str, ...] = tuple(r.name for r in ROLES)
NON_CHAT_ROLES: frozenset[str] = frozenset(r.name for r in ROLES if not r.chat)
DEVICE_ROLES: frozenset[str] = frozenset(r.name for r in ROLES if r.device_role)


def get(name: object) -> Role | None:
    """The role called *name*, or None when it is not a served role."""
    return BY_NAME.get(name) if isinstance(name, str) else None


